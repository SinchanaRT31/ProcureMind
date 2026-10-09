import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pytest
import shap

import ml.explain as explain_module
from ml.explain import ADDITIVITY_ATOL, _explain_features, explain_predictions
from ml.predict import predict_procurement_data
from ml.preprocess import ProcurementFeatureTransformer
from sklearn.ensemble import IsolationForest
from test_preprocess import raw_frame


def _assert_rng_states_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_explain_predictions_returns_top_feature_contributions():
    raw = raw_frame()
    train = raw.iloc[:3]
    test = raw.iloc[3:]
    transformer = ProcurementFeatureTransformer().fit(train)
    model = IsolationForest(n_estimators=20, random_state=42)
    model.fit(transformer.transform(train))

    background = transformer.transform(train)
    explanation = explain_predictions(test, model=model, transformer=transformer, background_features=background, top_k=2)

    expected = {
        "transaction_id",
        "top_feature_1",
        "top_contribution_1",
        "top_feature_2",
        "top_contribution_2",
    }
    assert expected.issubset(set(explanation.columns))
    assert len(explanation) == len(test)


def test_shap_reconstructs_exact_anomaly_score_and_preserves_feature_order():
    raw = raw_frame()
    transformer = ProcurementFeatureTransformer().fit(raw.iloc[:3])
    train_features = transformer.transform(raw.iloc[:3])
    model = IsolationForest(n_estimators=20, random_state=42).fit(train_features)
    rows = transformer.transform(raw.iloc[3:])
    values, bases = _explain_features(rows, model=model, background_features=train_features, batch_size=2)
    scores = -model.score_samples(rows)
    assert values.shape == (len(rows), len(transformer.feature_names_))
    assert np.allclose(bases + values.sum(axis=1), scores, atol=ADDITIVITY_ATOL, rtol=1e-6)
    assert list(rows.columns) == transformer.feature_names_
    # Efficiency axiom gives signed contributions relative to the baseline.
    assert np.allclose(values.sum(axis=1), scores - bases, atol=ADDITIVITY_ATOL, rtol=1e-6)


def test_repeated_explanations_are_reproducible_and_restore_numpy_rng_state():
    raw = raw_frame()
    train, row = raw.iloc[:3], raw.iloc[[3]]
    transformer = ProcurementFeatureTransformer().fit(train)
    background = transformer.transform(train)
    model = IsolationForest(n_estimators=20, random_state=42).fit(background)

    original_state = np.random.get_state()
    try:
        np.random.seed(8675309)
        expected_state = np.random.get_state()
        first = explain_predictions(row, model=model, transformer=transformer, background_features=background)
        _assert_rng_states_equal(np.random.get_state(), expected_state)
        second = explain_predictions(row, model=model, transformer=transformer, background_features=background)
        _assert_rng_states_equal(np.random.get_state(), expected_state)

        feature_columns = [f"top_feature_{rank}" for rank in range(1, 4)]
        contribution_columns = [f"top_contribution_{rank}" for rank in range(1, 4)]
        assert first[feature_columns].values.tolist() == second[feature_columns].values.tolist()
        np.testing.assert_allclose(
            first[contribution_columns].to_numpy(dtype=float),
            second[contribution_columns].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
    finally:
        np.random.set_state(original_state)


def test_numpy_rng_state_is_restored_when_shap_explanation_raises(monkeypatch):
    raw = raw_frame()
    transformer = ProcurementFeatureTransformer().fit(raw.iloc[:3])
    features = transformer.transform(raw.iloc[[3]])
    background = transformer.transform(raw.iloc[:3])
    model = IsolationForest(n_estimators=10, random_state=42).fit(background)

    original_state = np.random.get_state()
    try:
        np.random.seed(12345)
        expected_state = np.random.get_state()

        def failing_explainer(*args, seed=None, **kwargs):
            np.random.seed(seed)
            raise RuntimeError("simulated SHAP failure")

        monkeypatch.setattr(explain_module.shap, "Explainer", failing_explainer)
        with pytest.raises(RuntimeError, match="simulated SHAP failure"):
            _explain_features(features, model=model, background_features=background, batch_size=1)
        _assert_rng_states_equal(np.random.get_state(), expected_state)
    finally:
        np.random.set_state(original_state)


def test_concurrent_explanations_are_reproducible_and_restore_numpy_rng_state():
    raw = raw_frame()
    train, row = raw.iloc[:3], raw.iloc[[3]]
    transformer = ProcurementFeatureTransformer().fit(train)
    background = transformer.transform(train)
    model = IsolationForest(n_estimators=20, random_state=42).fit(background)

    original_state = np.random.get_state()
    try:
        np.random.seed(24680)
        expected_state = np.random.get_state()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(
                lambda _: explain_predictions(row, model=model, transformer=transformer, background_features=background),
                range(4),
            ))
        _assert_rng_states_equal(np.random.get_state(), expected_state)

        feature_columns = [f"top_feature_{rank}" for rank in range(1, 4)]
        contribution_columns = [f"top_contribution_{rank}" for rank in range(1, 4)]
        for result in results[1:]:
            assert result[feature_columns].values.tolist() == results[0][feature_columns].values.tolist()
            np.testing.assert_allclose(
                result[contribution_columns].to_numpy(dtype=float),
                results[0][contribution_columns].to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
    finally:
        np.random.set_state(original_state)


def test_permutation_shap_sign_tracks_known_score_direction():
    background = np.array([[0.0, 0.0], [1.0, 1.0]])
    row = np.array([[3.0, -2.0]])
    explainer = shap.Explainer(
        lambda values: values[:, 0] + values[:, 1],
        shap.maskers.Independent(background), algorithm="permutation",
    )
    result = explainer(row, max_evals=5)
    assert result.values[0, 0] > 0  # Feature 0 raises the explained score.
    assert result.values[0, 1] < 0  # Feature 1 lowers the explained score.


def test_duplicate_indices_keep_transaction_ids_in_input_order():
    raw = raw_frame().iloc[[0, 1]].copy()
    raw.index = [7, 7]
    transformer = ProcurementFeatureTransformer().fit(raw_frame().iloc[:3])
    features = transformer.transform(raw_frame().iloc[:3])
    model = IsolationForest(n_estimators=10, random_state=42).fit(features)
    result = explain_predictions(raw, model=model, transformer=transformer, background_features=features)
    assert result.transaction_id.tolist() == raw.transaction_id.tolist()


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        explain_predictions(raw_frame(), model=object(), transformer=object(), background_features=pd.DataFrame(), top_k=top_k)


def test_forbidden_outcome_fields_do_not_enter_feature_matrix():
    raw = raw_frame().drop(columns="is_anomaly").copy()
    transformer = ProcurementFeatureTransformer().fit(raw_frame().iloc[:3])
    baseline = transformer.transform(raw)
    for name in ["anomaly_type", "risk_score", "risk_level", "investigation_status", "potential_savings"]:
        raw[name] = "sensitive"
    with_outcomes = transformer.transform(raw)
    assert baseline.equals(with_outcomes)
    forbidden = {"is_anomaly", "anomaly_type", "risk_score", "risk_level", "investigation_status", "potential_savings"}
    assert not forbidden & set(transformer.feature_names_)


def test_prediction_cli_supports_explanations(tmp_path: Path):
    raw = raw_frame()
    train = raw.iloc[:3]
    test = raw.iloc[3:]
    transformer = ProcurementFeatureTransformer().fit(train)
    model = IsolationForest(n_estimators=20, random_state=42)
    model.fit(transformer.transform(train))
    model_path = tmp_path / "model.pkl"
    transformer_path = tmp_path / "transformer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(transformer, transformer_path)

    background_path = tmp_path / "background.pkl"
    joblib.dump(transformer.transform(train), background_path)
    default_result = predict_procurement_data(test.drop(columns="is_anomaly"), model_path, transformer_path)
    result = predict_procurement_data(test, model_path, transformer_path, include_explanations=True, explanation_background_path=background_path)

    assert "transaction_id" in result.columns
    assert any("top_feature_" in column for column in result.columns)
    assert default_result.columns.tolist() == ["transaction_id", "predicted_anomaly", "ml_anomaly_score"]
    assert np.array_equal(default_result.predicted_anomaly.to_numpy(), result.predicted_anomaly.to_numpy())
    assert np.allclose(default_result.ml_anomaly_score, result.ml_anomaly_score)
