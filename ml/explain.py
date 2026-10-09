from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

if __package__ in {None, ""}:  # pragma: no cover - CLI compatibility
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocess import TRANSACTION_ID_COLUMN, clean_data
from ml.train import EXPLANATION_BACKGROUND_PATH, MODEL_PATH, TRANSFORMER_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "ml" / "model" / "shap_explanations.csv"
ADDITIVITY_ATOL = 1e-6
DEFAULT_BATCH_SIZE = 10


def _explain_features(
    features: pd.DataFrame,
    *,
    model: object,
    background_features: pd.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Explain f(X)=-IsolationForest.score_samples(X) with permutation SHAP."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    expected_columns = list(features.columns)
    if list(background_features.columns) != expected_columns:
        raise ValueError("Background feature order does not match the fitted transformer feature order.")

    background = background_features.to_numpy(dtype=float)

    def anomaly_score(values: np.ndarray) -> np.ndarray:
        # SHAP passes arrays in the exact feature order established above.
        return -np.asarray(model.score_samples(pd.DataFrame(values, columns=expected_columns)), dtype=float)

    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(anomaly_score, masker, algorithm="permutation")
    feature_array = features.to_numpy(dtype=float)
    value_batches: list[np.ndarray] = []
    base_batches: list[np.ndarray] = []
    max_evals = 2 * len(expected_columns) + 1
    for start in range(0, len(feature_array), batch_size):
        result = explainer(feature_array[start : start + batch_size], max_evals=max_evals)
        values = np.asarray(result.values, dtype=float)
        bases = np.asarray(result.base_values, dtype=float).reshape(-1)
        scores = anomaly_score(feature_array[start : start + batch_size])
        if values.shape != (len(scores), len(expected_columns)):
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}.")
        if bases.size == 1:
            bases = np.repeat(bases, len(scores))
        if bases.shape != scores.shape:
            raise ValueError("SHAP baseline shape does not match the explained rows.")
        if not np.allclose(bases + values.sum(axis=1), scores, rtol=1e-6, atol=ADDITIVITY_ATOL):
            raise ValueError("Permutation SHAP values failed the anomaly-score additivity check.")
        value_batches.append(values)
        base_batches.append(bases)
    if not value_batches:
        return np.empty((0, len(expected_columns))), np.empty((0,))
    return np.concatenate(value_batches), np.concatenate(base_batches)


def explain_predictions(
    raw_data: pd.DataFrame,
    *,
    model: object | None = None,
    transformer: object | None = None,
    model_path: str | Path = MODEL_PATH,
    transformer_path: str | Path = TRANSFORMER_PATH,
    background_path: str | Path = EXPLANATION_BACKGROUND_PATH,
    background_features: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    top_k: int = 3,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Explain -model.score_samples in fitted transformer feature order.

    Positive SHAP values increase the model's anomaly score relative to the
    training-reference baseline; negative values decrease it. The default
    reference is a persisted, deterministic sample of transformed training rows.
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    cleaned = clean_data(raw_data)
    if model is None or transformer is None:
        if not Path(model_path).exists() or not Path(transformer_path).exists():
            raise FileNotFoundError("Model artifacts are missing. Run `python ml/train.py` first.")
        model = joblib.load(model_path)
        transformer = joblib.load(transformer_path)
    if background_features is None:
        if not Path(background_path).exists():
            raise FileNotFoundError(
                "Safe SHAP reference features are missing. Run `python ml/train.py` to create "
                "the training-only explanation background."
            )
        background_features = joblib.load(background_path)

    if features is None:
        features = transformer.transform(cleaned)
    expected_columns = list(transformer.feature_names_)
    features = features.loc[:, expected_columns]
    background_features = background_features.loc[:, expected_columns]
    shap_values, _ = _explain_features(
        features,
        model=model,
        background_features=background_features,
        batch_size=batch_size,
    )

    rows: list[dict[str, object]] = []
    # Use positional access: duplicate or non-monotonic DataFrame indices are safe.
    for row_number, contribution_row in enumerate(shap_values):
        ranked_indices = np.argsort(-np.abs(contribution_row), kind="stable")[:top_k]
        payload: dict[str, object] = {
            "transaction_id": str(cleaned.iloc[row_number][TRANSACTION_ID_COLUMN]),
            "top_explanation": "; ".join(
                f"{expected_columns[column]}={contribution_row[column]:.4f}" for column in ranked_indices
            ),
        }
        for rank, column in enumerate(ranked_indices, start=1):
            payload[f"top_feature_{rank}"] = expected_columns[column]
            payload[f"top_contribution_{rank}"] = float(contribution_row[column])
        for rank in range(len(ranked_indices) + 1, top_k + 1):
            payload[f"top_feature_{rank}"] = ""
            payload[f"top_contribution_{rank}"] = np.nan
        rows.append(payload)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for raw ProcureMind transactions.")
    parser.add_argument("input_csv", type=Path, help="Path to a raw procurement CSV file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination CSV for SHAP explanations.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of strongest feature contributions per transaction.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Rows explained per SHAP call; all input rows are processed.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--transformer-path", type=Path, default=TRANSFORMER_PATH)
    parser.add_argument("--background-path", type=Path, default=EXPLANATION_BACKGROUND_PATH)
    args = parser.parse_args()

    result = explain_predictions(
        pd.read_csv(args.input_csv), model_path=args.model_path,
        transformer_path=args.transformer_path, background_path=args.background_path,
        top_k=args.top_k, batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved SHAP explanations for {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
