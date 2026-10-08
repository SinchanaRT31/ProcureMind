from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Preserve direct CLI use (``python ml/train.py``) while serializing classes under
# their canonical ``ml.*`` module paths for portable joblib artifacts.
if __package__ in {None, ""}:  # pragma: no cover - CLI compatibility
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.evidence import ProcurementEvidenceEngine, evidence_evaluation
from ml.preprocess import ProcurementFeatureTransformer, TARGET_COLUMN, TRANSACTION_ID_COLUMN, preprocess_data
from ml.risk import ProcurementRiskEngine, risk_evaluation
from ml.verification import ProcurementVerificationEngine, merge_evidence_signals, temporal_vendor_cadence_experiment, verification_evaluation

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "ml" / "model"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
TRANSFORMER_PATH = MODEL_DIR / "procurement_feature_transformer.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
PREDICTIONS_PATH = MODEL_DIR / "test_predictions.csv"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
SPLIT_METADATA_PATH = MODEL_DIR / "split_metadata.json"
EVIDENCE_ENGINE_PATH = MODEL_DIR / "procurement_evidence_engine.pkl"
EVIDENCE_PREDICTIONS_PATH = MODEL_DIR / "test_evidence.csv"
EVIDENCE_EVALUATION_PATH = MODEL_DIR / "evidence_evaluation.json"
RISK_ENGINE_PATH = MODEL_DIR / "procurement_risk_engine.pkl"
RISK_PREDICTIONS_PATH = MODEL_DIR / "test_risk_assessments.csv"
RISK_EVALUATION_PATH = MODEL_DIR / "risk_evaluation.json"
VERIFICATION_ENGINE_PATH = MODEL_DIR / "procurement_verification_engine.pkl"
VERIFICATION_PREDICTIONS_PATH = MODEL_DIR / "test_verification.csv"
CADENCE_EXPERIMENT_PATH = MODEL_DIR / "temporal_cadence_experiment.json"
VERIFICATION_EVALUATION_PATH = MODEL_DIR / "verification_evaluation.json"
TEST_SIZE, RANDOM_STATE = 0.20, 42


def evaluate_model(y_true: pd.Series, predicted_anomaly: pd.Series) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_anomaly, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predicted_anomaly, zero_division=0),
        "recall": recall_score(y_true, predicted_anomaly, zero_division=0),
        "f1_score": f1_score(y_true, predicted_anomaly, zero_division=0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "anomaly_detection_rate": float(predicted_anomaly.mean()),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def train_model(X_train: pd.DataFrame) -> tuple[IsolationForest, float]:
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    start = time.perf_counter()
    model.fit(X_train)
    return model, time.perf_counter() - start


def main() -> None:
    raw = preprocess_data()
    train_raw, test_raw = train_test_split(raw, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=raw[TARGET_COLUMN])
    transformer = ProcurementFeatureTransformer().fit(train_raw)
    X_train, X_test = transformer.transform(train_raw), transformer.transform(test_raw)
    model, training_time = train_model(X_train)
    predicted = pd.Series((model.predict(X_test) == -1).astype(int), index=test_raw.index)
    scores = pd.Series(-model.score_samples(X_test), index=test_raw.index)
    metrics = evaluate_model(test_raw[TARGET_COLUMN], predicted)
    metrics["training_time_seconds"] = training_time
    evidence_engine = ProcurementEvidenceEngine().fit(train_raw)
    evidence = evidence_engine.assess(test_raw, ml_anomaly_scores=scores)
    verification_engine = ProcurementVerificationEngine().fit(train_raw)
    verification = verification_engine.assess(test_raw)
    evidence = merge_evidence_signals(evidence, verification)
    evidence.insert(0, "transaction_id", test_raw[TRANSACTION_ID_COLUMN].astype(str))
    evidence_summary = evidence_evaluation(evidence, predicted)
    evidence_summary["phase_1_ml_baseline"] = metrics
    risk_engine = ProcurementRiskEngine().fit(-model.score_samples(X_train))
    risk = risk_engine.assess(evidence, predicted, scores)
    risk.insert(0, "transaction_id", test_raw[TRANSACTION_ID_COLUMN].astype(str))
    risk_summary = risk_evaluation(risk, test_raw[TARGET_COLUMN])
    risk_summary["phase_1_ml_baseline"] = metrics
    risk_summary["priority_policy"] = risk_engine.priority_policy_
    risk_summary["ml_score_thresholds"] = risk_engine.ml_score_thresholds_
    cadence_experiment = temporal_vendor_cadence_experiment(raw)
    verification_summary = verification_evaluation(verification, risk, predicted, evidence)
    metadata = {"strategy": "stratified_random_split", "test_size": TEST_SIZE, "random_state": RANDOM_STATE, "train_rows": len(train_raw), "test_rows": len(test_raw), "train_anomaly_rate": float(train_raw[TARGET_COLUMN].mean()), "test_anomaly_rate": float(test_raw[TARGET_COLUMN].mean())}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(transformer, TRANSFORMER_PATH)
    joblib.dump(evidence_engine, EVIDENCE_ENGINE_PATH)
    joblib.dump(risk_engine, RISK_ENGINE_PATH)
    joblib.dump(verification_engine, VERIFICATION_ENGINE_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(transformer.feature_names_, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SPLIT_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    EVIDENCE_EVALUATION_PATH.write_text(json.dumps(evidence_summary, indent=2), encoding="utf-8")
    RISK_EVALUATION_PATH.write_text(json.dumps(risk_summary, indent=2), encoding="utf-8")
    CADENCE_EXPERIMENT_PATH.write_text(json.dumps(cadence_experiment, indent=2), encoding="utf-8")
    VERIFICATION_EVALUATION_PATH.write_text(json.dumps(verification_summary, indent=2), encoding="utf-8")
    pd.DataFrame({"transaction_id": test_raw[TRANSACTION_ID_COLUMN].astype(str), "predicted_anomaly": predicted, "ml_anomaly_score": scores, "actual_anomaly": test_raw[TARGET_COLUMN].astype(int)}).to_csv(PREDICTIONS_PATH, index=False)
    evidence.to_csv(EVIDENCE_PREDICTIONS_PATH, index=False)
    risk.to_csv(RISK_PREDICTIONS_PATH, index=False)
    verification.to_csv(VERIFICATION_PREDICTIONS_PATH, index=False)
    print(f"Train rows: {len(train_raw)} | Test rows: {len(test_raw)}")
    print(f"Features ({len(transformer.feature_names_)}): {transformer.feature_names_}")
    print(f"Training time (seconds): {training_time:.4f}")
    for name, value in metrics.items(): print(f"{name}: {value}")
    print(f"Saved evidence evaluation: {EVIDENCE_EVALUATION_PATH}")
    print(f"Saved risk evaluation: {RISK_EVALUATION_PATH}")
    print(f"Saved cadence experiment: {CADENCE_EXPERIMENT_PATH}")
    print(f"Saved verification evaluation: {VERIFICATION_EVALUATION_PATH}")


if __name__ == "__main__":
    main()
