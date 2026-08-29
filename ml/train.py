from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from preprocess import preprocess_data
except ImportError:  # pragma: no cover
    from ml.preprocess import preprocess_data


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "ml" / "model"
MODEL_PATH = MODEL_DIR / "isolation_forest.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
PREDICTIONS_PATH = MODEL_DIR / "predictions.csv"


def train_model(X: pd.DataFrame) -> tuple[IsolationForest, pd.Series, pd.Series, float]:
    """Fit Isolation Forest on feature data and return predictions and scores."""
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    start_time = time.perf_counter()
    model.fit(X)
    training_time = time.perf_counter() - start_time

    raw_predictions = model.predict(X)
    predicted_anomaly = pd.Series((raw_predictions == -1).astype(int), index=X.index)
    anomaly_scores = pd.Series(-model.score_samples(X), index=X.index, name="anomaly_score")

    return model, predicted_anomaly, anomaly_scores, training_time


def evaluate_model(
    y_true: pd.Series, predicted_anomaly: pd.Series
) -> dict[str, object]:
    """Evaluate predictions against ground-truth labels after inference."""
    metrics = {
        "precision": precision_score(y_true, predicted_anomaly, zero_division=0),
        "recall": recall_score(y_true, predicted_anomaly, zero_division=0),
        "f1_score": f1_score(y_true, predicted_anomaly, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, predicted_anomaly),
        "classification_report": classification_report(
            y_true, predicted_anomaly, zero_division=0
        ),
    }
    return metrics


def save_model(model: IsolationForest, feature_names: list[str]) -> None:
    """Persist the trained model and exact feature order."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURE_NAMES_PATH.write_text(json.dumps(feature_names, indent=2))


def save_predictions(
    transaction_ids: pd.Series,
    predicted_anomaly: pd.Series,
    anomaly_scores: pd.Series,
    actual_anomaly: pd.Series,
) -> None:
    """Save prediction outputs for downstream review."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "transaction_id": transaction_ids.astype(str),
            "predicted_anomaly": predicted_anomaly.astype(int),
            "anomaly_score": anomaly_scores.astype(float),
            "actual_anomaly": actual_anomaly.astype(int),
        }
    )
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)


def main() -> None:
    X, transaction_ids, y, feature_names = preprocess_data()
    model, predicted_anomaly, anomaly_scores, training_time = train_model(X)
    metrics = evaluate_model(y, predicted_anomaly)

    save_model(model, feature_names)
    save_predictions(transaction_ids, predicted_anomaly, anomaly_scores, y)

    detected_anomalies = int(predicted_anomaly.sum())
    anomaly_percentage = (detected_anomalies / len(predicted_anomaly)) * 100

    print(f"Number of rows: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"Training time (seconds): {training_time:.4f}")
    print(f"Number of detected anomalies: {detected_anomalies}")
    print(f"Anomaly percentage: {anomaly_percentage:.4f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(metrics["classification_report"])
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved feature names: {FEATURE_NAMES_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
