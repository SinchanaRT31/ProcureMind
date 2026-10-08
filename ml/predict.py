from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

if __package__ in {None, ""}:  # pragma: no cover - CLI compatibility
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocess import TRANSACTION_ID_COLUMN, clean_data
from ml.train import EVIDENCE_ENGINE_PATH, MODEL_PATH, RISK_ENGINE_PATH, TRANSFORMER_PATH, VERIFICATION_ENGINE_PATH


def load_artifacts(model_path: str | Path = MODEL_PATH, transformer_path: str | Path = TRANSFORMER_PATH):
    """Load a model and the exact train-fitted transformer saved with it."""
    if not Path(model_path).exists() or not Path(transformer_path).exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python ml/train.py` first.")
    return joblib.load(model_path), joblib.load(transformer_path)


def predict_procurement_data(raw_data: pd.DataFrame, model_path: str | Path = MODEL_PATH, transformer_path: str | Path = TRANSFORMER_PATH, *, include_risk: bool = False) -> pd.DataFrame:
    """Score raw transactions; labels and downstream outcome fields are unnecessary."""
    cleaned = clean_data(raw_data)
    model, transformer = load_artifacts(model_path, transformer_path)
    features = transformer.transform(cleaned)
    predictions = pd.DataFrame({
        "transaction_id": cleaned[TRANSACTION_ID_COLUMN].astype(str),
        "predicted_anomaly": (model.predict(features) == -1).astype(int),
        "ml_anomaly_score": -model.score_samples(features),
    }, index=cleaned.index)
    if not include_risk:
        return predictions
    from ml.evidence import ProcurementEvidenceEngine
    from ml.risk import ProcurementRiskEngine
    from ml.verification import ProcurementVerificationEngine, merge_evidence_signals
    if not Path(EVIDENCE_ENGINE_PATH).exists() or not Path(RISK_ENGINE_PATH).exists() or not Path(VERIFICATION_ENGINE_PATH).exists():
        raise FileNotFoundError("Evidence/risk artifacts are missing. Run `python ml/train.py` first.")
    evidence_engine: ProcurementEvidenceEngine = joblib.load(EVIDENCE_ENGINE_PATH)
    risk_engine: ProcurementRiskEngine = joblib.load(RISK_ENGINE_PATH)
    verification_engine: ProcurementVerificationEngine = joblib.load(VERIFICATION_ENGINE_PATH)
    evidence = evidence_engine.assess(cleaned, ml_anomaly_scores=predictions["ml_anomaly_score"])
    verification = verification_engine.assess(cleaned)
    evidence = merge_evidence_signals(evidence, verification)
    risk = risk_engine.assess(evidence, predictions["predicted_anomaly"], predictions["ml_anomaly_score"])
    return predictions.join(risk.drop(columns=["ml_anomaly_score"])).join(verification)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw ProcureMind procurement data.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ml/model/predictions.csv"))
    parser.add_argument("--include-risk", action="store_true", help="Append Phase 2 evidence and Phase 3 review priority fields.")
    args = parser.parse_args()
    result = predict_procurement_data(pd.read_csv(args.input_csv), include_risk=args.include_risk)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
