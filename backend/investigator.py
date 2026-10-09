"""Trusted transaction lookup and deterministic investigation reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
TRANSACTION_DATASET = ROOT_DIR / "dataset" / "ProcureMind" / "procuremind_procurement_dataset.csv"
RAW_TRANSACTION_COLUMNS = (
    "transaction_id",
    "vendor_id",
    "vendor_rating",
    "department_id",
    "item_category",
    "quantity",
    "unit_price",
    "total_amount",
    "order_date",
    "invoice_date",
    "due_date",
    "payment_status",
    "purchase_type",
    "vendor_location",
)
TRANSACTION_ID_PATTERN = re.compile(r"TXN-[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
REPORT_SCHEMA_VERSION = "1.0"
CSV_CHUNK_SIZE = 25_000


class InvalidTransactionIdError(ValueError):
    pass


class TransactionNotFoundError(LookupError):
    pass


class AmbiguousTransactionError(LookupError):
    pass


class TransactionSourceUnavailableError(RuntimeError):
    pass


class InferenceUnavailableError(RuntimeError):
    pass


class PredictionMismatchError(RuntimeError):
    pass


def _decode_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


class InvestigatorReportService:
    """Generate reports from exact-ID rows in the fixed trusted dataset."""

    def __init__(self, dataset_path: str | Path = TRANSACTION_DATASET):
        # dataset_path is an internal constructor seam for tests, never request input.
        self._dataset_path = Path(dataset_path)

    @staticmethod
    def validate_transaction_id(transaction_id: Any) -> str:
        if not isinstance(transaction_id, str) or not TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
            raise InvalidTransactionIdError("transaction_id must be a valid TXN identifier.")
        return transaction_id

    def lookup_transaction(self, transaction_id: str) -> pd.DataFrame:
        transaction_id = self.validate_transaction_id(transaction_id)
        matches: list[pd.DataFrame] = []
        try:
            for chunk in pd.read_csv(
                self._dataset_path,
                usecols=RAW_TRANSACTION_COLUMNS,
                dtype={"transaction_id": "string"},
                chunksize=CSV_CHUNK_SIZE,
            ):
                exact_rows = chunk.loc[chunk["transaction_id"] == transaction_id]
                if not exact_rows.empty:
                    matches.append(exact_rows)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise TransactionSourceUnavailableError("Trusted transaction data is unavailable.") from exc

        match_count = sum(len(rows) for rows in matches)
        if match_count == 0:
            raise TransactionNotFoundError("No exact transaction ID match was found.")
        if match_count > 1:
            raise AmbiguousTransactionError("The transaction ID has multiple exact matches.")
        return matches[0].reset_index(drop=True).loc[:, RAW_TRANSACTION_COLUMNS]

    def generate_report(self, transaction_id: str) -> dict[str, Any]:
        transaction_id = self.validate_transaction_id(transaction_id)
        raw_transaction = self.lookup_transaction(transaction_id)
        try:
            # Import only for a report request; preserve the existing public predictor.
            from ml.predict import predict_procurement_data

            prediction = predict_procurement_data(
                raw_transaction,
                include_risk=True,
                include_explanations=True,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
            raise InferenceUnavailableError("Required prediction or explanation artifacts are unavailable.") from exc

        if len(prediction) != 1 or "transaction_id" not in prediction.columns:
            raise PredictionMismatchError("Prediction output did not identify exactly one transaction.")
        prediction_id = prediction.iloc[0]["transaction_id"]
        if not isinstance(prediction_id, str) or prediction_id != transaction_id:
            raise PredictionMismatchError("Prediction transaction ID does not match the requested ID.")

        row = prediction.iloc[0]
        predicted_anomaly = self._integer(row.get("predicted_anomaly"))
        score = self._number(row.get("ml_anomaly_score"))
        if predicted_anomaly is None or score is None:
            model_finding: dict[str, Any] = {"status": "unavailable", "source": "Isolation Forest"}
        else:
            model_finding = {
                "status": "flagged" if predicted_anomaly == 1 else "not_flagged",
                "source": "Isolation Forest",
                "predicted_anomaly": bool(predicted_anomaly),
                "anomaly_score": score,
                "score_interpretation": "Higher values indicate a stronger model anomaly signal; this is not a probability.",
            }

        shap_contributions = self._shap_contributions(row)
        if shap_contributions is None:
            shap_result: dict[str, Any] = {"status": "unavailable", "source": "SHAP"}
        else:
            shap_result = {
                "status": "available",
                "source": "SHAP",
                "reference": "Saved training-only background",
                "contributions": shap_contributions,
                "interpretation": "Positive contributions raise the model anomaly score relative to the background; negative contributions lower it.",
            }

        risk_evidence = _decode_json(row.get("evidence"))
        recommendations = _decode_json(row.get("recommended_actions"))
        verification = {
            name: _decode_json(row.get(name))
            for name in ("duplicate_candidates", "consistency_checks", "verification_signals")
        }
        verification_available = all(value is not None for value in verification.values())
        selected_evidence = risk_evidence if isinstance(risk_evidence, list) else None
        evidence_summary = row.get("evidence_summary")
        evidence_summary = evidence_summary if isinstance(evidence_summary, str) else None
        priority = row.get("investigation_priority")
        priority = priority if isinstance(priority, str) else None

        why_flagged = {
            "model_prediction": model_finding,
            "model_explanation": shap_result,
        }
        supporting_findings = [{
            "source": "risk_engine.selected_evidence",
            "status": "available" if selected_evidence is not None else "unavailable",
            "findings": selected_evidence or [],
            "evidence_summary": evidence_summary,
            "investigation_priority": priority,
        }]
        if not verification_available:
            inconclusive: list[dict[str, Any]] = [{
                "source": "verification_engine",
                "status": "unavailable",
            }]
        else:
            inconclusive = [{
                "source": "verification_engine",
                "status": "findings_require_human_verification" if any(verification.values()) else "no_findings_returned",
                **verification,
            }]
        if recommendations is None or not isinstance(recommendations, list):
            follow_up_checks: dict[str, Any] = {"status": "unavailable", "source": "risk_engine.recommended_actions"}
        else:
            follow_up_checks = {
                "status": "available",
                "source": "risk_engine.recommended_actions",
                "checks": recommendations,
            }

        return {
            "transaction_id": transaction_id,
            "schema_version": REPORT_SCHEMA_VERSION,
            "why_flagged": why_flagged,
            "supporting_findings": supporting_findings,
            "inconclusive_or_conflicting": inconclusive,
            "missing_information": [
                "Supporting purchase-order and invoice documents are not supplied to this report service.",
                "Approval records and business justification are not supplied to this report service.",
            ],
            "follow_up_checks": follow_up_checks,
            "limitations": [
                "This report summarizes structured transaction data and existing model outputs; findings require human verification.",
                "Anomaly scores are not probabilities, and an anomaly signal does not establish fraud, intent, or wrongdoing.",
                "Report generation does not create or alter a human investigation decision or review history.",
            ],
        }

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number in (0, 1) else None

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if pd.notna(number) else None

    @classmethod
    def _shap_contributions(cls, row: pd.Series) -> list[dict[str, Any]] | None:
        contributions = []
        for rank in range(1, 4):
            feature = row.get(f"top_feature_{rank}")
            value = cls._number(row.get(f"top_contribution_{rank}"))
            if not isinstance(feature, str) or not feature or value is None:
                continue
            contributions.append({
                "feature": feature,
                "contribution": value,
                "direction": "raises_anomaly_score" if value > 0 else "lowers_anomaly_score" if value < 0 else "no_change",
            })
        return contributions or None
