"""Deterministic investigation-priority assessment for procurement evidence."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


ML_SCORE_QUANTILES = {"elevated": 0.95, "high": 0.99}
FAMILY_BY_SIGNAL = {
    "vendor_price_deviation": "PRICE",
    "vendor_category_price_deviation": "PRICE",
    "department_category_price_deviation": "PRICE",
    "vendor_spend_deviation": "SPEND",
    "department_spend_deviation": "SPEND",
    "large_transaction": "SPEND",
    "quantity_deviation": "QUANTITY",
    "invoice_delay_deviation": "TIMING",
    "payment_timing_deviation": "TIMING",
    "duplicate_invoice_candidate": "DUPLICATE",
    "invoice_before_order": "CONSISTENCY",
    "due_before_invoice": "CONSISTENCY",
    "amount_arithmetic_mismatch": "CONSISTENCY",
}
FAMILY_ORDER = ["PRICE", "SPEND", "QUANTITY", "TIMING", "DUPLICATE", "CONSISTENCY"]
SEVERITY_RANK = {"moderate": 1, "high": 2}
ACTION_BY_FAMILY = {
    "PRICE": "Compare unit price with recent vendor and category purchases.",
    "SPEND": "Review the transaction against vendor and department spending baselines and supporting approvals.",
    "QUANTITY": "Confirm the requested quantity with the requesting department.",
    "TIMING": "Verify the order, invoice, and payment due dates with supporting records.",
    "DUPLICATE": "Compare the candidate transaction and verify whether both invoices represent separate purchases.",
    "CONSISTENCY": "Verify the PO, invoice, transaction dates, and amount fields against supporting records.",
}


class ProcurementRiskEngine(BaseEstimator):
    """Convert ML indicators and de-correlated evidence into review priorities.

    This is a transparent triage heuristic, not a fraud model or probability.
    """

    def __init__(self, ml_score_quantiles: dict[str, float] | None = None):
        self.ml_score_quantiles = ml_score_quantiles or ML_SCORE_QUANTILES

    def fit(self, ml_anomaly_scores: pd.Series) -> "ProcurementRiskEngine":
        scores = pd.Series(ml_anomaly_scores).dropna()
        if scores.empty:
            raise ValueError("At least one ML anomaly score is required to fit the risk engine.")
        self.ml_score_thresholds_ = {
            level: float(scores.quantile(quantile))
            for level, quantile in self.ml_score_quantiles.items()
        }
        self.priority_policy_ = {
            "low": "No evidence family and no ML indication.",
            "medium": "One moderate evidence family or an ML-only indication.",
            "high": "At least one high-severity family or at least two independent evidence families.",
            "critical": "At least two independent high-severity evidence families.",
            "double_counting": "Only the strongest signal in each PRICE, SPEND, QUANTITY, or TIMING family contributes to priority.",
        }
        return self

    def assess(self, evidence: pd.DataFrame, ml_predictions: pd.Series, ml_scores: pd.Series) -> pd.DataFrame:
        if not hasattr(self, "ml_score_thresholds_"):
            raise ValueError("ProcurementRiskEngine must be fitted before assessment.")
        predictions = pd.Series(ml_predictions, index=evidence.index).astype(bool)
        scores = pd.Series(ml_scores, index=evidence.index).astype(float)
        rows = [self._assess_row(evidence.loc[index, "signals"], bool(predictions[index]), float(scores[index])) for index in evidence.index]
        result = pd.DataFrame(rows, index=evidence.index)
        result["ml_anomaly"] = predictions.astype(int)
        result["ml_anomaly_score"] = scores
        return result[["ml_anomaly", "ml_anomaly_score", "ml_signal_level", "evidence", "evidence_summary", "evidence_family_count", "high_severity_signal_count", "investigation_priority", "recommended_actions"]]

    def _assess_row(self, serialized_signals: str, ml_flagged: bool, ml_score: float) -> dict[str, Any]:
        signals = json.loads(serialized_signals)
        families: dict[str, dict[str, Any]] = {}
        high_signal_count = 0
        for signal in signals:
            family = FAMILY_BY_SIGNAL[signal["name"]]
            signal["family"] = family
            if signal["severity"] == "high":
                high_signal_count += 1
            current = families.get(family)
            if current is None or SEVERITY_RANK[signal["severity"]] > SEVERITY_RANK[current["severity"]]:
                families[family] = signal
        selected = [families[family] for family in FAMILY_ORDER if family in families]
        ml_level = self._ml_level(ml_score)
        priority = self._priority(len(selected), sum(item["severity"] == "high" for item in selected), ml_flagged, ml_level)
        return {
            "ml_signal_level": ml_level,
            "evidence": json.dumps(selected, sort_keys=True),
            "evidence_summary": self._summary(selected, ml_flagged),
            "evidence_family_count": len(selected),
            "high_severity_signal_count": high_signal_count,
            "investigation_priority": priority,
            "recommended_actions": json.dumps(self._actions(selected, ml_flagged), sort_keys=True),
        }

    def _ml_level(self, score: float) -> str:
        if score >= self.ml_score_thresholds_["high"]:
            return "high"
        if score >= self.ml_score_thresholds_["elevated"]:
            return "elevated"
        return "normal"

    @staticmethod
    def _priority(family_count: int, high_family_count: int, ml_flagged: bool, ml_level: str) -> str:
        if high_family_count >= 2:
            return "CRITICAL"
        if high_family_count >= 1 or family_count >= 2:
            return "HIGH"
        if family_count == 1 or ml_flagged or ml_level != "normal":
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _summary(selected: list[dict[str, Any]], ml_flagged: bool) -> str:
        if selected:
            text = " ".join(item["explanation"] for item in selected[:3])
            return f"{text} Requires verification."
        if ml_flagged:
            return "Isolation Forest marked this transaction as anomalous; review supporting records for verification."
        return "No material model or evidence signal requires additional investigation."

    @staticmethod
    def _actions(selected: list[dict[str, Any]], ml_flagged: bool) -> list[str]:
        actions = [ACTION_BY_FAMILY[item["family"]] for item in selected]
        if len(selected) >= 2:
            actions.append("Escalate for procurement review because multiple independent evidence families require verification.")
        elif ml_flagged and not selected:
            actions.append("Review the anomalous transaction and supporting documentation before deciding on next steps.")
        return actions or ["No additional investigation action is recommended; retain normal review controls."]


def risk_evaluation(risk: pd.DataFrame, actual_anomaly: pd.Series) -> dict[str, Any]:
    """Describe priority distribution and labelled rates without tuning on labels."""
    priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    result: dict[str, Any] = {"priority_distribution": {}, "evidence_family_distribution": {}, "ml_overlap": {}}
    for priority in priorities:
        rows = risk["investigation_priority"] == priority
        result["priority_distribution"][priority] = {
            "count": int(rows.sum()), "percentage": float(rows.mean() * 100),
            "ml_anomalies": int(risk.loc[rows, "ml_anomaly"].sum()),
            "labelled_anomaly_rate": float(pd.Series(actual_anomaly, index=risk.index)[rows].mean()) if rows.any() else 0.0,
        }
    families = risk["evidence_family_count"].value_counts().sort_index()
    result["evidence_family_distribution"] = {str(int(key)): int(value) for key, value in families.items()}
    result["ml_overlap"] = {"ml_anomalies": int(risk["ml_anomaly"].sum()), "high_or_critical": int(risk["investigation_priority"].isin(["HIGH", "CRITICAL"]).sum()), "both": int(((risk["ml_anomaly"] == 1) & risk["investigation_priority"].isin(["HIGH", "CRITICAL"])).sum())}
    return result
