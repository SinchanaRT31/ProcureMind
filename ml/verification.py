"""Deterministic procurement document verification and temporal experiments."""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ml.preprocess import TRANSACTION_ID_COLUMN, clean_data


DUPLICATE_SIGNATURE = ["vendor_id", "item_category", "quantity", "unit_price", "total_amount"]


class ProcurementVerificationEngine(BaseEstimator):
    """Find duplicate candidates and deterministic document/date consistency issues.

    The three-day window is a configurable review policy for otherwise exact
    business-metadata matches; it is not learned and does not use labels.
    """

    def __init__(self, duplicate_window_days: int = 3, arithmetic_rtol: float = 1e-4):
        self.duplicate_window_days = duplicate_window_days
        self.arithmetic_rtol = arithmetic_rtol

    def fit(self, X: pd.DataFrame, y: object = None) -> "ProcurementVerificationEngine":
        df = clean_data(X)
        self.reference_ = df[[TRANSACTION_ID_COLUMN, "invoice_date", *DUPLICATE_SIGNATURE]].copy()
        return self

    def assess(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "reference_"):
            raise ValueError("ProcurementVerificationEngine must be fitted before assessment.")
        df = clean_data(X)
        duplicate_candidates = self._duplicate_candidates(df)
        consistency = self._consistency_checks(df)
        output = pd.DataFrame(index=df.index)
        output["duplicate_candidates"] = [json.dumps(duplicate_candidates.get(txn, []), sort_keys=True) for txn in df[TRANSACTION_ID_COLUMN]]
        output["consistency_checks"] = [json.dumps(consistency.get(txn, []), sort_keys=True) for txn in df[TRANSACTION_ID_COLUMN]]
        output["verification_signals"] = [json.dumps(self._signals(duplicate_candidates.get(txn, []), consistency.get(txn, [])), sort_keys=True) for txn in df[TRANSACTION_ID_COLUMN]]
        return output

    def _duplicate_candidates(self, incoming: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        reference = self.reference_.copy()
        reference["_incoming"] = False
        current = incoming[[TRANSACTION_ID_COLUMN, "invoice_date", *DUPLICATE_SIGNATURE]].copy()
        current["_incoming"] = True
        combined = pd.concat([reference, current], ignore_index=True).drop_duplicates(TRANSACTION_ID_COLUMN, keep="last")
        duplicated = combined[combined.duplicated(DUPLICATE_SIGNATURE, keep=False)].sort_values([*DUPLICATE_SIGNATURE, "invoice_date", TRANSACTION_ID_COLUMN])
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Only duplicated blocks are iterated; the full 200k set is never pairwise compared.
        for _, group in duplicated.groupby(DUPLICATE_SIGNATURE, dropna=False, sort=False):
            rows = group.reset_index(drop=True)
            for position in range(1, len(rows)):
                earlier, later = rows.iloc[position - 1], rows.iloc[position]
                if pd.isna(earlier.invoice_date) or pd.isna(later.invoice_date):
                    continue
                days = abs((later.invoice_date - earlier.invoice_date).days)
                if days > self.duplicate_window_days or not (earlier._incoming or later._incoming):
                    continue
                for source, match in ((earlier, later), (later, earlier)):
                    if not source._incoming:
                        continue
                    candidates[str(source.transaction_id)].append({
                        "candidate_type": "duplicate_invoice",
                        "matched_transaction_id": str(match.transaction_id),
                        "match_reason": "Potential duplicate: exact vendor, category, quantity, unit price, and amount; invoice dates are close.",
                        "match_strength": "exact_metadata_same_date" if days == 0 else "exact_metadata_close_date",
                        "days_between_invoices": int(days),
                        "fields_compared": DUPLICATE_SIGNATURE,
                    })
        return candidates

    def _consistency_checks(self, df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
        checks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        invoice_before_order = df["invoice_date"] < df["order_date"]
        due_before_invoice = df["due_date"] < df["invoice_date"]
        expected = df["quantity"] * df["unit_price"]
        material_amount_difference = ~np.isclose(df["total_amount"], expected, rtol=self.arithmetic_rtol, atol=0.01, equal_nan=True)
        for mask, name, explanation in [
            (invoice_before_order, "invoice_before_order", "Invoice date precedes order date."),
            (due_before_invoice, "due_before_invoice", "Due date precedes invoice date."),
            (material_amount_difference, "amount_arithmetic_mismatch", "Total amount materially differs from quantity multiplied by unit price."),
        ]:
            for transaction_id in df.loc[mask, TRANSACTION_ID_COLUMN].astype(str):
                checks[transaction_id].append({"check": name, "status": "issue", "explanation": explanation})
        return checks

    @staticmethod
    def _signals(candidates: list[dict[str, Any]], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        signals: list[dict[str, Any]] = []
        if candidates:
            strongest = "high" if any(item["days_between_invoices"] == 0 for item in candidates) else "moderate"
            signals.append({"name": "duplicate_invoice_candidate", "severity": strongest, "value": len(candidates), "baseline": None, "explanation": "Potential duplicate invoice candidate requires review against the matched transaction."})
        for check in checks:
            signals.append({"name": check["check"], "severity": "moderate", "value": None, "baseline": None, "explanation": check["explanation"]})
        return signals


def merge_evidence_signals(evidence: pd.DataFrame, verification: pd.DataFrame) -> pd.DataFrame:
    """Append Phase 4 signals without mutating the Phase 2 evidence artifact."""
    merged = evidence.copy()
    merged["signals"] = [json.dumps(json.loads(base) + json.loads(extra), sort_keys=True) for base, extra in zip(evidence["signals"], verification["verification_signals"])]
    for column in ["duplicate_candidates", "consistency_checks", "verification_signals"]:
        merged[column] = verification[column]
    return merged


def temporal_vendor_cadence_experiment(df: pd.DataFrame, cutoff: str = "2025-01-01") -> dict[str, Any]:
    """Separate chronological experiment; it does not alter random-split scoring."""
    data = clean_data(df)
    data = data.assign(month=data["order_date"].dt.to_period("M"))
    historical = data[data["order_date"] < pd.Timestamp(cutoff)]
    later = data[data["order_date"] >= pd.Timestamp(cutoff)]
    history = historical.groupby(["vendor_id", "month"]).agg(count=(TRANSACTION_ID_COLUMN, "size"), spend=("total_amount", "sum")).reset_index()
    baseline = history.groupby("vendor_id").agg(count_median=("count", "median"), spend_median=("spend", "median")).reset_index()
    later_months = later.groupby(["vendor_id", "month"]).agg(count=(TRANSACTION_ID_COLUMN, "size"), spend=("total_amount", "sum")).reset_index().merge(baseline, on="vendor_id", how="left")
    for measure in ["count", "spend"]:
        ratio = later_months[measure].div(later_months[f"{measure}_median"].replace(0, np.nan))
        later_months[f"{measure}_log_ratio"] = np.log(ratio.where(ratio > 0))
    train_ratios = history.merge(baseline, on="vendor_id", how="left")
    train_values = []
    for measure in ["count", "spend"]:
        ratio = train_ratios[measure].div(train_ratios[f"{measure}_median"].replace(0, np.nan))
        train_values.append(np.log(ratio.where(ratio > 0)).dropna())
    threshold = float(pd.concat(train_values).quantile(0.99))
    spikes = (later_months["count_log_ratio"] > threshold) | (later_months["spend_log_ratio"] > threshold)
    flagged = later_months.loc[spikes, ["vendor_id", "month"]]
    flagged_keys = set(zip(flagged.vendor_id, flagged.month))
    transaction_count = int(sum((vendor, month) in flagged_keys for vendor, month in zip(later.vendor_id, later.month)))
    return {"experiment": "chronological_vendor_month_cadence", "historical_end_exclusive": cutoff, "historical_rows": len(historical), "later_rows": len(later), "training_log_ratio_99th_percentile": threshold, "flagged_vendor_months": int(spikes.sum()), "flagged_vendors": int(flagged.vendor_id.nunique()), "transactions_in_flagged_vendor_months": transaction_count, "limitation": "This separate chronological experiment is not the Phase 1 stratified benchmark and is not added to random-split priorities."}


def verification_evaluation(verification: pd.DataFrame, risk: pd.DataFrame, ml_predictions: pd.Series, evidence: pd.DataFrame) -> dict[str, Any]:
    """Describe deterministic verification evidence on a held-out set."""
    duplicate = verification["duplicate_candidates"].map(lambda value: bool(json.loads(value)))
    checks = verification["consistency_checks"].map(json.loads)
    issue_types: dict[str, int] = defaultdict(int)
    for items in checks:
        for item in items:
            issue_types[item["check"]] += 1
    consistency = checks.map(bool)
    ml = pd.Series(ml_predictions, index=verification.index).astype(bool)
    high_priority = risk["investigation_priority"].isin(["HIGH", "CRITICAL"])
    phase2_timing = evidence["invoice_delay_deviation_severity"].isin(["moderate", "high"]) | evidence["payment_timing_deviation_severity"].isin(["moderate", "high"])
    return {
        "duplicate": {"candidate_transactions": int(duplicate.sum()), "percentage": float(duplicate.mean() * 100), "overlap_ml_anomalies": int((duplicate & ml).sum()), "overlap_high_or_critical": int((duplicate & high_priority).sum())},
        "consistency": {"issue_transactions": int(consistency.sum()), "issue_types": dict(issue_types), "overlap_ml_anomalies": int((consistency & ml).sum())},
        "temporal_consistency": {"issue_transactions": int(consistency.sum()), "overlap_phase2_timing_evidence": int((consistency & phase2_timing).sum())},
    }
