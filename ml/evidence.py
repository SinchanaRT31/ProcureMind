"""Transparent, training-data-fitted procurement evidence signals.

This module intentionally does not train a fraud classifier or combine signals into
a risk score. It supplies reviewable evidence alongside the Phase 1 ML score.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ml.preprocess import clean_data


ALERT_QUANTILES = {"moderate": 0.95, "high": 0.99}
MIN_GROUP_SIZE = 5
FORBIDDEN_OUTCOME_COLUMNS = {
    "is_anomaly", "anomaly_type", "risk_score", "risk_level",
    "investigation_status", "potential_savings",
}


def _absolute_log_ratio(value: pd.Series, baseline: pd.Series) -> pd.Series:
    ratio = value.div(baseline.replace(0, np.nan))
    return np.log(ratio.where(ratio > 0)).abs().replace([np.inf, -np.inf], np.nan)


def _thresholds(values: pd.Series, quantiles: dict[str, float]) -> dict[str, float]:
    valid = values.dropna()
    if valid.empty:
        return {level: float("inf") for level in quantiles}
    return {level: float(valid.quantile(quantile)) for level, quantile in quantiles.items()}


class ProcurementEvidenceEngine(BaseEstimator):
    """Fit behavioral baselines on training data and return structured evidence."""

    def __init__(self, alert_quantiles: dict[str, float] | None = None, min_group_size: int = MIN_GROUP_SIZE):
        self.alert_quantiles = alert_quantiles or ALERT_QUANTILES
        self.min_group_size = min_group_size

    def fit(self, X: pd.DataFrame, y: object = None) -> "ProcurementEvidenceEngine":
        df = clean_data(X)
        self.global_amount_median_ = float(df["total_amount"].median())
        self.global_price_median_ = float(df["unit_price"].median())
        self.global_quantity_median_ = float(df["quantity"].median())
        self.invoice_delay_median_ = float((df["invoice_date"] - df["order_date"]).dt.days.median())
        self.payment_due_median_ = float((df["due_date"] - df["invoice_date"]).dt.days.median())

        self.vendor_price_median_ = self._group_median(df, ["vendor_id"], "unit_price")
        self.vendor_category_price_median_ = self._group_median(df, ["vendor_id", "item_category"], "unit_price")
        self.department_category_price_median_ = self._group_median(df, ["department_id", "item_category"], "unit_price")
        self.vendor_amount_median_ = self._group_median(df, ["vendor_id"], "total_amount")
        self.department_amount_median_ = self._group_median(df, ["department_id"], "total_amount")
        self.vendor_category_quantity_median_ = self._group_median(df, ["vendor_id", "item_category"], "quantity")
        self.department_category_amount_values_ = self._group_values(df, ["department_id", "item_category"], "total_amount")
        self.global_amount_values_ = np.sort(df["total_amount"].dropna().to_numpy())
        self.vendor_counts_ = df["vendor_id"].value_counts()
        self.vendor_count_percentiles_ = self.vendor_counts_.rank(pct=True)

        training_evidence = self._measure(df)
        self.thresholds_ = {
            signal: _thresholds(training_evidence[signal], self.alert_quantiles)
            for signal in [
                "vendor_price_deviation", "vendor_category_price_deviation",
                "department_category_price_deviation", "vendor_spend_deviation",
                "department_spend_deviation", "quantity_deviation",
                "invoice_delay_deviation", "payment_timing_deviation",
            ]
        }
        # Amount percentiles are already empirical percentiles from the train data.
        self.amount_percentile_thresholds_ = dict(self.alert_quantiles)
        self.threshold_method_ = {
            "deviation": "absolute log-ratio (or absolute day difference) at training-data 95th/99th percentiles",
            "amount_percentile": "empirical percentile of a department/category amount distribution fitted on training data, with global fallback",
            "vendor_activity": "training vendor transaction count and its percentile; informational only",
        }
        return self

    def assess(self, X: pd.DataFrame, ml_anomaly_scores: pd.Series | None = None) -> pd.DataFrame:
        """Return one structured evidence record per raw transaction."""
        if not hasattr(self, "thresholds_"):
            raise ValueError("ProcurementEvidenceEngine must be fitted before assessment.")
        df = clean_data(X)
        measures = self._measure(df)
        output = pd.DataFrame(index=df.index)
        output["signals"] = [json.dumps(self._signals_for_row(measures.loc[idx]), sort_keys=True) for idx in df.index]
        output["context"] = [json.dumps(self._context_for_row(df.loc[idx]), sort_keys=True) for idx in df.index]
        for name in self.thresholds_:
            output[f"{name}_severity"] = [self._severity(name, value) for value in measures[name]]
        output["large_transaction_severity"] = [self._percentile_severity(value) for value in measures["contextual_amount_percentile"]]
        output["flagged_signal_count"] = (output.filter(like="_severity") != "none").sum(axis=1)
        if ml_anomaly_scores is not None:
            output["ml_anomaly_score"] = pd.Series(ml_anomaly_scores, index=df.index)
        return output

    def _group_median(self, df: pd.DataFrame, group: list[str], value: str) -> pd.Series:
        grouped = df.groupby(group)[value].agg(["median", "count"])
        return grouped.loc[grouped["count"] >= self.min_group_size, "median"]

    @staticmethod
    def _group_values(df: pd.DataFrame, group: list[str], value: str) -> dict[tuple[str, ...], np.ndarray]:
        result: dict[tuple[str, ...], np.ndarray] = {}
        for key, values in df.groupby(group)[value]:
            result[key if isinstance(key, tuple) else (key,)] = np.sort(values.dropna().to_numpy())
        return result

    @staticmethod
    def _map_group(df: pd.DataFrame, mapping: pd.Series, columns: list[str], fallback: float) -> pd.Series:
        if len(columns) == 1:
            return df[columns[0]].map(mapping).fillna(fallback)
        keys = pd.MultiIndex.from_frame(df[columns])
        return pd.Series(mapping.reindex(keys).to_numpy(), index=df.index).fillna(fallback)

    def _measure(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        vendor_price = self._map_group(df, self.vendor_price_median_, ["vendor_id"], self.global_price_median_)
        vendor_category_price = self._map_group(df, self.vendor_category_price_median_, ["vendor_id", "item_category"], self.global_price_median_)
        department_category_price = self._map_group(df, self.department_category_price_median_, ["department_id", "item_category"], self.global_price_median_)
        vendor_amount = self._map_group(df, self.vendor_amount_median_, ["vendor_id"], self.global_amount_median_)
        department_amount = self._map_group(df, self.department_amount_median_, ["department_id"], self.global_amount_median_)
        vendor_category_quantity = self._map_group(df, self.vendor_category_quantity_median_, ["vendor_id", "item_category"], self.global_quantity_median_)
        result["vendor_price_baseline"] = vendor_price
        result["vendor_category_price_baseline"] = vendor_category_price
        result["department_category_price_baseline"] = department_category_price
        result["vendor_spend_baseline"] = vendor_amount
        result["department_spend_baseline"] = department_amount
        result["quantity_baseline"] = vendor_category_quantity
        result["vendor_price_deviation"] = _absolute_log_ratio(df["unit_price"], vendor_price)
        result["vendor_category_price_deviation"] = _absolute_log_ratio(df["unit_price"], vendor_category_price)
        result["department_category_price_deviation"] = _absolute_log_ratio(df["unit_price"], department_category_price)
        result["vendor_spend_deviation"] = _absolute_log_ratio(df["total_amount"], vendor_amount)
        result["department_spend_deviation"] = _absolute_log_ratio(df["total_amount"], department_amount)
        result["quantity_deviation"] = _absolute_log_ratio(df["quantity"], vendor_category_quantity)
        result["invoice_delay_deviation"] = ((df["invoice_date"] - df["order_date"]).dt.days - self.invoice_delay_median_).abs()
        result["payment_timing_deviation"] = ((df["due_date"] - df["invoice_date"]).dt.days - self.payment_due_median_).abs()
        result["contextual_amount_percentile"] = [self._amount_percentile(row) for _, row in df.iterrows()]
        result["vendor_transaction_count"] = df["vendor_id"].map(self.vendor_counts_).fillna(0.0)
        result["vendor_frequency_percentile"] = df["vendor_id"].map(self.vendor_count_percentiles_).fillna(0.0)
        return result

    def _amount_percentile(self, row: pd.Series) -> float:
        key = (row["department_id"], row["item_category"])
        values = self.department_category_amount_values_.get(key, self.global_amount_values_)
        if len(values) == 0 or pd.isna(row["total_amount"]):
            return np.nan
        return float(np.searchsorted(values, row["total_amount"], side="right") / len(values))

    def _severity(self, name: str, value: float) -> str:
        if pd.isna(value):
            return "none"
        threshold = self.thresholds_[name]
        if value > 0 and value > threshold["high"]:
            return "high"
        if value > 0 and value > threshold["moderate"]:
            return "moderate"
        return "none"

    def _percentile_severity(self, value: float) -> str:
        if pd.isna(value):
            return "none"
        if value > self.amount_percentile_thresholds_["high"]:
            return "high"
        if value > self.amount_percentile_thresholds_["moderate"]:
            return "moderate"
        return "none"

    def _signals_for_row(self, row: pd.Series) -> list[dict[str, Any]]:
        descriptions = {
            "vendor_price_deviation": "Unit price differs substantially from the learned vendor price baseline.",
            "vendor_category_price_deviation": "Unit price differs substantially from the learned vendor/category price baseline.",
            "department_category_price_deviation": "Unit price differs substantially from the learned department/category price baseline.",
            "vendor_spend_deviation": "Transaction amount differs substantially from the learned vendor spend baseline.",
            "department_spend_deviation": "Transaction amount differs substantially from the learned department spend baseline.",
            "quantity_deviation": "Quantity differs substantially from the learned vendor/category quantity baseline.",
            "invoice_delay_deviation": "Invoice delay differs substantially from the training-data timing pattern.",
            "payment_timing_deviation": "Payment due timing differs substantially from the training-data timing pattern.",
        }
        baseline_columns = {
            "vendor_price_deviation": "vendor_price_baseline",
            "vendor_category_price_deviation": "vendor_category_price_baseline",
            "department_category_price_deviation": "department_category_price_baseline",
            "vendor_spend_deviation": "vendor_spend_baseline",
            "department_spend_deviation": "department_spend_baseline",
            "quantity_deviation": "quantity_baseline",
            "invoice_delay_deviation": None,
            "payment_timing_deviation": None,
        }
        signals = []
        for name, explanation in descriptions.items():
            severity = self._severity(name, row[name])
            if severity != "none":
                baseline_column = baseline_columns[name]
                baseline = self.invoice_delay_median_ if name == "invoice_delay_deviation" else self.payment_due_median_ if name == "payment_timing_deviation" else row[baseline_column]
                signals.append({"name": name, "severity": severity, "value": round(float(row[name]), 6), "baseline": round(float(baseline), 6), "explanation": explanation})
        severity = self._percentile_severity(row["contextual_amount_percentile"])
        if severity != "none":
            signals.append({"name": "large_transaction", "severity": severity, "value": round(float(row["contextual_amount_percentile"]), 6), "baseline": None, "explanation": "Transaction amount is in the upper tail of its learned department/category context."})
        return signals

    def _context_for_row(self, row: pd.Series) -> dict[str, Any]:
        count = float(self.vendor_counts_.get(row["vendor_id"], 0))
        frequency_percentile = float(self.vendor_count_percentiles_.get(row["vendor_id"], 0))
        return {
            "department_id": str(row["department_id"]), "item_category": str(row["item_category"]),
            "purchase_type": str(row["purchase_type"]), "payment_status": str(row["payment_status"]),
            "vendor_location": str(row["vendor_location"]), "vendor_rating": None if pd.isna(row["vendor_rating"]) else float(row["vendor_rating"]),
            "vendor_transaction_count": count, "vendor_frequency_percentile": frequency_percentile,
            "vendor_activity_note": "Context only; vendor frequency and rating are not anomaly proof.",
        }


def evidence_evaluation(evidence: pd.DataFrame, ml_predictions: pd.Series) -> dict[str, Any]:
    """Summarize held-out evidence without tuning it against held-out labels."""
    severity_columns = [column for column in evidence if column.endswith("_severity")]
    result: dict[str, Any] = {"signal_distribution": {}, "ml_signal_overlap": {}}
    ml = pd.Series(ml_predictions, index=evidence.index).astype(bool)
    for column in severity_columns:
        counts = Counter(evidence[column])
        flagged = evidence[column].isin(["moderate", "high"])
        name = column.removesuffix("_severity")
        result["signal_distribution"][name] = {"count": int(flagged.sum()), "percentage": float(flagged.mean() * 100), "severity_counts": dict(counts)}
        result["ml_signal_overlap"][name] = {"signal_flagged": int(flagged.sum()), "ml_anomalies": int(ml.sum()), "both": int((flagged & ml).sum())}
    return result
