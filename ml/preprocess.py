from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "dataset" / "ProcureMind" / "procuremind_procurement_dataset.csv"
DATE_COLUMNS = ["order_date", "invoice_date", "due_date"]
TARGET_COLUMN = "is_anomaly"
TRANSACTION_ID_COLUMN = "transaction_id"
REQUIRED_RAW_COLUMNS = [
    "transaction_id", "vendor_id", "vendor_rating", "department_id", "item_category",
    "quantity", "unit_price", "total_amount", "order_date", "invoice_date", "due_date",
    "payment_status", "purchase_type", "vendor_location",
]
NUMERIC_COLUMNS = ["vendor_rating", "quantity", "unit_price", "total_amount"]
POSITIVE_NUMERIC_COLUMNS = ["quantity", "unit_price", "total_amount"]
FREQUENCY_COLUMNS = ["vendor_id", "department_id", "item_category", "purchase_type", "payment_status", "vendor_location"]


def load_data(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path))


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def clean_data(df: pd.DataFrame, *, require_target: bool = False) -> pd.DataFrame:
    """Normalize raw values without calculating any learned statistics."""
    _validate_columns(df, REQUIRED_RAW_COLUMNS + ([TARGET_COLUMN] if require_target else []))
    cleaned = df.copy()
    for column in DATE_COLUMNS:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["vendor_rating"] = cleaned["vendor_rating"].where(cleaned["vendor_rating"].between(1, 5), np.nan)
    for column in POSITIVE_NUMERIC_COLUMNS:
        cleaned[column] = cleaned[column].where(cleaned[column] > 0, np.nan)
    if TARGET_COLUMN in cleaned:
        cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    return cleaned


class ProcurementFeatureTransformer(BaseEstimator, TransformerMixin):
    """Feature builder whose baselines, frequencies, and imputations are train-fitted."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "ProcurementFeatureTransformer":
        df = clean_data(X)
        self.global_medians_ = {
            column: float(df[column].median()) if pd.notna(df[column].median()) else 1.0
            for column in POSITIVE_NUMERIC_COLUMNS
        }
        self.vendor_amount_median_ = df.groupby("vendor_id")["total_amount"].median()
        self.department_amount_median_ = df.groupby("department_id")["total_amount"].median()
        self.category_quantity_median_ = df.groupby("item_category")["quantity"].median()
        self.category_price_median_ = df.groupby("item_category")["unit_price"].median()
        self.vendor_category_quantity_median_ = df.groupby(["vendor_id", "item_category"])["quantity"].median()
        self.vendor_category_price_median_ = df.groupby(["vendor_id", "item_category"])["unit_price"].median()
        self.frequency_maps_ = {
            column: df[column].fillna("__MISSING__").astype(str).value_counts().div(len(df))
            for column in FREQUENCY_COLUMNS
        }
        self.feature_names_ = [
            "vendor_rating", "quantity", "unit_price", "total_amount", "invoice_delay_days",
            "payment_due_days", "order_month", "order_day_of_week", "is_weekend",
            "amount_vs_vendor_baseline", "amount_vs_department_baseline",
            "quantity_vs_vendor_category_baseline", "unit_price_vs_vendor_category_baseline",
            *[f"{column}_frequency" for column in FREQUENCY_COLUMNS],
        ]
        provisional = self._build_features(df)
        self.imputation_values_ = {
            column: float(provisional[column].median()) if pd.notna(provisional[column].median()) else 0.0
            for column in self.feature_names_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "imputation_values_"):
            raise ValueError("ProcurementFeatureTransformer must be fitted before transform.")
        return self._build_features(clean_data(X)).loc[:, self.feature_names_].fillna(self.imputation_values_).astype(float)

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        for column in ["vendor_rating", "quantity", "unit_price", "total_amount"]:
            features[column] = df[column]
        features["invoice_delay_days"] = (df["invoice_date"] - df["order_date"]).dt.days
        features["payment_due_days"] = (df["due_date"] - df["invoice_date"]).dt.days
        features["order_month"] = df["order_date"].dt.month
        features["order_day_of_week"] = df["order_date"].dt.dayofweek
        features["is_weekend"] = (df["order_date"].dt.dayofweek >= 5).astype(float)
        vendor_amount = df["vendor_id"].map(self.vendor_amount_median_).fillna(self.global_medians_["total_amount"])
        department_amount = df["department_id"].map(self.department_amount_median_).fillna(self.global_medians_["total_amount"])
        keys = pd.MultiIndex.from_frame(df[["vendor_id", "item_category"]])
        quantity_baseline = pd.Series(self.vendor_category_quantity_median_.reindex(keys).to_numpy(), index=df.index)
        price_baseline = pd.Series(self.vendor_category_price_median_.reindex(keys).to_numpy(), index=df.index)
        quantity_baseline = quantity_baseline.fillna(df["item_category"].map(self.category_quantity_median_)).fillna(self.global_medians_["quantity"])
        price_baseline = price_baseline.fillna(df["item_category"].map(self.category_price_median_)).fillna(self.global_medians_["unit_price"])
        features["amount_vs_vendor_baseline"] = _safe_ratio(df["total_amount"], vendor_amount)
        features["amount_vs_department_baseline"] = _safe_ratio(df["total_amount"], department_amount)
        features["quantity_vs_vendor_category_baseline"] = _safe_ratio(df["quantity"], quantity_baseline)
        features["unit_price_vs_vendor_category_baseline"] = _safe_ratio(df["unit_price"], price_baseline)
        for column in FREQUENCY_COLUMNS:
            features[f"{column}_frequency"] = df[column].fillna("__MISSING__").astype(str).map(self.frequency_maps_[column]).fillna(0.0)
        return features.replace([np.inf, -np.inf], np.nan)


def preprocess_data(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load cleaned raw data; callers must split before fitting a transformer."""
    return clean_data(load_data(dataset_path), require_target=True)
