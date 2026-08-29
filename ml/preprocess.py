from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "dataset" / "ProcureMind" / "procuremind_procurement_dataset.csv"
)

DATE_COLUMNS = ["order_date", "invoice_date", "due_date"]
TARGET_COLUMN = "is_anomaly"
TRANSACTION_ID_COLUMN = "transaction_id"

REQUIRED_COLUMNS = [
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
    "is_anomaly",
    "vendor_historical_avg",
    "department_historical_avg",
    "vendor_transaction_frequency",
    "vendor_price_deviation",
    "amount_vs_vendor_average",
]

POSITIVE_NUMERIC_COLUMNS = [
    "quantity",
    "unit_price",
    "total_amount",
    "vendor_historical_avg",
    "department_historical_avg",
    "vendor_transaction_frequency",
]

FEATURE_COLUMNS = [
    "vendor_rating",
    "quantity",
    "unit_price",
    "total_amount",
    "vendor_historical_avg",
    "department_historical_avg",
    "vendor_transaction_frequency",
    "vendor_price_deviation",
    "amount_vs_vendor_average",
    "invoice_delay_days",
    "payment_due_days",
    "order_month",
    "order_day_of_week",
    "is_weekend",
    "amount_vs_department_average",
    "quantity_vs_expected",
    "unit_price_vs_expected",
    "vendor_frequency_percentile",
]


def load_data(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the procurement dataset from disk."""
    path = Path(dataset_path)
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")


def _safe_ratio(
    numerator: pd.Series, denominator: pd.Series, default: float = np.nan
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    ratio = numerator.div(denominator)
    return ratio.replace([np.inf, -np.inf], default)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, coerce numeric types, and sanitize invalid values."""
    _validate_columns(df, REQUIRED_COLUMNS)

    cleaned = df.copy()

    for column in DATE_COLUMNS:
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    numeric_columns = [
        "vendor_rating",
        "quantity",
        "unit_price",
        "total_amount",
        "vendor_historical_avg",
        "department_historical_avg",
        "vendor_transaction_frequency",
        "vendor_price_deviation",
        "amount_vs_vendor_average",
        TARGET_COLUMN,
    ]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["vendor_rating"] = cleaned["vendor_rating"].where(
        cleaned["vendor_rating"].between(1, 5), np.nan
    )

    for column in POSITIVE_NUMERIC_COLUMNS:
        cleaned[column] = cleaned[column].where(cleaned[column] > 0, np.nan)

    cleaned["amount_vs_vendor_average"] = cleaned["amount_vs_vendor_average"].where(
        cleaned["amount_vs_vendor_average"] >= 0, np.nan
    )
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].fillna(0).astype(int)

    return cleaned


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Isolation Forest features without using label or risk outputs."""
    features = pd.DataFrame(index=df.index)

    for column in FEATURE_COLUMNS[:9]:
        features[column] = df[column]

    features["invoice_delay_days"] = (
        df["invoice_date"] - df["order_date"]
    ).dt.days.astype("float64")
    features["payment_due_days"] = (
        df["due_date"] - df["invoice_date"]
    ).dt.days.astype("float64")
    features["order_month"] = df["order_date"].dt.month.astype("float64")
    features["order_day_of_week"] = df["order_date"].dt.dayofweek.astype("float64")
    features["is_weekend"] = (df["order_date"].dt.dayofweek >= 5).astype("float64")

    features["amount_vs_department_average"] = _safe_ratio(
        df["total_amount"], df["department_historical_avg"]
    )

    quantity_expected = (
        df.groupby(["vendor_id", "item_category"])["quantity"].transform("median")
    )
    quantity_expected = quantity_expected.fillna(
        df.groupby("item_category")["quantity"].transform("median")
    )
    quantity_expected = quantity_expected.fillna(df["quantity"].median())
    features["quantity_vs_expected"] = _safe_ratio(df["quantity"], quantity_expected)

    unit_price_expected = (
        df.groupby(["vendor_id", "item_category"])["unit_price"].transform("median")
    )
    unit_price_expected = unit_price_expected.fillna(
        df.groupby("item_category")["unit_price"].transform("median")
    )
    unit_price_expected = unit_price_expected.fillna(df["unit_price"].median())
    features["unit_price_vs_expected"] = _safe_ratio(
        df["unit_price"], unit_price_expected
    )

    vendor_frequency_by_vendor = df.groupby("vendor_id")[
        "vendor_transaction_frequency"
    ].median()
    vendor_frequency_percentiles = vendor_frequency_by_vendor.rank(
        method="average", pct=True
    )
    features["vendor_frequency_percentile"] = df["vendor_id"].map(
        vendor_frequency_percentiles
    )

    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features[FEATURE_COLUMNS]

    median_values = features.median(numeric_only=True)
    features = features.fillna(median_values)

    return features


def preprocess_data(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return feature matrix, transaction ids, evaluation labels, and feature names."""
    raw_df = load_data(dataset_path)
    cleaned_df = clean_data(raw_df)
    X = create_features(cleaned_df)
    transaction_ids = cleaned_df[TRANSACTION_ID_COLUMN].astype(str)
    y = cleaned_df[TARGET_COLUMN].copy()
    feature_names = X.columns.tolist()
    return X, transaction_ids, y, feature_names


if __name__ == "__main__":
    X, transaction_ids, y, feature_names = preprocess_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Number of rows: {len(X)}")
    print(f"Number of generated features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    print("Missing values after preprocessing:")
    print(X.isna().sum().to_string())
    print("First 5 rows of X:")
    print(X.head().to_string())
    print(f"Number of anomaly labels in y: {int(y.sum())}")
