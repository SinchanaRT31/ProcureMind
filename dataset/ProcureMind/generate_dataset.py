import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
NUM_TRANSACTIONS = 200_000
NUM_VENDORS = 800
NUM_DEPARTMENTS = 18

OUTPUT_DIR = Path(__file__).parent


# ============================================================
# RANDOM GENERATOR
# ============================================================

rng = np.random.default_rng(SEED)


# ============================================================
# MASTER DATA
# ============================================================

departments = [
    "IT",
    "Finance",
    "HR",
    "Operations",
    "Procurement",
    "Marketing",
    "Sales",
    "Manufacturing",
    "Logistics",
    "Facilities",
    "R&D",
    "Legal",
    "Customer Support",
    "Administration",
    "Security",
    "Quality",
    "Engineering",
    "Supply Chain",
]

categories = [
    "IT Hardware",
    "Software",
    "Office Supplies",
    "Consulting",
    "Industrial Equipment",
    "Raw Materials",
    "Logistics",
    "Maintenance",
    "Professional Services",
    "Travel",
    "Marketing",
    "Facilities",
    "Safety Equipment",
    "Utilities",
]

locations = [
    "Bengaluru",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Ahmedabad",
    "Kolkata",
    "Noida",
    "Gurugram",
]

payment_statuses = [
    "Paid",
    "Pending",
    "Overdue",
    "Partially Paid",
]

purchase_types = [
    "Goods",
    "Services",
    "Software",
    "Equipment",
]

investigation_statuses = [
    "Pending",
    "Under Review",
    "Resolved",
    "False Positive",
]


# ============================================================
# VENDORS
# ============================================================

vendor_ids = np.array(
    [f"V{i:04d}" for i in range(1, NUM_VENDORS + 1)]
)

vendor_names = np.array(
    [f"Vendor {i:04d}" for i in range(1, NUM_VENDORS + 1)]
)

vendor_locations = rng.choice(
    locations,
    size=NUM_VENDORS
)

vendor_ratings = np.clip(
    rng.normal(4.0, 0.55, NUM_VENDORS),
    1.5,
    5.0
)


# ============================================================
# TRANSACTION INDEXES
# ============================================================

vendor_index = rng.integers(
    0,
    NUM_VENDORS,
    NUM_TRANSACTIONS
)

department_index = rng.integers(
    0,
    NUM_DEPARTMENTS,
    NUM_TRANSACTIONS
)

category_index = rng.integers(
    0,
    len(categories),
    NUM_TRANSACTIONS
)


# ============================================================
# DATES
# ============================================================

order_dates = (
    pd.Timestamp("2023-01-01")
    + pd.to_timedelta(
        rng.integers(0, 1095, NUM_TRANSACTIONS),
        unit="D"
    )
)


invoice_dates = (
    order_dates
    + pd.to_timedelta(
        rng.integers(1, 8, NUM_TRANSACTIONS),
        unit="D"
    )
)


due_dates = (
    invoice_dates
    + pd.to_timedelta(
        rng.choice(
            [15, 30, 45, 60],
            size=NUM_TRANSACTIONS,
            p=[0.05, 0.55, 0.25, 0.15]
        ),
        unit="D"
    )
)


# ============================================================
# NORMAL PROCUREMENT BEHAVIOUR
# ============================================================

# Typical price for each category
category_base_prices = np.array([
    55_000,
    18_000,
    1_200,
    85_000,
    125_000,
    45_000,
    22_000,
    28_000,
    65_000,
    18_000,
    30_000,
    40_000,
    9_000,
    15_000,
])


quantity = np.maximum(
    1,
    rng.lognormal(
        mean=1.6,
        sigma=0.75,
        size=NUM_TRANSACTIONS
    ).round().astype(int)
)


unit_price = (
    category_base_prices[category_index]
    * rng.lognormal(
        mean=0,
        sigma=0.32,
        size=NUM_TRANSACTIONS
    )
)


# Department-specific spending behaviour
department_multiplier = rng.uniform(
    0.75,
    1.35,
    NUM_DEPARTMENTS
)

unit_price *= department_multiplier[department_index]


# Normal procurement discounts
discount = rng.uniform(
    0,
    0.12,
    NUM_TRANSACTIONS
)

unit_price *= (1 - discount)


total_amount = quantity * unit_price


# ============================================================
# CREATE BASE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "transaction_id": [
        f"TXN-{i:07d}"
        for i in range(1, NUM_TRANSACTIONS + 1)
    ],

    "po_id": [
        f"PO-{i:07d}"
        for i in range(1, NUM_TRANSACTIONS + 1)
    ],

    "invoice_id": [
        f"INV-{i:07d}"
        for i in range(1, NUM_TRANSACTIONS + 1)
    ],

    "vendor_id": vendor_ids[vendor_index],

    "vendor_name": vendor_names[vendor_index],

    "vendor_location": vendor_locations[vendor_index],

    "vendor_rating": np.round(
        vendor_ratings[vendor_index],
        2
    ),

    "department_id": [
        f"D{i + 1:02d}"
        for i in department_index
    ],

    "department_name": np.array(
        departments
    )[department_index],

    "item_category": np.array(
        categories
    )[category_index],

    "quantity": quantity,

    "unit_price": np.round(
        unit_price,
        2
    ),

    "total_amount": np.round(
        total_amount,
        2
    ),

    "order_date": order_dates,

    "invoice_date": invoice_dates,

    "due_date": due_dates,

    "payment_status": rng.choice(
        payment_statuses,
        NUM_TRANSACTIONS,
        p=[0.70, 0.16, 0.08, 0.06]
    ),

    "purchase_type": rng.choice(
        purchase_types,
        NUM_TRANSACTIONS,
        p=[0.45, 0.25, 0.15, 0.15]
    ),
})


# ============================================================
# GROUND-TRUTH ANOMALY COLUMNS
# ============================================================

df["is_anomaly"] = 0

df["anomaly_type"] = "Normal"


def mark_anomaly(mask, anomaly_name):
    """
    Mark transactions as anomalous.

    Multiple anomaly types can exist on
    the same transaction.
    """

    df.loc[mask, "is_anomaly"] = 1

    normal_mask = (
        mask
        & (df["anomaly_type"] == "Normal")
    )

    df.loc[
        normal_mask,
        "anomaly_type"
    ] = anomaly_name

    existing_mask = (
        mask
        & (df["anomaly_type"] != "Normal")
        & (~normal_mask)
    )

    df.loc[
        existing_mask,
        "anomaly_type"
    ] = (
        df.loc[
            existing_mask,
            "anomaly_type"
        ]
        + "|"
        + anomaly_name
    )


# ============================================================
# ANOMALY 1 — PRICE ANOMALIES
# ============================================================

price_mask = (
    rng.random(NUM_TRANSACTIONS)
    < 0.018
)

price_multiplier = rng.uniform(
    2.0,
    4.0,
    price_mask.sum()
)

df.loc[
    price_mask,
    "unit_price"
] *= price_multiplier

df.loc[
    price_mask,
    "total_amount"
] = (
    df.loc[
        price_mask,
        "quantity"
    ]
    * df.loc[
        price_mask,
        "unit_price"
    ]
)

mark_anomaly(
    price_mask,
    "Price Anomaly"
)


# ============================================================
# ANOMALY 2 — UNUSUAL SPENDING
# ============================================================

spending_mask = (
    rng.random(NUM_TRANSACTIONS)
    < 0.015
)

spending_multiplier = rng.integers(
    4,
    10,
    spending_mask.sum()
)

df.loc[
    spending_mask,
    "quantity"
] *= spending_multiplier

df.loc[
    spending_mask,
    "total_amount"
] = (
    df.loc[
        spending_mask,
        "quantity"
    ]
    * df.loc[
        spending_mask,
        "unit_price"
    ]
)

mark_anomaly(
    spending_mask,
    "Unusual Spending"
)


# ============================================================
# ANOMALY 3 — QUANTITY ANOMALIES
# ============================================================

quantity_mask = (
    rng.random(NUM_TRANSACTIONS)
    < 0.012
)

quantity_multiplier = rng.integers(
    3,
    8,
    quantity_mask.sum()
)

df.loc[
    quantity_mask,
    "quantity"
] *= quantity_multiplier

df.loc[
    quantity_mask,
    "total_amount"
] = (
    df.loc[
        quantity_mask,
        "quantity"
    ]
    * df.loc[
        quantity_mask,
        "unit_price"
    ]
)

mark_anomaly(
    quantity_mask,
    "Quantity Anomaly"
)


# ============================================================
# ANOMALY 4 — SUSPICIOUS VENDORS
# ============================================================

suspicious_vendor_indexes = rng.choice(
    NUM_VENDORS,
    size=35,
    replace=False
)

suspicious_vendor_mask = (
    np.isin(
        vendor_index,
        suspicious_vendor_indexes
    )
    & (
        rng.random(NUM_TRANSACTIONS)
        < 0.18
    )
)

df.loc[
    suspicious_vendor_mask,
    "vendor_rating"
] = np.clip(
    df.loc[
        suspicious_vendor_mask,
        "vendor_rating"
    ]
    - rng.uniform(
        1.0,
        2.0,
        suspicious_vendor_mask.sum()
    ),
    1.0,
    5.0
)

mark_anomaly(
    suspicious_vendor_mask,
    "Suspicious Vendor"
)


# ============================================================
# ANOMALY 5 — DUPLICATE INVOICES
# ============================================================

duplicate_count = int(
    NUM_TRANSACTIONS * 0.012
)

source_indexes = rng.choice(
    NUM_TRANSACTIONS,
    size=duplicate_count,
    replace=False
)

target_indexes = rng.choice(
    NUM_TRANSACTIONS,
    size=duplicate_count,
    replace=False
)


for source, target in zip(
    source_indexes,
    target_indexes
):

    if source == target:
        continue

    columns_to_copy = [
        "vendor_id",
        "vendor_name",
        "vendor_location",
        "vendor_rating",
        "department_id",
        "department_name",
        "item_category",
        "quantity",
        "unit_price",
        "total_amount",
    ]

    for column in columns_to_copy:

        df.at[
            target,
            column
        ] = df.at[
            source,
            column
        ]

    df.at[
        target,
        "invoice_date"
    ] = (
        df.at[
            source,
            "invoice_date"
        ]
        + pd.Timedelta(
            days=int(
                rng.integers(
                    0,
                    4
                )
            )
        )
    )


duplicate_mask = np.zeros(
    NUM_TRANSACTIONS,
    dtype=bool
)

duplicate_mask[
    target_indexes
] = True

mark_anomaly(
    duplicate_mask,
    "Duplicate Invoice"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

vendor_average = (
    df.groupby(
        "vendor_id"
    )["total_amount"]
    .transform("mean")
)

department_average = (
    df.groupby(
        "department_id"
    )["total_amount"]
    .transform("mean")
)

vendor_frequency = (
    df.groupby(
        "vendor_id"
    )["transaction_id"]
    .transform("count")
)

vendor_category_average = (
    df.groupby(
        [
            "vendor_id",
            "item_category"
        ]
    )["unit_price"]
    .transform("mean")
)


df["vendor_historical_avg"] = (
    vendor_average.round(2)
)

df["department_historical_avg"] = (
    department_average.round(2)
)

df["vendor_transaction_frequency"] = (
    vendor_frequency.astype(int)
)


df["vendor_price_deviation"] = (
    (
        df["unit_price"]
        - vendor_category_average
    )
    / (
        vendor_category_average
        + 1e-9
    )
).round(4)


df["amount_vs_vendor_average"] = (
    df["total_amount"]
    / (
        df["vendor_historical_avg"]
        + 1e-9
    )
).round(4)


# ============================================================
# RISK SCORE
# ============================================================

amount_signal = np.clip(
    (
        df["amount_vs_vendor_average"]
        - 1
    ) / 4,
    0,
    1
)

price_signal = np.clip(
    df["vendor_price_deviation"].abs()
    / 2,
    0,
    1
)

rating_signal = (
    5 - df["vendor_rating"]
) / 4

quantity_signal = np.clip(
    df["quantity"]
    / (
        df["quantity"].median()
        * 5
    ),
    0,
    1
)


risk_score = (
    28 * amount_signal
    + 25 * price_signal
    + 20 * rating_signal
    + 15 * df["is_anomaly"]
    + 12 * quantity_signal
    + rng.normal(
        0,
        4,
        NUM_TRANSACTIONS
    )
)


df["risk_score"] = np.clip(
    np.round(risk_score),
    0,
    100
).astype(int)


# ============================================================
# RISK LEVEL
# ============================================================

df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[
        -1,
        49,
        69,
        84,
        100
    ],
    labels=[
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
).astype(str)


# ============================================================
# INVESTIGATION STATUS
# ============================================================

anomaly_rows = (
    df["is_anomaly"] == 1
)

df["investigation_status"] = "Not Required"

df.loc[
    anomaly_rows,
    "investigation_status"
] = rng.choice(
    investigation_statuses,
    size=anomaly_rows.sum(),
    p=[
        0.48,
        0.28,
        0.16,
        0.08
    ]
)


# ============================================================
# POTENTIAL SAVINGS
# ============================================================

df["potential_savings"] = np.where(
    df["is_anomaly"] == 1,
    df["total_amount"]
    * rng.uniform(
        0.02,
        0.25,
        NUM_TRANSACTIONS
    ),
    0
)

df["potential_savings"] = (
    df["potential_savings"]
    .round(2)
)


# ============================================================
# CLEAN DATE COLUMNS
# ============================================================

df["order_date"] = pd.to_datetime(
    df["order_date"]
).dt.strftime("%Y-%m-%d")

df["invoice_date"] = pd.to_datetime(
    df["invoice_date"]
).dt.strftime("%Y-%m-%d")

df["due_date"] = pd.to_datetime(
    df["due_date"]
).dt.strftime("%Y-%m-%d")


# ============================================================
# SAVE DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "procuremind_procurement_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 60)
print("PROCUREMIND DATASET GENERATED")
print("=" * 60)

print(
    f"Transactions : {len(df):,}"
)

print(
    f"Vendors      : "
    f"{df['vendor_id'].nunique():,}"
)

print(
    f"Departments  : "
    f"{df['department_id'].nunique():,}"
)

print(
    f"Categories   : "
    f"{df['item_category'].nunique():,}"
)

print(
    f"Anomalies    : "
    f"{df['is_anomaly'].sum():,}"
)

print(
    f"Anomaly rate : "
    f"{df['is_anomaly'].mean() * 100:.2f}%"
)

print()
print("Anomaly distribution:")
print(
    df["anomaly_type"]
    .value_counts()
)

print()
print(
    f"Saved to:\n{output_file}"
)

print("=" * 60)