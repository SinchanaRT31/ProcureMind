import pandas as pd

from ml.preprocess import ProcurementFeatureTransformer, clean_data


def raw_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4"], "vendor_id": ["v1", "v1", "v2", "v3"],
        "vendor_rating": [4, 4, 3, 5], "department_id": ["d1", "d1", "d2", "d2"],
        "item_category": ["c1", "c1", "c2", "c3"], "quantity": [2, 4, 3, 5],
        "unit_price": [10, 20, 30, 40], "total_amount": [20, 80, 90, 200],
        "order_date": ["2025-01-01"] * 4, "invoice_date": ["2025-01-03"] * 4,
        "due_date": ["2025-01-13"] * 4, "payment_status": ["Paid"] * 4,
        "purchase_type": ["Equipment"] * 4, "vendor_location": ["Bengaluru"] * 4,
        "is_anomaly": [0, 0, 1, 0],
    })


def test_transformer_handles_missing_values_and_zero_denominators():
    train = raw_frame().iloc[:3].copy()
    train.loc[0, ["quantity", "unit_price", "total_amount"]] = 0
    features = ProcurementFeatureTransformer().fit(train).transform(train)
    assert features.notna().all().all()
    assert not features.isin([float("inf"), float("-inf")]).any().any()


def test_transformer_uses_only_train_baselines_for_unseen_group():
    transformer = ProcurementFeatureTransformer().fit(raw_frame().iloc[:2])
    result = transformer.transform(raw_frame().iloc[2:])
    assert result.loc[2, "amount_vs_vendor_baseline"] == 90 / 50


def test_feature_order_is_stable():
    transformer = ProcurementFeatureTransformer().fit(raw_frame().iloc[:3])
    assert transformer.transform(raw_frame().iloc[3:]).columns.tolist() == transformer.feature_names_


def test_clean_data_can_score_rows_without_labels():
    assert "is_anomaly" not in clean_data(raw_frame().drop(columns="is_anomaly")).columns
