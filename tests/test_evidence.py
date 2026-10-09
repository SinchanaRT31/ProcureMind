import json

from ml.evidence import ProcurementEvidenceEngine
from test_preprocess import raw_frame


def training_frame():
    base = raw_frame()
    return base.loc[base.index.repeat(3)].reset_index(drop=True)


def test_normal_transaction_has_no_active_signals():
    train = training_frame()
    engine = ProcurementEvidenceEngine().fit(train)
    result = engine.assess(train.iloc[[0]])
    assert json.loads(result.iloc[0]["signals"]) == []


def test_high_price_spend_and_quantity_are_explained():
    train = training_frame()
    candidate = train.iloc[[0]].copy()
    candidate.loc[candidate.index[0], ["unit_price", "total_amount", "quantity"]] = [10000, 50000, 1000]
    signals = json.loads(ProcurementEvidenceEngine().fit(train).assess(candidate).iloc[0]["signals"])
    names = {signal["name"] for signal in signals}
    assert {"vendor_price_deviation", "vendor_spend_deviation", "quantity_deviation"} <= names


def test_unseen_vendor_category_uses_fallback_and_is_deterministic():
    train = training_frame()
    candidate = train.iloc[[0]].copy()
    candidate.loc[candidate.index[0], ["vendor_id", "item_category"]] = ["unseen", "unseen"]
    engine = ProcurementEvidenceEngine().fit(train)
    assert engine.assess(candidate).to_dict() == engine.assess(candidate).to_dict()


def test_missing_or_zero_values_do_not_create_invalid_baselines():
    train = training_frame()
    candidate = train.iloc[[0]].copy()
    candidate.loc[candidate.index[0], ["quantity", "unit_price", "total_amount"]] = [0, 0, 0]
    result = ProcurementEvidenceEngine().fit(train).assess(candidate)
    assert result.filter(like="_severity").isin(["none", "moderate", "high"]).all().all()
    assert json.loads(result.iloc[0]["signals"]) == []


def test_thresholds_come_from_training_data_and_forbidden_fields_are_unused():
    train = training_frame()
    engine = ProcurementEvidenceEngine().fit(train)
    assert set(engine.thresholds_) == {
        "vendor_price_deviation", "vendor_category_price_deviation", "department_category_price_deviation",
        "vendor_spend_deviation", "department_spend_deviation", "quantity_deviation",
        "invoice_delay_deviation", "payment_timing_deviation",
    }
    train["risk_score"] = 999
    train["potential_savings"] = 999999
    assert ProcurementEvidenceEngine().fit(train).thresholds_ == engine.thresholds_
