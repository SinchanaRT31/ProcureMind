import json

import pandas as pd

from ml.verification import ProcurementVerificationEngine, temporal_vendor_cadence_experiment
from tests.test_preprocess import raw_frame


def duplicated_frame() -> pd.DataFrame:
    data = raw_frame().copy()
    data.loc[1, ["vendor_id", "item_category", "quantity", "unit_price", "total_amount", "invoice_date"]] = ["v1", "c1", 2, 10, 20, "2025-01-02"]
    return data


def test_exact_duplicate_candidate_and_different_vendor_or_date_are_handled():
    data = duplicated_frame()
    engine = ProcurementVerificationEngine().fit(data.iloc[:1])
    result = engine.assess(data.iloc[1:])
    assert len(json.loads(result.iloc[0]["duplicate_candidates"])) == 1
    far = data.iloc[[1]].copy(); far.loc[1, "invoice_date"] = "2025-02-01"
    assert json.loads(engine.assess(far).iloc[0]["duplicate_candidates"]) == []
    other = data.iloc[[1]].copy(); other.loc[1, "vendor_id"] = "other"
    assert json.loads(engine.assess(other).iloc[0]["duplicate_candidates"]) == []


def test_multiple_candidates_and_date_consistency_checks():
    data = pd.concat([duplicated_frame(), duplicated_frame().iloc[[1]]], ignore_index=True)
    data.loc[4, "transaction_id"] = "t5"
    engine = ProcurementVerificationEngine().fit(data.iloc[:1])
    output = engine.assess(data.iloc[1:])
    assert any(json.loads(value) for value in output["duplicate_candidates"])
    invalid = data.iloc[[1]].copy()
    invalid.loc[1, ["invoice_date", "due_date"]] = ["2024-12-31", "2024-12-30"]
    checks = json.loads(engine.assess(invalid).iloc[0]["consistency_checks"])
    assert {item["check"] for item in checks} == {"invoice_before_order", "due_before_invoice"}


def test_valid_and_missing_dates_do_not_create_false_consistency_issue():
    data = raw_frame()
    engine = ProcurementVerificationEngine().fit(data)
    assert json.loads(engine.assess(data.iloc[[0]]).iloc[0]["consistency_checks"]) == []
    missing = data.iloc[[0]].copy(); missing.loc[0, "invoice_date"] = None
    assert json.loads(engine.assess(missing).iloc[0]["consistency_checks"]) == []


def test_cadence_experiment_is_chronological_and_does_not_use_labels():
    data = raw_frame().copy()
    data.loc[:, "order_date"] = ["2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01"]
    report = temporal_vendor_cadence_experiment(data, cutoff="2025-01-01")
    assert report["historical_rows"] == 2
    assert report["later_rows"] == 2
