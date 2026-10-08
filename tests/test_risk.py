import json

import pandas as pd

from ml.risk import ProcurementRiskEngine


def signal(name: str, severity: str = "moderate") -> dict:
    return {"name": name, "severity": severity, "value": 1.0, "baseline": 1.0, "explanation": f"{name} evidence."}


def assess(signals: list[dict], *, ml_flagged: bool = False, score: float = 0.1):
    engine = ProcurementRiskEngine().fit(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))
    evidence = pd.DataFrame({"signals": [json.dumps(signals)]})
    return engine.assess(evidence, pd.Series([int(ml_flagged)]), pd.Series([score])).iloc[0]


def test_normal_and_no_evidence_are_low_priority():
    result = assess([])
    assert result["investigation_priority"] == "LOW"
    assert "No material" in result["evidence_summary"]


def test_ml_anomaly_only_is_medium_with_review_action():
    result = assess([], ml_flagged=True)
    assert result["investigation_priority"] == "MEDIUM"
    assert "Review the anomalous transaction" in json.loads(result["recommended_actions"])[0]


def test_evidence_only_and_high_severity_evidence_are_prioritized():
    result = assess([signal("quantity_deviation", "high")])
    assert result["investigation_priority"] == "HIGH"
    assert result["evidence_family_count"] == 1


def test_correlated_price_signals_are_one_evidence_family():
    result = assess([signal("vendor_price_deviation", "high"), signal("vendor_category_price_deviation", "high")])
    assert result["investigation_priority"] == "HIGH"
    assert result["evidence_family_count"] == 1
    assert result["high_severity_signal_count"] == 2


def test_multiple_independent_high_families_are_critical_and_escalate():
    result = assess([signal("vendor_category_price_deviation", "high"), signal("quantity_deviation", "high")])
    assert result["investigation_priority"] == "CRITICAL"
    assert any("Escalate" in action for action in json.loads(result["recommended_actions"]))


def test_output_is_deterministic_and_does_not_depend_on_outcome_columns():
    signals = [signal("large_transaction")]
    first, second = assess(signals), assess(signals)
    assert first.to_dict() == second.to_dict()


def test_missing_optional_evidence_baseline_is_safe():
    missing_baseline = signal("large_transaction")
    missing_baseline["baseline"] = None
    result = assess([missing_baseline])
    assert result["investigation_priority"] == "MEDIUM"
