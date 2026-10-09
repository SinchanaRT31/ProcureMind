from __future__ import annotations

import json
import shutil
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

import backend.main as backend
import backend.investigator as investigator
import ml.predict
import ml.preprocess as preprocess
from backend.investigator import InvestigatorReportService
from backend.reviews import InvestigationStore
from ml.preprocess import REQUIRED_RAW_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
OUTCOME_COLUMNS = [
    "is_anomaly", "anomaly_type", "risk_score", "risk_level",
    "investigation_status", "potential_savings", "actual_anomaly",
]


def raw_row(transaction_id: str) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "vendor_id": "V-1",
        "vendor_rating": 4,
        "department_id": "D-1",
        "item_category": "Office Supplies",
        "quantity": 2,
        "unit_price": 10.0,
        "total_amount": 20.0,
        "order_date": "2024-01-01",
        "invoice_date": "2024-01-02",
        "due_date": "2024-02-01",
        "payment_status": "Pending",
        "purchase_type": "Standard",
        "vendor_location": "North",
        "is_anomaly": 1,
        "anomaly_type": "fixture-label",
        "risk_score": 99,
        "risk_level": "CRITICAL",
        "investigation_status": "Closed",
        "potential_savings": 12345,
        "actual_anomaly": 1,
    }


def predictor_result(transaction_id: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "transaction_id": transaction_id,
        "predicted_anomaly": 1,
        "ml_anomaly_score": 0.71,
        "top_feature_1": "total_amount",
        "top_contribution_1": 0.08,
        "top_feature_2": "vendor_id_frequency",
        "top_contribution_2": -0.02,
        "top_feature_3": "quantity",
        "top_contribution_3": 0.0,
        "evidence": json.dumps([{
            "name": "vendor_price_deviation",
            "family": "PRICE",
            "severity": "high",
            "value": 1.8,
            "baseline": 0.4,
            "explanation": "Price differs from its learned comparison baseline.",
        }]),
        "evidence_summary": "Price differs from its learned comparison baseline. Requires verification.",
        "evidence_family_count": 1,
        "high_severity_signal_count": 1,
        "investigation_priority": "HIGH",
        "recommended_actions": json.dumps(["Compare unit price with recent vendor and category purchases."]),
        "duplicate_candidates": "[]",
        "consistency_checks": json.dumps([{"check": "amount_arithmetic_mismatch", "explanation": "Amount should be checked."}]),
        "verification_signals": json.dumps([{"name": "amount_arithmetic_mismatch", "severity": "moderate"}]),
    }])


@pytest.fixture
def report_api(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    pd.DataFrame([raw_row("TXN-1234567"), raw_row("TXN-12345678")]).to_csv(dataset_path, index=False)
    sample_path = tmp_path / "sample_data.json"
    shutil.copyfile(ROOT / "backend" / "data" / "sample_data.json", sample_path)
    store_path = tmp_path / "investigations.json"
    monkeypatch.setattr(backend, "DATA_FILE", sample_path)
    monkeypatch.setattr(backend, "REVIEW_STORE", InvestigationStore(sample_path, store_path))
    monkeypatch.setattr(backend, "INVESTIGATOR", InvestigatorReportService(dataset_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), backend.FrontendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", dataset_path, sample_path, store_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def request(base: str, path: str) -> tuple[int, dict]:
    try:
        with urlopen(Request(base + path), timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_report_endpoint_uses_exact_id_allowlisted_predictor_input_and_separate_sections(report_api, monkeypatch):
    base, _, _, _ = report_api
    seen: dict[str, object] = {}

    def fake_predictor(rows, *, include_risk, include_explanations):
        seen["columns"] = list(rows.columns)
        seen["values"] = rows.iloc[0].to_dict()
        seen["flags"] = (include_risk, include_explanations)
        return predictor_result("TXN-1234567")

    monkeypatch.setattr(ml.predict, "predict_procurement_data", fake_predictor)
    status, payload = request(base, "/api/investigations/report?transaction_id=TXN-1234567")
    assert status == 200
    report = payload["report"]
    assert report["transaction_id"] == "TXN-1234567"
    assert report["schema_version"] == "1.0"
    assert seen["columns"] == REQUIRED_RAW_COLUMNS
    assert not set(OUTCOME_COLUMNS).intersection(seen["columns"])
    assert seen["flags"] == (True, True)
    assert report["why_flagged"]["model_prediction"]["anomaly_score"] == 0.71
    assert report["why_flagged"]["model_explanation"]["contributions"][0] == {
        "feature": "total_amount", "contribution": 0.08, "direction": "raises_anomaly_score",
    }
    assert report["supporting_findings"][0]["source"] == "risk_engine.selected_evidence"
    assert report["supporting_findings"][0]["findings"][0]["name"] == "vendor_price_deviation"
    assert report["inconclusive_or_conflicting"][0]["consistency_checks"][0]["check"] == "amount_arithmetic_mismatch"
    assert report["follow_up_checks"]["checks"] == ["Compare unit price with recent vendor and category purchases."]
    serialized = json.dumps(report).lower()
    assert not any(value in serialized for value in ("fixture-label", "actual_anomaly", "potential_savings", "risk_score"))
    assert "fraud" not in report["why_flagged"]["model_prediction"]


@pytest.mark.parametrize("path", [
    "/api/investigations/report",
    "/api/investigations/report?transaction_id=",
    "/api/investigations/report?transaction_id=tx-1234567",
    "/api/investigations/report?transaction_id=TXN-1234567&transaction_id=TXN-12345678",
    "/api/investigations/report?transaction_id=TXN-1234567&path=/etc/passwd",
])
def test_report_endpoint_rejects_missing_or_malformed_parameters(report_api, path):
    status, payload = request(report_api[0], path)
    assert status == 400
    assert "error" in payload


def test_report_endpoint_rejects_prefix_and_fuzzy_matches(report_api, monkeypatch):
    monkeypatch.setattr(ml.predict, "predict_procurement_data", lambda *args, **kwargs: pytest.fail("predictor must not run"))
    assert request(report_api[0], "/api/investigations/report?transaction_id=TXN-123456")[0] == 404
    assert request(report_api[0], "/api/investigations/report?transaction_id=TXN-1234569")[0] == 404


def test_report_endpoint_rejects_duplicate_exact_matches_as_conflict(report_api):
    base, dataset_path, _, _ = report_api
    duplicated = pd.DataFrame([raw_row("TXN-1234567"), raw_row("TXN-1234567")])
    duplicated.to_csv(dataset_path, index=False)
    assert request(base, "/api/investigations/report?transaction_id=TXN-1234567")[0] == 409


def test_prediction_id_mismatch_returns_controlled_internal_error(report_api, monkeypatch):
    monkeypatch.setattr(ml.predict, "predict_procurement_data", lambda *args, **kwargs: predictor_result("TXN-12345678"))
    status, payload = request(report_api[0], "/api/investigations/report?transaction_id=TXN-1234567")
    assert status == 500
    assert payload == {"error": "Investigation report could not be generated."}


@pytest.mark.parametrize("failure", [FileNotFoundError("private path"), ImportError("private detail")])
def test_missing_artifacts_or_explanation_dependency_return_controlled_503(report_api, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(ml.predict, "predict_procurement_data", unavailable)
    status, payload = request(report_api[0], "/api/investigations/report?transaction_id=TXN-1234567")
    assert status == 503
    assert payload == {"error": "Investigation report capability is unavailable."}


def test_unavailable_findings_are_not_fabricated(monkeypatch):
    service = InvestigatorReportService("unused.csv")
    service.lookup_transaction = lambda transaction_id: pd.DataFrame([raw_row(transaction_id)]).loc[:, REQUIRED_RAW_COLUMNS]
    monkeypatch.setattr(ml.predict, "predict_procurement_data", lambda *args, **kwargs: pd.DataFrame([{
        "transaction_id": "TXN-1234567",
        "predicted_anomaly": 0,
        "ml_anomaly_score": 0.4,
    }]))
    report = service.generate_report("TXN-1234567")
    assert report["why_flagged"]["model_explanation"] == {"status": "unavailable", "source": "SHAP"}
    assert report["supporting_findings"][0]["status"] == "unavailable"
    assert report["inconclusive_or_conflicting"] == [{
        "source": "verification_engine", "status": "unavailable",
    }]
    assert report["follow_up_checks"]["status"] == "unavailable"


def test_mutating_ml_raw_column_list_cannot_expand_investigator_allowlist(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    pd.DataFrame([raw_row("TXN-1234567")]).to_csv(dataset_path, index=False)
    service = InvestigatorReportService(dataset_path)
    ml_columns = preprocess.REQUIRED_RAW_COLUMNS
    assert tuple(ml_columns) == investigator.RAW_TRANSACTION_COLUMNS

    selected_columns: list[tuple[str, ...]] = []
    predictor_columns: list[list[str]] = []
    original_read_csv = investigator.pd.read_csv

    def capture_read_csv(*args, **kwargs):
        selected_columns.append(tuple(kwargs["usecols"]))
        return original_read_csv(*args, **kwargs)

    def capture_predictor(rows, *, include_risk, include_explanations):
        predictor_columns.append(list(rows.columns))
        return predictor_result("TXN-1234567")

    monkeypatch.setattr(investigator.pd, "read_csv", capture_read_csv)
    monkeypatch.setattr(ml.predict, "predict_procurement_data", capture_predictor)
    ml_columns.append("actual_anomaly")
    try:
        report = service.generate_report("TXN-1234567")
    finally:
        ml_columns.pop()

    assert selected_columns == [investigator.RAW_TRANSACTION_COLUMNS]
    assert predictor_columns == [list(investigator.RAW_TRANSACTION_COLUMNS)]
    assert "actual_anomaly" not in json.dumps(report)


def test_report_generation_does_not_change_phase6_review_store(report_api, monkeypatch):
    base, _, sample_path, store_path = report_api
    before_sample = sample_path.read_bytes()
    case = backend.REVIEW_STORE.get_case("PM-118")
    before_record = json.loads(json.dumps(case))
    before_store = store_path.read_bytes() if store_path.exists() else None
    monkeypatch.setattr(ml.predict, "predict_procurement_data", lambda *args, **kwargs: predictor_result("TXN-1234567"))
    status, _ = request(base, "/api/investigations/report?transaction_id=TXN-1234567")
    assert status == 200
    assert backend.REVIEW_STORE.get_case("PM-118") == before_record
    assert sample_path.read_bytes() == before_sample
    assert (store_path.read_bytes() if store_path.exists() else None) == before_store
