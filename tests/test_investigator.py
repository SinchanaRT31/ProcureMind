from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
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


def request(base: str, path: str, *, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    api_request = Request(base + path, data=data, method=method)
    if data is not None:
        api_request.add_header("Content-Type", "application/json")
    try:
        with urlopen(api_request, timeout=5) as response:
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


def test_trusted_transaction_review_open_is_exact_idempotent_and_persistent(report_api, monkeypatch):
    base, _, _, _ = report_api
    lookup_calls = []
    original_lookup = backend.INVESTIGATOR.lookup_transaction

    def tracked_lookup(transaction_id):
        lookup_calls.append(transaction_id)
        return original_lookup(transaction_id)

    monkeypatch.setattr(backend.INVESTIGATOR, "lookup_transaction", tracked_lookup)
    monkeypatch.setattr(ml.predict, "predict_procurement_data", lambda *args, **kwargs: pytest.fail("case opening must not generate a full report"))

    status, opened = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})
    assert status == 201
    case = opened["case"]
    assert case["transaction_id"] == "TXN-1234567"
    assert case["review_status"] == "OPEN"
    assert case["investigation_decision"] is None
    case_id = case["id"]

    status, repeated = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})
    assert status == 200
    assert repeated["case"]["id"] == case_id
    assert lookup_calls == ["TXN-1234567"]

    status, updated = request(base, f"/api/cases/{case_id}", method="PATCH", body={
        "review_status": "REVIEWING",
        "investigation_decision": "NEEDS_MORE_INFORMATION",
        "note": "Request the signed purchase order.",
    })
    assert status == 200
    assert updated["case"]["review_status"] == "REVIEWING"
    assert updated["case"]["investigation_decision"] == "NEEDS_MORE_INFORMATION"
    assert "Request the signed purchase order." in updated["case"]["investigation_notes"]

    status, detail = request(base, f"/api/cases/{case_id}")
    assert status == 200
    assert detail["case"] == updated["case"]
    status, history = request(base, f"/api/cases/{case_id}/history")
    assert status == 200
    assert history["review_history"] == updated["case"]["review_history"]


def test_trusted_case_open_rejects_invalid_missing_and_unknown_ids(report_api):
    base = report_api[0]
    assert request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-"})[0] == 400
    assert request(base, "/api/cases", method="POST", body={})[0] == 400
    status, payload = request(base, "/api/cases", method="POST", body={
        "transaction_id": "TXN-9999999",
        "dataset_path": "/etc/passwd",
        "transaction": raw_row("TXN-9999999"),
    })
    assert status == 404 and "error" in payload
    status, cases = request(base, "/api/cases")
    assert status == 200
    assert not any(case["transaction_id"] in {"TXN-", "TXN-9999999"} for case in cases["cases"])


def test_trusted_case_open_rejects_duplicate_and_unavailable_source(report_api, monkeypatch):
    base, dataset_path, _, _ = report_api
    pd.DataFrame([raw_row("TXN-1234567"), raw_row("TXN-1234567")]).to_csv(dataset_path, index=False)
    status, payload = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})
    assert status == 409 and "error" in payload
    cases = request(base, "/api/cases")[1]["cases"]
    assert not any(case["transaction_id"] == "TXN-1234567" for case in cases)

    monkeypatch.setattr(
        backend.INVESTIGATOR,
        "lookup_transaction",
        lambda _transaction_id: (_ for _ in ()).throw(investigator.TransactionSourceUnavailableError("unavailable")),
    )
    status, payload = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-12345678"})
    assert status == 503 and "error" in payload
    cases = request(base, "/api/cases")[1]["cases"]
    assert not any(case["transaction_id"] in {"TXN-1234567", "TXN-12345678"} for case in cases)


def test_concurrent_trusted_case_open_returns_one_active_case(report_api, monkeypatch):
    base = report_api[0]
    request_count = 8
    lookup_barrier = threading.Barrier(request_count)
    original_lookup = backend.INVESTIGATOR.lookup_transaction

    def synchronized_lookup(transaction_id):
        lookup_barrier.wait(timeout=5)
        return original_lookup(transaction_id)

    monkeypatch.setattr(backend.INVESTIGATOR, "lookup_transaction", synchronized_lookup)

    def open_case(_):
        return request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})

    with ThreadPoolExecutor(max_workers=request_count) as pool:
        responses = list(pool.map(open_case, range(request_count)))

    assert all(status in {200, 201} for status, _ in responses)
    case_ids = {payload["case"]["id"] for _, payload in responses}
    assert len(case_ids) == 1
    assert sum(status == 201 for status, _ in responses) == 1
    cases = request(base, "/api/cases")[1]["cases"]
    active = [case for case in cases if case["transaction_id"] == "TXN-1234567" and case["review_status"] != "RESOLVED"]
    assert len(active) == 1
    assert active[0]["id"] in case_ids


def test_resolved_trusted_case_reopens_as_new_case_without_changing_old_history(report_api, monkeypatch):
    base = report_api[0]
    lookup_calls = []
    original_lookup = backend.INVESTIGATOR.lookup_transaction

    def tracked_lookup(transaction_id):
        lookup_calls.append(transaction_id)
        return original_lookup(transaction_id)

    monkeypatch.setattr(backend.INVESTIGATOR, "lookup_transaction", tracked_lookup)
    status, first = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})
    assert status == 201
    resolved_id = first["case"]["id"]
    status, resolved = request(base, f"/api/cases/{resolved_id}", method="PATCH", body={
        "review_status": "RESOLVED",
        "investigation_decision": "NEEDS_MORE_INFORMATION",
        "note": "Original investigation completed.",
    })
    assert status == 200
    original_record = resolved["case"]
    original_history = original_record["review_history"]

    status, reopened = request(base, "/api/cases", method="POST", body={"transaction_id": "TXN-1234567"})
    assert status == 201
    new_record = reopened["case"]
    assert new_record["id"] != resolved_id
    assert new_record["transaction_id"] == "TXN-1234567"
    assert new_record["review_status"] == "OPEN"
    assert len(new_record["review_history"]) == 1
    assert new_record["review_history"][0]["event_type"] == "investigation_opened"

    status, reloaded_old = request(base, f"/api/cases/{resolved_id}")
    assert status == 200
    assert reloaded_old["case"] == original_record
    assert reloaded_old["case"]["review_history"] == original_history
    status, cases = request(base, "/api/cases")
    assert status == 200
    transaction_cases = [case for case in cases["cases"] if case["transaction_id"] == "TXN-1234567"]
    assert {case["id"] for case in transaction_cases} == {resolved_id, new_record["id"]}
    assert sum(case["review_status"] != "RESOLVED" for case in transaction_cases) == 1
    assert lookup_calls == ["TXN-1234567", "TXN-1234567"]


def write_trusted_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def current_csv_lookup(path: Path, transaction_id: str) -> pd.DataFrame:
    matches = []
    for chunk in pd.read_csv(
        path,
        usecols=investigator.RAW_TRANSACTION_COLUMNS,
        dtype={"transaction_id": "string"},
        chunksize=investigator.CSV_CHUNK_SIZE,
    ):
        matches.extend(chunk.loc[chunk["transaction_id"] == transaction_id].to_dict("records"))
    return pd.DataFrame(matches, columns=investigator.RAW_TRANSACTION_COLUMNS).loc[:, investigator.RAW_TRANSACTION_COLUMNS]


def test_sqlite_lookup_matches_csv_values_and_preserves_only_allowlisted_columns(tmp_path):
    dataset_path = tmp_path / "trusted.csv"
    row = raw_row("TXN-A_2")
    row.update({"vendor_rating": None, "quantity": "unusual quantity", "unit_price": 0, "vendor_location": "North, West"})
    write_trusted_dataset(dataset_path, [row])

    actual = InvestigatorReportService(dataset_path).lookup_transaction("TXN-A_2")
    expected = current_csv_lookup(dataset_path, "TXN-A_2")

    assert list(actual.columns) == list(investigator.RAW_TRANSACTION_COLUMNS)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert actual.loc[0, "quantity"] == "unusual quantity"
    assert pd.isna(actual.loc[0, "vendor_rating"])
    assert "actual_anomaly" not in actual.columns


def test_sqlite_lookup_preserves_not_found_and_duplicate_statuses_across_chunks(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [
        raw_row("TXN-FIRST1"),
        raw_row("TXN-OTHER1"),
        raw_row("TXN-FIRST1"),
        raw_row("TXN-FIRST1"),
    ])
    monkeypatch.setattr(investigator, "CSV_CHUNK_SIZE", 1)
    service = InvestigatorReportService(dataset_path)

    with pytest.raises(investigator.TransactionNotFoundError):
        service.lookup_transaction("TXN-ABSENT1")
    with pytest.raises(investigator.AmbiguousTransactionError):
        service.lookup_transaction("TXN-FIRST1")
    with sqlite3.connect(service._index_path) as connection:
        stored_rows = connection.execute(
            "SELECT COUNT(*) FROM transactions WHERE transaction_id = ? COLLATE BINARY", ("TXN-FIRST1",)
        ).fetchone()[0]
        stored_columns = [row[1] for row in connection.execute("PRAGMA table_info(transactions)")][1:]
    assert stored_rows == 3
    assert stored_columns == list(investigator.RAW_TRANSACTION_COLUMNS)


@pytest.mark.parametrize("contents,expected", [
    ("", "unavailable"),
    ("transaction_id,vendor_id\nTXN-EMPTY01,V-1\n", "unavailable"),
    ("transaction_id,vendor_id,vendor_rating,department_id,item_category,quantity,unit_price,total_amount,order_date,invoice_date,due_date,payment_status,purchase_type,vendor_location\n", "not_found"),
])
def test_empty_or_malformed_dataset_has_controlled_lookup_result(tmp_path, contents, expected):
    dataset_path = tmp_path / "trusted.csv"
    dataset_path.write_text(contents, encoding="utf-8")
    service = InvestigatorReportService(dataset_path)

    error = investigator.TransactionSourceUnavailableError if expected == "unavailable" else investigator.TransactionNotFoundError
    with pytest.raises(error):
        service.lookup_transaction("TXN-EMPTY01")


def test_failed_atomic_publish_leaves_no_partial_or_stale_index(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-FAIL001")])
    service = InvestigatorReportService(dataset_path)

    def fail_replace(source, destination):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(investigator.os, "replace", fail_replace)
    with pytest.raises(investigator.TransactionSourceUnavailableError):
        service.lookup_transaction("TXN-FAIL001")
    assert not service._index_path.exists()
    assert list(service._index_path.parent.glob("*.tmp")) == []


def test_corrupt_index_is_rebuilt_from_the_trusted_source(tmp_path):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-CORRUPT1")])
    service = InvestigatorReportService(dataset_path)
    assert service.lookup_transaction("TXN-CORRUPT1").iloc[0]["transaction_id"] == "TXN-CORRUPT1"

    service._index_path.write_bytes(b"not a sqlite database")
    rebuilt = service.lookup_transaction("TXN-CORRUPT1")
    assert rebuilt.iloc[0]["transaction_id"] == "TXN-CORRUPT1"
    with sqlite3.connect(service._index_path) as connection:
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)


def test_valid_index_content_digest_matches_stored_rows(tmp_path):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-DIGEST01"), raw_row("TXN-DIGEST02")])
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-DIGEST01")

    assert service._index_matches_source(investigator._sha256_file(dataset_path))


def test_full_content_digest_is_not_recomputed_on_ordinary_lookup(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-ONCE001")])
    original_digest = investigator._index_content_digest
    digest_calls = 0

    def counted_digest(connection):
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(connection)

    monkeypatch.setattr(investigator, "_index_content_digest", counted_digest)
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-ONCE001")
    assert digest_calls == 1
    service.lookup_transaction("TXN-ONCE001")
    assert digest_calls == 1


@pytest.mark.parametrize("corruption", ["field", "delete_with_updated_count", "transaction_id"])
def test_content_validation_rejects_valid_sqlite_with_altered_rows(tmp_path, corruption):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-CONTENT1"), raw_row("TXN-CONTENT2")])
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-CONTENT1")

    with sqlite3.connect(service._index_path) as connection:
        if corruption == "field":
            connection.execute(
                "UPDATE transactions SET vendor_location = ? WHERE transaction_id = ?",
                ("altered value", "TXN-CONTENT1"),
            )
        elif corruption == "delete_with_updated_count":
            connection.execute("DELETE FROM transactions WHERE transaction_id = ?", ("TXN-CONTENT2",))
            connection.execute("UPDATE metadata SET value = '1' WHERE key = 'row_count'")
        else:
            connection.execute(
                "UPDATE transactions SET transaction_id = ? WHERE transaction_id = ?",
                ("TXN-REPLACED1", "TXN-CONTENT1"),
            )
        assert connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
        assert connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == int(
            connection.execute("SELECT value FROM metadata WHERE key = 'row_count'").fetchone()[0]
        )

    assert not service._index_matches_source(investigator._sha256_file(dataset_path))


def test_digest_distinguishes_null_empty_and_field_boundaries():
    def digest_values(*values):
        digest = hashlib.sha256()
        for value in values:
            investigator._update_digest_value(digest, value)
        return digest.digest()

    assert digest_values(None) != digest_values("")
    assert digest_values("ab", "c") != digest_values("a", "bc")


def test_valid_looking_altered_index_is_rebuilt_before_lookup(tmp_path):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-REPAIR01")])
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-REPAIR01")

    with sqlite3.connect(service._index_path) as connection:
        connection.execute(
            "UPDATE transactions SET vendor_location = ? WHERE transaction_id = ?",
            ("tampered", "TXN-REPAIR01"),
        )
    stat = service._index_path.stat()
    os.utime(service._index_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    repaired = service.lookup_transaction("TXN-REPAIR01")
    assert repaired.iloc[0]["vendor_location"] == "North"
    assert service._index_matches_source(investigator._sha256_file(dataset_path))


def test_failed_integrity_rebuild_does_not_serve_altered_index(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row("TXN-NOREPAIR1")])
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-NOREPAIR1")

    with sqlite3.connect(service._index_path) as connection:
        connection.execute(
            "UPDATE transactions SET vendor_location = ? WHERE transaction_id = ?",
            ("tampered", "TXN-NOREPAIR1"),
        )
    stat = service._index_path.stat()
    os.utime(service._index_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    monkeypatch.setattr(investigator.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("publish failed")))

    with pytest.raises(investigator.TransactionSourceUnavailableError):
        service.lookup_transaction("TXN-NOREPAIR1")
    with sqlite3.connect(service._index_path) as connection:
        assert connection.execute(
            "SELECT vendor_location FROM transactions WHERE transaction_id = ?", ("TXN-NOREPAIR1",)
        ).fetchone()[0] == "tampered"


def test_source_change_invalidates_and_rebuilds_index(tmp_path):
    dataset_path = tmp_path / "trusted.csv"
    original = raw_row("TXN-CHANGE1")
    write_trusted_dataset(dataset_path, [original])
    service = InvestigatorReportService(dataset_path)
    assert service.lookup_transaction("TXN-CHANGE1").iloc[0]["vendor_location"] == "North"

    changed = dict(original, vendor_location="Updated location")
    write_trusted_dataset(dataset_path, [changed])
    stat = dataset_path.stat()
    os.utime(dataset_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    assert service.lookup_transaction("TXN-CHANGE1").iloc[0]["vendor_location"] == "Updated location"


def test_failed_rebuild_never_serves_an_old_index(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    original = raw_row("TXN-STALE001")
    write_trusted_dataset(dataset_path, [original])
    service = InvestigatorReportService(dataset_path)
    service.lookup_transaction("TXN-STALE001")
    old_index = service._index_path.read_bytes()

    write_trusted_dataset(dataset_path, [dict(original, vendor_location="New source value")])
    stat = dataset_path.stat()
    os.utime(dataset_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    monkeypatch.setattr(investigator.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("publish failed")))

    with pytest.raises(investigator.TransactionSourceUnavailableError):
        service.lookup_transaction("TXN-STALE001")
    assert service._index_path.read_bytes() == old_index


def test_concurrent_index_initialization_and_lookup_are_serialized(tmp_path, monkeypatch):
    dataset_path = tmp_path / "trusted.csv"
    write_trusted_dataset(dataset_path, [raw_row(f"TXN-THREAD{i:04d}") for i in range(12)])
    original_build = InvestigatorReportService._build_index
    build_count = 0
    build_count_lock = threading.Lock()

    def counted_build(self, *args, **kwargs):
        nonlocal build_count
        with build_count_lock:
            build_count += 1
        return original_build(self, *args, **kwargs)

    monkeypatch.setattr(InvestigatorReportService, "_build_index", counted_build)
    services = [InvestigatorReportService(dataset_path) for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda pair: pair[0].lookup_transaction(f"TXN-THREAD{pair[1]:04d}").iloc[0]["transaction_id"],
            [(service, index) for index, service in enumerate(services)],
        ))

    assert results == [f"TXN-THREAD{i:04d}" for i in range(8)]
    assert build_count == 1
    with ThreadPoolExecutor(max_workers=8) as executor:
        repeated = list(executor.map(
            lambda index: services[index].lookup_transaction(f"TXN-THREAD{index:04d}").iloc[0]["transaction_id"],
            range(8),
        ))
    assert repeated == results
