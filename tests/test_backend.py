from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import backend.main as backend
import backend.reviews as review_module
from backend.reviews import InvestigationStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api(tmp_path, monkeypatch):
    sample_path = tmp_path / "sample_data.json"
    shutil.copyfile(ROOT / "backend" / "data" / "sample_data.json", sample_path)
    store_path = tmp_path / "investigations.json"
    monkeypatch.setattr(backend, "DATA_FILE", sample_path)
    monkeypatch.setattr(backend, "REVIEW_STORE", InvestigationStore(sample_path, store_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), backend.FrontendHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", sample_path, store_path
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def call(base, path, *, method="GET", body=None, raw_body=None):
    data = raw_body if raw_body is not None else (json.dumps(body).encode() if body is not None else None)
    request = Request(base + path, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_existing_routes_and_legacy_case_update_remain_compatible(api):
    base, sample_path, _ = api
    original_sample = sample_path.read_bytes()
    assert call(base, "/api/health") == (200, {"status": "ok"})
    dashboard_status, dashboard = call(base, "/api/dashboard")
    assert dashboard_status == 200
    assert dashboard["transactions"]
    assert len(dashboard["cases"]) == 3
    case_status, case_payload = call(base, "/api/cases/PM-118")
    assert case_status == 200 and case_payload["case"]["transaction_id"] == "tx-2048"
    transaction_status, transactions = call(base, "/api/transactions")
    assert transaction_status == 200 and transactions["transactions"]

    status, result = call(base, "/api/cases/PM-118", method="POST", body={"status": "Review Pending", "notes": "Vendor response received."})
    assert status == 200
    assert result["case"]["status"] == "Review Pending"
    assert result["case"]["review_status"] == "OPEN"
    assert result["case"]["notes"].endswith("Vendor response received.")
    assert any(event["event_type"] == "note_added" for event in result["case"]["review_history"])
    assert sample_path.read_bytes() == original_sample  # all review writes use the separate store


def test_open_update_history_reload_and_migration_are_idempotent(api):
    base, sample_path, store_path = api
    status, opened = call(base, "/api/cases", method="POST", body={"transaction_id": "tx-1931", "reviewer_id": "reviewer-7"})
    assert status == 201
    case_id = opened["case"]["id"]
    assert opened["case"]["investigation_decision"] is None
    status, duplicate_open = call(base, "/api/cases", method="POST", body={"transaction_id": "tx-1931"})
    assert status == 200 and duplicate_open["case"]["id"] == case_id

    status, updated = call(base, f"/api/cases/{case_id}", method="PATCH", body={
        "review_status": "REVIEWING",
        "investigation_decision": "NEEDS_MORE_INFORMATION",
        "note": "Requesting supporting purchase documentation.",
        "reviewer_id": "reviewer-7",
    })
    assert status == 200
    assert updated["case"]["investigation_decision"] == "NEEDS_MORE_INFORMATION"
    assert len(updated["case"]["review_history"]) == 4
    events = updated["case"]["review_history"]
    status_event = next(event for event in events if event["event_type"] == "status_changed")
    decision_event = next(event for event in events if event["event_type"] == "decision_recorded")
    assert status_event["changes"]["from"] == "OPEN" and status_event["changes"]["to"] == "REVIEWING"
    assert decision_event["changes"]["from"] is None
    assert decision_event["changes"]["to"] == "NEEDS_MORE_INFORMATION"
    status, history = call(base, f"/api/cases/{case_id}/history")
    assert status == 200 and len(history["review_history"]) == 4

    before = {case["id"]: (case.get("investigation_decision"), case["investigation_notes"], case["review_history"]) for case in InvestigationStore(sample_path, store_path).list_cases()}
    restarted = InvestigationStore(sample_path, store_path)
    after = {case["id"]: (case.get("investigation_decision"), case["investigation_notes"], case["review_history"]) for case in restarted.list_cases()}
    assert before == after


def test_migrated_seed_decision_notes_and_history_survive_reload(api):
    base, sample_path, store_path = api
    status, migrated = call(base, "/api/cases/PM-118")
    assert status == 200
    status, updated = call(base, "/api/cases/PM-118", method="PATCH", body={
        "review_status": "REVIEWING",
        "investigation_decision": "ESCALATED",
        "note": "Seed case investigation note.",
    })
    assert status == 200
    before = updated["case"]
    reloaded = InvestigationStore(sample_path, store_path).get_case("PM-118")
    assert reloaded["investigation_decision"] == "ESCALATED"
    assert reloaded["investigation_notes"].endswith("Seed case investigation note.")
    assert reloaded["review_history"] == before["review_history"]


@pytest.mark.parametrize("body", [
    {"review_status": "CLOSED"},
    {"investigation_decision": "FRAUD"},
    {"note": "   "},
    {"unexpected": "field"},
    {"review_status": 4},
])
def test_invalid_review_updates_return_clear_400(api, body):
    base, _, _ = api
    status, result = call(base, "/api/cases/PM-118", method="POST", body=body)
    assert status == 400
    assert "error" in result


def test_legacy_label_updates_when_canonical_status_is_unchanged_without_history_noise(api):
    base, _, _ = api
    status, resolved = call(base, "/api/cases/PM-118", method="PATCH", body={"review_status": "RESOLVED"})
    assert status == 200
    assert resolved["case"]["status"] == "Resolved"
    history = resolved["case"]["review_history"]
    status_events_before = [event for event in history if event["event_type"] == "status_changed"]
    assert status_events_before[-1]["changes"]["from"] == "REVIEWING"
    assert status_events_before[-1]["changes"]["to"] == "RESOLVED"

    status, updated = call(base, "/api/cases/PM-118", method="POST", body={"status": "Confirmed"})
    assert status == 200
    assert updated["case"]["status"] == "Confirmed"
    assert updated["case"]["review_status"] == "RESOLVED"
    assert updated["case"]["review_history"] == history


def test_conflicting_status_fields_are_rejected_without_any_record_or_history_change(api):
    base, _, store_path = api
    call(base, "/api/cases")
    before_bytes = store_path.read_bytes()
    before_record = backend.REVIEW_STORE.get_case("PM-118")
    status, error = call(base, "/api/cases/PM-118", method="PATCH", body={
        "status": "Confirmed",
        "review_status": "OPEN",
        "note": "Must not be appended.",
    })
    assert status == 400 and "error" in error
    assert store_path.read_bytes() == before_bytes
    assert backend.REVIEW_STORE.get_case("PM-118") == before_record


def test_compatible_legacy_and_canonical_status_fields_are_accepted(api):
    base, _, _ = api
    status, response = call(base, "/api/cases/PM-118", method="PATCH", body={
        "status": "Confirmed",
        "review_status": "RESOLVED",
    })
    assert status == 200
    case = response["case"]
    assert case["status"] == "Confirmed"
    assert case["review_status"] == "RESOLVED"
    status_events = [event for event in case["review_history"] if event["event_type"] == "status_changed"]
    assert len(status_events) == 1
    assert status_events[0]["changes"]["from"] == "REVIEWING"
    assert status_events[0]["changes"]["to"] == "RESOLVED"


def test_malformed_non_object_unknown_case_and_unknown_transaction_errors(api):
    base, _, _ = api
    assert call(base, "/api/cases/PM-118", method="POST", raw_body=b"{")[0] == 400
    assert call(base, "/api/cases/PM-118", method="POST", body=[])[0] == 400
    status, error = call(base, "/api/cases/absent", method="PATCH", body={"review_status": "OPEN"})
    assert status == 404 and "not found" in error["error"]
    status, error = call(base, "/api/cases", method="POST", body={"transaction_id": "missing"})
    assert status == 404 and "Transaction" in error["error"]


def test_concurrent_notes_are_serialized_without_lost_updates(api):
    base, _, store_path = api
    call(base, "/api/cases")  # ensure initial migration is complete
    notes = [f"concurrent note {index}" for index in range(12)]

    def save(note):
        return call(base, "/api/cases/PM-118", method="PATCH", body={"note": note})[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(save, notes))
    assert statuses == [200] * len(notes)
    case = next(item for item in json.loads(store_path.read_text())["cases"] if item["id"] == "PM-118")
    assert all(note in case["investigation_notes"] for note in notes)
    note_events = [event for event in case["review_history"] if event["event_type"] == "note_added"]
    assert len(note_events) == len(notes)


def test_failed_write_preserves_existing_store_and_returns_json_error(api, monkeypatch):
    base, _, store_path = api
    call(base, "/api/cases")
    existing = store_path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("forced atomic replacement failure")

    monkeypatch.setattr(review_module.os, "replace", fail_replace)
    status, result = call(base, "/api/cases/PM-118", method="PATCH", body={"note": "Must not persist."})
    assert status == 500 and "error" in result
    assert store_path.read_bytes() == existing
    assert not list(store_path.parent.glob(f".{store_path.name}.*.tmp"))
    case = backend.REVIEW_STORE.get_case("PM-118")
    assert "Must not persist." not in case["investigation_notes"]


def test_malformed_persisted_store_is_not_overwritten(api):
    base, _, store_path = api
    store_path.write_text("{broken", encoding="utf-8")
    status, result = call(base, "/api/cases")
    assert status == 500 and "error" in result
    assert store_path.read_text(encoding="utf-8") == "{broken"
