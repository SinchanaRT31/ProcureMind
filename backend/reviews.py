"""Persistent human-review records for the sample dashboard workflow."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROCESS_LOCK = threading.RLock()
REVIEW_STATUSES = {"OPEN", "REVIEWING", "RESOLVED"}
INVESTIGATION_DECISIONS = {
    "CONFIRMED_ISSUE",
    "NO_ISSUE_FOUND",
    "NEEDS_MORE_INFORMATION",
    "ESCALATED",
}
LEGACY_STATUSES = {
    "Under Investigation": "REVIEWING",
    "Review Pending": "OPEN",
    "Confirmed": "RESOLVED",
    "False Positive": "RESOLVED",
}
MAX_NOTE_LENGTH = 10_000


class ReviewValidationError(ValueError):
    pass


class ReviewNotFoundError(LookupError):
    pass


class ReviewStoreError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _optional_text(value: Any, field: str, *, maximum: int = 200) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ReviewValidationError(f"{field} must be a string.")
    value = value.strip()
    if len(value) > maximum:
        raise ReviewValidationError(f"{field} must be at most {maximum} characters.")
    return value or None


class InvestigationStore:
    """JSON store with process-wide serialized updates and atomic replacement."""

    def __init__(self, sample_data_path: str | Path, store_path: str | Path):
        self.sample_data_path = Path(sample_data_path)
        self.store_path = Path(store_path)

    def list_cases(self) -> list[dict[str, Any]]:
        with PROCESS_LOCK:
            return [self._copy(case) for case in self._load_cases()]

    def get_case(self, case_id: str) -> dict[str, Any]:
        with PROCESS_LOCK:
            case = next((item for item in self._load_cases() if item["id"] == case_id), None)
            if case is None:
                raise ReviewNotFoundError(f"Case '{case_id}' was not found.")
            return self._copy(case)

    def open_case(self, transaction_id: str, reviewer_id: str | None = None) -> tuple[dict[str, Any], bool]:
        transaction_id = _optional_text(transaction_id, "transaction_id", maximum=200)
        if transaction_id is None:
            raise ReviewValidationError("transaction_id is required.")
        reviewer_id = _optional_text(reviewer_id, "reviewer_id")
        with PROCESS_LOCK:
            cases = self._load_cases()
            sample = self._read_sample()
            transactions = sample.get("transactions")
            if not isinstance(transactions, list) or any(not isinstance(item, dict) for item in transactions):
                raise ReviewStoreError("Sample data does not contain a valid transactions list.")
            known_transactions = {str(item.get("id")) for item in transactions if item.get("id") is not None}
            if transaction_id not in known_transactions:
                raise ReviewNotFoundError(f"Transaction '{transaction_id}' was not found.")
            active = next(
                (case for case in cases if case["transaction_id"] == transaction_id and case["review_status"] != "RESOLVED"),
                None,
            )
            if active is not None:
                return self._copy(active), False
            numbers = [int(case["id"].removeprefix("PM-")) for case in cases if case["id"].startswith("PM-") and case["id"][3:].isdigit()]
            case_id = f"PM-{max(numbers, default=0) + 1}"
            now = utc_now()
            case = {
                "id": case_id,
                "transaction_id": transaction_id,
                "owner": None,
                "status": "Review Pending",
                "priority": None,
                "notes": "",
                "review_status": "OPEN",
                "investigation_decision": None,
                "reviewer_id": reviewer_id,
                "investigation_notes": "",
                "created_at": now,
                "updated_at": now,
                "review_history": [],
            }
            self._append_event(case, "investigation_opened", reviewer_id, {"transaction_id": transaction_id})
            cases.append(case)
            self._write_cases(cases)
            return self._copy(case), True

    def update_case(self, case_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(changes, dict):
            raise ReviewValidationError("Request body must be a JSON object.")
        allowed = {"status", "review_status", "investigation_decision", "reviewer_id", "note", "notes", "investigation_notes"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ReviewValidationError(f"Unsupported field(s): {', '.join(unknown)}.")
        reviewer_id = _optional_text(changes.get("reviewer_id"), "reviewer_id")
        status_present = "review_status" in changes or "status" in changes
        new_status: str | None = None
        legacy_status: str | None = None
        if "review_status" in changes:
            candidate = changes["review_status"]
            if not isinstance(candidate, str) or candidate not in REVIEW_STATUSES:
                raise ReviewValidationError(f"review_status must be one of: {', '.join(sorted(REVIEW_STATUSES))}.")
            new_status = candidate
        if "status" in changes:
            candidate = changes["status"]
            if not isinstance(candidate, str):
                raise ReviewValidationError("status must be a string.")
            if candidate in LEGACY_STATUSES:
                legacy_status = candidate
                status_meaning = LEGACY_STATUSES[candidate]
            elif candidate in REVIEW_STATUSES:
                status_meaning = candidate
            else:
                allowed_values = sorted(set(LEGACY_STATUSES) | REVIEW_STATUSES)
                raise ReviewValidationError(f"status must be one of: {', '.join(allowed_values)}.")
            if new_status is not None and new_status != status_meaning:
                raise ReviewValidationError("status and review_status conflict.")
            new_status = status_meaning
        decision_present = "investigation_decision" in changes
        decision: str | None = None
        if decision_present:
            candidate = changes["investigation_decision"]
            if candidate is not None and (not isinstance(candidate, str) or candidate not in INVESTIGATION_DECISIONS):
                raise ReviewValidationError(
                    f"investigation_decision must be null or one of: {', '.join(sorted(INVESTIGATION_DECISIONS))}."
                )
            decision = candidate
        note_present = any(key in changes for key in ("note", "notes", "investigation_notes"))
        note_value = next((changes[key] for key in ("note", "investigation_notes", "notes") if key in changes), None)
        note = _optional_text(note_value, "note", maximum=MAX_NOTE_LENGTH)
        if note_present and note is None:
            raise ReviewValidationError("note must contain non-whitespace text.")
        if not status_present and not decision_present and not note_present and "reviewer_id" not in changes:
            raise ReviewValidationError("At least one review field is required.")

        with PROCESS_LOCK:
            cases = self._load_cases()
            case = next((item for item in cases if item["id"] == case_id), None)
            if case is None:
                raise ReviewNotFoundError(f"Case '{case_id}' was not found.")
            events: list[tuple[str, dict[str, Any]]] = []
            changed = False
            status_changed = new_status is not None and new_status != case["review_status"]
            legacy_label_changed = legacy_status is not None and legacy_status != case.get("status")
            if status_changed:
                previous = case["review_status"]
                case["review_status"] = new_status
                events.append(("status_changed", {"from": previous, "to": new_status, "legacy_status": legacy_status}))
                changed = True
            if legacy_label_changed:
                case["status"] = legacy_status
                changed = True
            elif status_changed and legacy_status is None:
                case["status"] = {"OPEN": "Review Pending", "REVIEWING": "Under Investigation", "RESOLVED": "Resolved"}[new_status]
            if decision_present and decision != case.get("investigation_decision"):
                previous = case.get("investigation_decision")
                case["investigation_decision"] = decision
                events.append(("decision_recorded", {"from": previous, "to": decision}))
                changed = True
            if note is not None:
                existing = case.get("investigation_notes", "")
                case["investigation_notes"] = f"{existing}\n{note}".strip()
                case["notes"] = case["investigation_notes"]
                events.append(("note_added", {"note": note}))
                changed = True
            if reviewer_id is not None and reviewer_id != case.get("reviewer_id"):
                previous = case.get("reviewer_id")
                case["reviewer_id"] = reviewer_id
                events.append(("reviewer_identified", {"from": previous, "to": reviewer_id}))
                changed = True
            if changed:
                now = utc_now()
                case["updated_at"] = now
                for event_type, event_changes in events:
                    self._append_event(case, event_type, reviewer_id or case.get("reviewer_id"), event_changes, timestamp=now)
                self._write_cases(cases)
            return self._copy(case)

    def _load_cases(self) -> list[dict[str, Any]]:
        sample = self._read_sample()
        seeded = sample.get("cases")
        if not isinstance(seeded, list):
            raise ReviewStoreError("Sample data does not contain a valid cases list.")
        by_id: dict[str, dict[str, Any]] = {}
        if self.store_path.exists():
            try:
                stored = json.loads(self.store_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ReviewStoreError("Investigation store is unreadable or malformed.") from exc
            if not isinstance(stored, dict) or not isinstance(stored.get("cases"), list):
                raise ReviewStoreError("Investigation store has an invalid structure.")
            stored_ids: set[str] = set()
            for item in stored["cases"]:
                if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                    raise ReviewStoreError("Investigation store contains an invalid case record.")
                if item["id"] in stored_ids:
                    raise ReviewStoreError("Investigation store contains duplicate case identifiers.")
                self._validate_record(item)
                stored_ids.add(item["id"])
                by_id[item["id"]] = item
        changed = False
        for seed in seeded:
            if not isinstance(seed, dict) or not isinstance(seed.get("id"), str):
                raise ReviewStoreError("Sample data contains an invalid case record.")
            if seed["id"] in by_id:
                continue
            by_id[seed["id"]] = self._migrate_seed(seed)
            changed = True
        cases = list(by_id.values())
        if changed or not self.store_path.exists():
            self._write_cases(cases)
        return cases

    @staticmethod
    def _validate_record(case: dict[str, Any]) -> None:
        required = {
            "transaction_id": str,
            "review_status": str,
            "investigation_decision": (str, type(None)),
            "reviewer_id": (str, type(None)),
            "investigation_notes": str,
            "created_at": str,
            "updated_at": str,
            "review_history": list,
        }
        for field, expected_type in required.items():
            if not isinstance(case.get(field), expected_type):
                raise ReviewStoreError(f"Investigation store contains an invalid {field} value.")
        if case["review_status"] not in REVIEW_STATUSES:
            raise ReviewStoreError("Investigation store contains an unsupported review status.")
        if case["investigation_decision"] is not None and case["investigation_decision"] not in INVESTIGATION_DECISIONS:
            raise ReviewStoreError("Investigation store contains an unsupported decision.")
        for event in case["review_history"]:
            if (
                not isinstance(event, dict)
                or not isinstance(event.get("event_type"), str)
                or not isinstance(event.get("timestamp"), str)
                or not isinstance(event.get("changes"), dict)
                or event.get("reviewer_id") is not None and not isinstance(event.get("reviewer_id"), str)
            ):
                raise ReviewStoreError("Investigation store contains an invalid review history event.")

    def _migrate_seed(self, seed: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        legacy_status = seed.get("status", "Review Pending")
        review_status = LEGACY_STATUSES.get(legacy_status, "OPEN")
        notes = seed.get("notes", "")
        case = {
            **seed,
            "review_status": review_status,
            "investigation_decision": None,
            "reviewer_id": None,
            "investigation_notes": notes if isinstance(notes, str) else "",
            "created_at": now,
            "updated_at": now,
            "review_history": [],
        }
        self._append_event(case, "investigation_migrated", None, {"source": "sample_data.json", "status": legacy_status}, timestamp=now)
        return case

    def _read_sample(self) -> dict[str, Any]:
        try:
            data = json.loads(self.sample_data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewStoreError("Dashboard sample data is unreadable or malformed.") from exc
        if not isinstance(data, dict):
            raise ReviewStoreError("Dashboard sample data must be a JSON object.")
        return data

    def _write_cases(self, cases: list[dict[str, Any]]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.store_path.parent, prefix=f".{self.store_path.name}.", suffix=".tmp", delete=False) as file:
                temp_name = file.name
                json.dump({"cases": cases}, file, indent=2, ensure_ascii=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, self.store_path)
            temp_name = None
            try:
                directory_fd = os.open(self.store_path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Atomic replacement still protects readers; directory fsync is platform-dependent.
                pass
        except OSError as exc:
            raise ReviewStoreError("Could not persist investigation changes.") from exc
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except OSError:
                    pass

    @staticmethod
    def _append_event(
        case: dict[str, Any], event_type: str, reviewer_id: str | None,
        changes: dict[str, Any], *, timestamp: str | None = None,
    ) -> None:
        case.setdefault("review_history", []).append({
            "event_type": event_type,
            "timestamp": timestamp or utc_now(),
            "reviewer_id": reviewer_id,
            "changes": changes,
        })

    @staticmethod
    def _copy(case: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(case))
