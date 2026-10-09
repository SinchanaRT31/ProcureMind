import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from ml.preprocess import clean_data
from ml.verification import DUPLICATE_SIGNATURE, ProcurementVerificationEngine, temporal_vendor_cadence_experiment
from test_preprocess import raw_frame



def duplicated_frame() -> pd.DataFrame:
    data = raw_frame().copy()
    data.loc[1, ["vendor_id", "item_category", "quantity", "unit_price", "total_amount", "invoice_date"]] = ["v1", "c1", 2, 10, 20, "2025-01-02"]
    return data


def _legacy_duplicate_candidates(engine: ProcurementVerificationEngine, incoming: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Preserve the pre-optimization algorithm as a fixture oracle."""
    reference = engine.reference_.copy()
    reference["_incoming"] = False
    current = incoming[["transaction_id", "invoice_date", *DUPLICATE_SIGNATURE]].copy()
    current["_incoming"] = True
    combined = pd.concat([reference, current], ignore_index=True).drop_duplicates("transaction_id", keep="last")
    duplicated = combined[combined.duplicated(DUPLICATE_SIGNATURE, keep=False)].sort_values(
        [*DUPLICATE_SIGNATURE, "invoice_date", "transaction_id"]
    )
    candidates: dict[str, list[dict[str, Any]]] = {}
    for _, group in duplicated.groupby(DUPLICATE_SIGNATURE, dropna=False, sort=False):
        rows = group.reset_index(drop=True)
        for position in range(1, len(rows)):
            earlier, later = rows.iloc[position - 1], rows.iloc[position]
            if pd.isna(earlier.invoice_date) or pd.isna(later.invoice_date):
                continue
            days = abs((later.invoice_date - earlier.invoice_date).days)
            if days > engine.duplicate_window_days or not (earlier._incoming or later._incoming):
                continue
            for source, match in ((earlier, later), (later, earlier)):
                if not source._incoming:
                    continue
                candidates.setdefault(str(source.transaction_id), []).append({
                    "candidate_type": "duplicate_invoice",
                    "matched_transaction_id": str(match.transaction_id),
                    "match_reason": "Potential duplicate: exact vendor, category, quantity, unit price, and amount; invoice dates are close.",
                    "match_strength": "exact_metadata_same_date" if days == 0 else "exact_metadata_close_date",
                    "days_between_invoices": int(days),
                    "fields_compared": DUPLICATE_SIGNATURE,
                })
    return candidates


def _duplicate_row(transaction_id: str, *, invoice_date: str | None = "2025-01-02", vendor_id: str | None = "v1", category: str | None = "c1") -> pd.DataFrame:
    row = raw_frame().iloc[[0]].copy()
    row.loc[:, "transaction_id"] = transaction_id
    row.loc[:, "vendor_id"] = vendor_id
    row.loc[:, "item_category"] = category
    row.loc[:, "quantity"] = 2
    row.loc[:, "unit_price"] = 10
    row.loc[:, "total_amount"] = 20
    row.loc[:, "invoice_date"] = invoice_date
    return row


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


def test_duplicate_candidate_optimization_matches_original_for_exact_and_unrelated_groups():
    reference = pd.concat([
        _duplicate_row("ref-1"),
        _duplicate_row("unrelated-1", vendor_id="other"),
        _duplicate_row("unrelated-2", vendor_id="other"),
    ], ignore_index=True)
    incoming = _duplicate_row("incoming-1")
    engine = ProcurementVerificationEngine().fit(reference)

    actual = engine._duplicate_candidates(clean_data(incoming))

    assert actual == _legacy_duplicate_candidates(engine, clean_data(incoming))
    assert [item["matched_transaction_id"] for item in actual["incoming-1"]] == ["ref-1"]


def test_duplicate_candidate_optimization_preserves_null_signature_grouping():
    reference = pd.concat([
        _duplicate_row("null-1", vendor_id=None, category=None),
        _duplicate_row("unrelated"),
    ], ignore_index=True)
    incoming = _duplicate_row("null-incoming", vendor_id=None, category=None)
    for frame in (reference, incoming):
        for column in ("quantity", "unit_price", "total_amount"):
            frame[column] = pd.Series(float("nan"), index=frame.index, dtype="float64")
    engine = ProcurementVerificationEngine().fit(reference)

    actual = engine._duplicate_candidates(clean_data(incoming))

    assert actual == _legacy_duplicate_candidates(engine, clean_data(incoming))
    assert actual["null-incoming"][0]["matched_transaction_id"] == "null-1"


def test_duplicate_candidate_optimization_preserves_multiple_rows_id_replacement_and_adjacent_pairs():
    reference = pd.concat([
        _duplicate_row("a"), _duplicate_row("b"), _duplicate_row("c"),
        _duplicate_row("unrelated-1", vendor_id="other"),
        _duplicate_row("unrelated-2", vendor_id="other"),
    ], ignore_index=True)
    incoming = pd.concat([
        _duplicate_row("b"),  # Superseded by the later row with this same ID.
        _duplicate_row("b", vendor_id="replacement"),  # Replaces reference transaction b.
        _duplicate_row("d"),
        _duplicate_row("e"),
        _duplicate_row("f", vendor_id="other"),
    ], ignore_index=True)
    engine = ProcurementVerificationEngine().fit(reference)
    cleaned = clean_data(incoming)

    actual = engine._duplicate_candidates(cleaned)

    assert actual == _legacy_duplicate_candidates(engine, cleaned)
    # After b is replaced, the sorted signature group is a, c, d, e. Only
    # adjacent pairs are evaluated, so d matches c and e, not a.
    assert [item["matched_transaction_id"] for item in actual["d"]] == ["c", "e"]
    assert [item["matched_transaction_id"] for item in actual["e"]] == ["d"]
    assert [item["matched_transaction_id"] for item in actual["f"]] == ["unrelated-1"]
    assert "b" not in actual


def test_duplicate_candidate_optimization_preserves_dates_ties_order_and_missing_dates():
    reference = pd.concat([
        _duplicate_row("outside", invoice_date="2025-01-06"),
        _duplicate_row("within", invoice_date="2025-01-07"),
        _duplicate_row("same-day-before", invoice_date="2025-01-10"),
        _duplicate_row("same-day-after", invoice_date="2025-01-10"),
        _duplicate_row("missing-date", invoice_date=None),
    ], ignore_index=True)
    incoming = _duplicate_row("middle", invoice_date="2025-01-10")
    engine = ProcurementVerificationEngine().fit(reference)
    cleaned = clean_data(incoming)

    actual = engine._duplicate_candidates(cleaned)

    assert actual == _legacy_duplicate_candidates(engine, cleaned)
    assert [item["matched_transaction_id"] for item in actual["middle"]] == ["within", "same-day-after"]
    assert all(item["days_between_invoices"] <= engine.duplicate_window_days for item in actual["middle"])
    assert all(item["matched_transaction_id"] != "missing-date" for item in actual["middle"])


def test_duplicate_assessment_does_not_mutate_reference_and_is_concurrency_safe():
    reference = pd.concat([_duplicate_row("ref-1"), _duplicate_row("ref-2")], ignore_index=True)
    incoming = _duplicate_row("incoming")
    engine = ProcurementVerificationEngine().fit(reference)
    before = engine.reference_.copy(deep=True)
    serial = engine.assess(incoming)

    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(engine.assess, [incoming] * 8))

    pd.testing.assert_frame_equal(engine.reference_, before)
    for result in concurrent:
        pd.testing.assert_frame_equal(result, serial)


def test_cadence_experiment_is_chronological_and_does_not_use_labels():
    data = raw_frame().copy()
    data.loc[:, "order_date"] = ["2024-01-01", "2024-02-01", "2025-01-01", "2025-02-01"]
    report = temporal_vendor_cadence_experiment(data, cutoff="2025-01-01")
    assert report["historical_rows"] == 2
    assert report["later_rows"] == 2
