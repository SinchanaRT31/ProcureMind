"""Trusted transaction lookup and deterministic investigation reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
TRANSACTION_DATASET = ROOT_DIR / "dataset" / "ProcureMind" / "procuremind_procurement_dataset.csv"
RAW_TRANSACTION_COLUMNS = (
    "transaction_id",
    "vendor_id",
    "vendor_rating",
    "department_id",
    "item_category",
    "quantity",
    "unit_price",
    "total_amount",
    "order_date",
    "invoice_date",
    "due_date",
    "payment_status",
    "purchase_type",
    "vendor_location",
)
TRANSACTION_ID_PATTERN = re.compile(r"TXN-[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
REPORT_SCHEMA_VERSION = "1.0"
CSV_CHUNK_SIZE = 25_000
INDEX_SCHEMA_VERSION = "2"
INDEX_BUILD_LOCK = threading.Lock()
NUMERIC_RAW_COLUMNS = frozenset({"vendor_rating", "quantity", "unit_price", "total_amount"})


class InvalidTransactionIdError(ValueError):
    pass


class TransactionNotFoundError(LookupError):
    pass


class AmbiguousTransactionError(LookupError):
    pass


class TransactionSourceUnavailableError(RuntimeError):
    pass


class InferenceUnavailableError(RuntimeError):
    pass


class PredictionMismatchError(RuntimeError):
    pass


def _source_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _update_digest_value(digest: Any, value: Any) -> None:
    """Add one typed, length-delimited SQLite value to an index digest."""
    if value is None:
        digest.update(b"\x00")
    elif isinstance(value, int):
        digest.update(b"\x01" + struct.pack(">q", value))
    elif isinstance(value, float):
        digest.update(b"\x02" + struct.pack(">d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"\x03" + struct.pack(">Q", len(encoded)) + encoded)
    elif isinstance(value, bytes):
        digest.update(b"\x04" + struct.pack(">Q", len(value)) + value)
    else:
        raise TypeError(f"Unsupported SQLite value type: {type(value).__name__}")


def _index_content_digest(connection: sqlite3.Connection) -> str:
    """Hash all stored transaction fields in deterministic source-row order."""
    digest = hashlib.sha256(b"ProcureMind SQLite transaction index content v1\0")
    columns = ("_row_number", *RAW_TRANSACTION_COLUMNS)
    for column in columns:
        encoded = column.encode("utf-8")
        digest.update(b"\x05" + struct.pack(">Q", len(encoded)) + encoded)

    quoted_columns = ", ".join(f'"{column}"' for column in columns)
    cursor = connection.execute(
        f"SELECT {quoted_columns} FROM transactions ORDER BY _row_number"
    )
    while rows := cursor.fetchmany(10_000):
        for row in rows:
            for value in row:
                _update_digest_value(digest, value)
    return digest.hexdigest()


def _decode_json(value: Any) -> Any | None:
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


class InvestigatorReportService:
    """Generate reports from exact-ID rows in the fixed trusted dataset."""

    def __init__(self, dataset_path: str | Path = TRANSACTION_DATASET):
        # dataset_path is an internal constructor seam for tests, never request input.
        self._dataset_path = Path(dataset_path)
        self._index_path = self._dataset_path.parent / ".procuremind-cache" / f"{self._dataset_path.name}.sqlite3"
        self._validated_signature: tuple[int, int, int, int] | None = None
        self._validated_index_signature: tuple[int, int, int, int] | None = None

    @staticmethod
    def validate_transaction_id(transaction_id: Any) -> str:
        if not isinstance(transaction_id, str) or not TRANSACTION_ID_PATTERN.fullmatch(transaction_id):
            raise InvalidTransactionIdError("transaction_id must be a valid TXN identifier.")
        return transaction_id

    def lookup_transaction(self, transaction_id: str) -> pd.DataFrame:
        transaction_id = self.validate_transaction_id(transaction_id)
        try:
            self._ensure_index()
            rows = self._query_index(transaction_id)
        except (OSError, sqlite3.Error, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise TransactionSourceUnavailableError("Trusted transaction data is unavailable.") from exc
        if not rows:
            raise TransactionNotFoundError("No exact transaction ID match was found.")
        if len(rows) > 1:
            raise AmbiguousTransactionError("The transaction ID has multiple exact matches.")
        result = pd.DataFrame(rows, columns=RAW_TRANSACTION_COLUMNS)
        result["transaction_id"] = pd.array(result["transaction_id"], dtype="string")
        for column in RAW_TRANSACTION_COLUMNS:
            if column != "transaction_id":
                result[column] = result[column].where(result[column].notna(), float("nan"))
        return result.loc[:, RAW_TRANSACTION_COLUMNS]

    def _ensure_index(self, *, force_rebuild: bool = False) -> None:
        signature = _source_signature(self._dataset_path)
        index_signature = self._current_index_signature()
        if (
            not force_rebuild
            and signature == self._validated_signature
            and index_signature == self._validated_index_signature
        ):
            return

        # One process-wide lock prevents duplicate builds and keeps publication
        # coordinated across service instances in this server process.
        with INDEX_BUILD_LOCK:
            signature = _source_signature(self._dataset_path)
            index_signature = self._current_index_signature()
            if (
                not force_rebuild
                and signature == self._validated_signature
                and index_signature == self._validated_index_signature
            ):
                return
            source_hash = _sha256_file(self._dataset_path)
            if _source_signature(self._dataset_path) != signature:
                raise OSError("Trusted transaction data changed while checking the index.")
            if self._index_matches_source(source_hash):
                self._validated_signature = signature
                self._validated_index_signature = self._current_index_signature()
                return
            self._build_index(source_hash, signature)
            self._validated_signature = signature
            self._validated_index_signature = self._current_index_signature()

    def _current_index_signature(self) -> tuple[int, int, int, int] | None:
        try:
            return _source_signature(self._index_path)
        except OSError:
            return None

    def _index_matches_source(self, source_hash: str) -> bool:
        if not self._index_path.is_file():
            return False
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{self._index_path}?mode=ro", uri=True)
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if integrity != ("ok",):
                return False
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("schema_version") != INDEX_SCHEMA_VERSION or metadata.get("source_sha256") != source_hash:
                return False
            stored_schema = [
                (row[1], row[2].upper(), row[3], row[5])
                for row in connection.execute("PRAGMA table_info(transactions)")
            ]
            expected_schema = [("_row_number", "INTEGER", 1, 0)] + [
                (column, "REAL" if column in NUMERIC_RAW_COLUMNS else "TEXT", 0, 0)
                for column in RAW_TRANSACTION_COLUMNS
            ]
            if stored_schema != expected_schema:
                return False
            row_count = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            duplicate_ids = connection.execute(
                "SELECT COUNT(*) FROM (SELECT transaction_id FROM transactions "
                "WHERE transaction_id IS NOT NULL GROUP BY transaction_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            duplicate_rows = connection.execute(
                "SELECT COALESCE(SUM(id_count - 1), 0) FROM "
                "(SELECT COUNT(*) AS id_count FROM transactions WHERE transaction_id IS NOT NULL "
                "GROUP BY transaction_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            content_digest = _index_content_digest(connection)
            return (
                row_count == int(metadata["row_count"])
                and duplicate_ids == int(metadata["duplicate_id_count"])
                and duplicate_rows == int(metadata["duplicate_row_count"])
                and content_digest == metadata.get("content_sha256")
            )
        except (OSError, sqlite3.Error, KeyError, ValueError, TypeError):
            return False
        finally:
            if connection is not None:
                connection.close()

    def _build_index(self, source_hash: str, starting_signature: tuple[int, int, int, int]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._index_path.name}.", suffix=".tmp", dir=self._index_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        connection: sqlite3.Connection | None = None
        try:
            column_definitions = [
                f'"{column}" {"REAL" if column in NUMERIC_RAW_COLUMNS else "TEXT"}'
                for column in RAW_TRANSACTION_COLUMNS
            ]
            insert_columns = ["_row_number", *RAW_TRANSACTION_COLUMNS]
            placeholders = ", ".join("?" for _ in insert_columns)
            quoted_columns = ", ".join(f'"{column}"' for column in insert_columns)
            connection = sqlite3.connect(temporary_path)
            with connection:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                connection.execute(
                    "CREATE TABLE transactions ("
                    "_row_number INTEGER NOT NULL, " + ", ".join(column_definitions) + ")"
                )
                connection.execute(
                    "CREATE INDEX transactions_by_id ON transactions (transaction_id COLLATE BINARY)"
                )
                row_count = 0
                for chunk in pd.read_csv(
                    self._dataset_path,
                    usecols=RAW_TRANSACTION_COLUMNS,
                    dtype={"transaction_id": "string"},
                    chunksize=CSV_CHUNK_SIZE,
                ):
                    values = []
                    for record in chunk.loc[:, RAW_TRANSACTION_COLUMNS].itertuples(index=False, name=None):
                        row_count += 1
                        values.append((row_count, *(_sqlite_value(value) for value in record)))
                    connection.executemany(
                        f"INSERT INTO transactions ({quoted_columns}) VALUES ({placeholders})", values
                    )
                indexed_rows = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                duplicate_ids = connection.execute(
                    "SELECT COUNT(*) FROM (SELECT transaction_id FROM transactions "
                    "WHERE transaction_id IS NOT NULL GROUP BY transaction_id HAVING COUNT(*) > 1)"
                ).fetchone()[0]
                duplicate_rows = connection.execute(
                    "SELECT COALESCE(SUM(id_count - 1), 0) FROM "
                    "(SELECT COUNT(*) AS id_count FROM transactions WHERE transaction_id IS NOT NULL "
                    "GROUP BY transaction_id HAVING COUNT(*) > 1)"
                ).fetchone()[0]
                if indexed_rows != row_count:
                    raise sqlite3.DatabaseError("Transaction index row count validation failed.")
                content_digest = _index_content_digest(connection)
                connection.executemany(
                    "INSERT INTO metadata (key, value) VALUES (?, ?)",
                    [
                        ("schema_version", INDEX_SCHEMA_VERSION),
                        ("source_sha256", source_hash),
                        ("row_count", str(row_count)),
                        ("duplicate_id_count", str(duplicate_ids)),
                        ("duplicate_row_count", str(duplicate_rows)),
                        ("content_sha256", content_digest),
                    ],
                )
                connection.commit()
                if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
                    raise sqlite3.DatabaseError("Transaction index integrity validation failed.")
            connection.close()
            connection = None

            # Dataset deployments must be immutable while a server process is
            # running. Detect ordinary concurrent changes during build and refuse
            # to publish an index from a moving source.
            if _source_signature(self._dataset_path) != starting_signature or _sha256_file(self._dataset_path) != source_hash:
                raise OSError("Trusted transaction data changed during index construction.")
            os.replace(temporary_path, self._index_path)
        finally:
            if connection is not None:
                connection.close()
            temporary_path.unlink(missing_ok=True)

    def _query_index(self, transaction_id: str) -> list[tuple[Any, ...]]:
        select_columns = ", ".join(f'"{column}"' for column in RAW_TRANSACTION_COLUMNS)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{self._index_path}?mode=ro", uri=True)
            return connection.execute(
                f"SELECT {select_columns} FROM transactions "
                "WHERE transaction_id = ? COLLATE BINARY ORDER BY _row_number LIMIT 2",
                (transaction_id,),
            ).fetchall()
        except sqlite3.Error:
            # A corrupt or externally removed index is rebuilt from the trusted
            # source. If rebuilding fails, do not query the previous index.
            if connection is not None:
                connection.close()
                connection = None
            self._ensure_index(force_rebuild=True)
            connection = sqlite3.connect(f"file:{self._index_path}?mode=ro", uri=True)
            return connection.execute(
                f"SELECT {select_columns} FROM transactions "
                "WHERE transaction_id = ? COLLATE BINARY ORDER BY _row_number LIMIT 2",
                (transaction_id,),
            ).fetchall()
        finally:
            if connection is not None:
                connection.close()

    def generate_report(self, transaction_id: str) -> dict[str, Any]:
        transaction_id = self.validate_transaction_id(transaction_id)
        raw_transaction = self.lookup_transaction(transaction_id)
        try:
            # Import only for a report request; preserve the existing public predictor.
            from ml.predict import predict_procurement_data

            prediction = predict_procurement_data(
                raw_transaction,
                include_risk=True,
                include_explanations=True,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
            raise InferenceUnavailableError("Required prediction or explanation artifacts are unavailable.") from exc

        if len(prediction) != 1 or "transaction_id" not in prediction.columns:
            raise PredictionMismatchError("Prediction output did not identify exactly one transaction.")
        prediction_id = prediction.iloc[0]["transaction_id"]
        if not isinstance(prediction_id, str) or prediction_id != transaction_id:
            raise PredictionMismatchError("Prediction transaction ID does not match the requested ID.")

        row = prediction.iloc[0]
        predicted_anomaly = self._integer(row.get("predicted_anomaly"))
        score = self._number(row.get("ml_anomaly_score"))
        if predicted_anomaly is None or score is None:
            model_finding: dict[str, Any] = {"status": "unavailable", "source": "Isolation Forest"}
        else:
            model_finding = {
                "status": "flagged" if predicted_anomaly == 1 else "not_flagged",
                "source": "Isolation Forest",
                "predicted_anomaly": bool(predicted_anomaly),
                "anomaly_score": score,
                "score_interpretation": "Higher values indicate a stronger model anomaly signal; this is not a probability.",
            }

        shap_contributions = self._shap_contributions(row)
        if shap_contributions is None:
            shap_result: dict[str, Any] = {"status": "unavailable", "source": "SHAP"}
        else:
            shap_result = {
                "status": "available",
                "source": "SHAP",
                "reference": "Saved training-only background",
                "contributions": shap_contributions,
                "interpretation": "Positive contributions raise the model anomaly score relative to the background; negative contributions lower it.",
            }

        risk_evidence = _decode_json(row.get("evidence"))
        recommendations = _decode_json(row.get("recommended_actions"))
        verification = {
            name: _decode_json(row.get(name))
            for name in ("duplicate_candidates", "consistency_checks", "verification_signals")
        }
        verification_available = all(value is not None for value in verification.values())
        selected_evidence = risk_evidence if isinstance(risk_evidence, list) else None
        evidence_summary = row.get("evidence_summary")
        evidence_summary = evidence_summary if isinstance(evidence_summary, str) else None
        priority = row.get("investigation_priority")
        priority = priority if isinstance(priority, str) else None

        why_flagged = {
            "model_prediction": model_finding,
            "model_explanation": shap_result,
        }
        supporting_findings = [{
            "source": "risk_engine.selected_evidence",
            "status": "available" if selected_evidence is not None else "unavailable",
            "findings": selected_evidence or [],
            "evidence_summary": evidence_summary,
            "investigation_priority": priority,
        }]
        if not verification_available:
            inconclusive: list[dict[str, Any]] = [{
                "source": "verification_engine",
                "status": "unavailable",
            }]
        else:
            inconclusive = [{
                "source": "verification_engine",
                "status": "findings_require_human_verification" if any(verification.values()) else "no_findings_returned",
                **verification,
            }]
        if recommendations is None or not isinstance(recommendations, list):
            follow_up_checks: dict[str, Any] = {"status": "unavailable", "source": "risk_engine.recommended_actions"}
        else:
            follow_up_checks = {
                "status": "available",
                "source": "risk_engine.recommended_actions",
                "checks": recommendations,
            }

        return {
            "transaction_id": transaction_id,
            "schema_version": REPORT_SCHEMA_VERSION,
            "why_flagged": why_flagged,
            "supporting_findings": supporting_findings,
            "inconclusive_or_conflicting": inconclusive,
            "missing_information": [
                "Supporting purchase-order and invoice documents are not supplied to this report service.",
                "Approval records and business justification are not supplied to this report service.",
            ],
            "follow_up_checks": follow_up_checks,
            "limitations": [
                "This report summarizes structured transaction data and existing model outputs; findings require human verification.",
                "Anomaly scores are not probabilities, and an anomaly signal does not establish fraud, intent, or wrongdoing.",
                "Report generation does not create or alter a human investigation decision or review history.",
            ],
        }

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            number = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number in (0, 1) else None

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if pd.notna(number) else None

    @classmethod
    def _shap_contributions(cls, row: pd.Series) -> list[dict[str, Any]] | None:
        contributions = []
        for rank in range(1, 4):
            feature = row.get(f"top_feature_{rank}")
            value = cls._number(row.get(f"top_contribution_{rank}"))
            if not isinstance(feature, str) or not feature or value is None:
                continue
            contributions.append({
                "feature": feature,
                "contribution": value,
                "direction": "raises_anomaly_score" if value > 0 else "lowers_anomaly_score" if value < 0 else "no_change",
            })
        return contributions or None
