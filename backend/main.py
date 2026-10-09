import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from backend.investigator import (
    AmbiguousTransactionError,
    InferenceUnavailableError,
    InvestigatorReportService,
    InvalidTransactionIdError,
    PredictionMismatchError,
    TransactionNotFoundError,
    TransactionSourceUnavailableError,
)
from backend.reviews import (
    InvestigationStore,
    ReviewNotFoundError,
    ReviewStoreError,
    ReviewValidationError,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_FILE = ROOT_DIR / "backend" / "data" / "sample_data.json"
INVESTIGATIONS_FILE = ROOT_DIR / "backend" / "data" / "investigations.json"
REVIEW_STORE = InvestigationStore(DATA_FILE, INVESTIGATIONS_FILE)
INVESTIGATOR = InvestigatorReportService()
MAX_REQUEST_BYTES = 64 * 1024


def load_data() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def filter_transactions(transactions: list[dict], query: dict) -> list[dict]:
    vendor = query.get("vendor", [""])[0].strip().lower()
    department = query.get("department", [""])[0].strip().lower()
    risk_level = query.get("risk_level", [""])[0].strip().lower()
    status = query.get("status", [""])[0].strip().lower()
    transaction_type = query.get("type", [""])[0].strip().lower()

    filtered = []
    for item in transactions:
        if vendor and vendor not in item["vendor"].lower():
            continue
        if department and department not in item["department"].lower():
            continue
        if risk_level and risk_level != item["level"]:
            continue
        if status and status not in item["status"].lower():
            continue
        if transaction_type and transaction_type not in item["type"].lower():
            continue
        filtered.append(item)
    return filtered


def build_dashboard_payload(data: dict) -> dict:
    highlight_transaction = max(data["transactions"], key=lambda item: item["risk"])
    return {
        "summary": data["summary"],
        "alerts": data["alerts"],
        "risk_trend": data["risk_trend"],
        "workflow": data["workflow"],
        "transactions": data["transactions"],
        "vendors": data["vendors"],
        "duplicates": data["duplicates"],
        "cases": data["cases"],
        "highlight_transaction": highlight_transaction,
    }


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/dashboard":
            try:
                data = load_data()
                data["cases"] = REVIEW_STORE.list_cases()
                return self.send_json(build_dashboard_payload(data))
            except (OSError, json.JSONDecodeError, ReviewStoreError):
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Dashboard data is unavailable.")

        if parsed.path == "/api/transactions":
            data = load_data()
            filtered = filter_transactions(data["transactions"], parse_qs(parsed.query))
            return self.send_json({"transactions": filtered})

        if parsed.path == "/api/health":
            return self.send_json({"status": "ok"})

        if parsed.path == "/api/investigations/report":
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
            if set(query) != {"transaction_id"} or len(query["transaction_id"]) != 1:
                return self.send_api_error(HTTPStatus.BAD_REQUEST, "Provide exactly one transaction_id parameter.")
            transaction_id = query["transaction_id"][0]
            try:
                report = INVESTIGATOR.generate_report(transaction_id)
            except InvalidTransactionIdError as exc:
                return self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
            except TransactionNotFoundError as exc:
                return self.send_api_error(HTTPStatus.NOT_FOUND, str(exc))
            except AmbiguousTransactionError as exc:
                return self.send_api_error(HTTPStatus.CONFLICT, str(exc))
            except (TransactionSourceUnavailableError, InferenceUnavailableError):
                return self.send_api_error(HTTPStatus.SERVICE_UNAVAILABLE, "Investigation report capability is unavailable.")
            except PredictionMismatchError:
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Investigation report could not be generated.")
            except Exception:
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Investigation report could not be generated.")
            return self.send_json({"report": report})

        if parsed.path == "/api/cases":
            try:
                return self.send_json({"cases": REVIEW_STORE.list_cases()})
            except ReviewStoreError:
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Investigation records are unavailable.")

        if parsed.path.startswith("/api/cases/"):
            case_id, is_history = self.case_route(parsed.path)
            if case_id is None:
                return self.send_api_error(HTTPStatus.NOT_FOUND, "Case endpoint not found.")
            try:
                case = REVIEW_STORE.get_case(case_id)
            except ReviewNotFoundError as exc:
                return self.send_api_error(HTTPStatus.NOT_FOUND, str(exc))
            except ReviewStoreError:
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Investigation records are unavailable.")
            if is_history:
                return self.send_json({"case_id": case_id, "review_history": case["review_history"]})
            return self.send_json({"case": case})

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/cases":
            body = self.read_json_body()
            if body is None:
                return self.send_api_error(HTTPStatus.BAD_REQUEST, "Malformed JSON request body.")
            try:
                case, created = REVIEW_STORE.open_case(body.get("transaction_id"), body.get("reviewer_id"))
            except ReviewValidationError as exc:
                return self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
            except ReviewNotFoundError as exc:
                return self.send_api_error(HTTPStatus.NOT_FOUND, str(exc))
            except ReviewStoreError:
                return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save investigation record.")
            return self.send_json({"case": case}, HTTPStatus.CREATED if created else HTTPStatus.OK)

        if not parsed.path.startswith("/api/cases/"):
            return self.send_api_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")

        case_id, is_history = self.case_route(parsed.path)
        if case_id is None or is_history:
            return self.send_api_error(HTTPStatus.NOT_FOUND, "Case endpoint not found.")
        body = self.read_json_body()
        if body is None:
            return self.send_api_error(HTTPStatus.BAD_REQUEST, "Malformed JSON request body.")
        try:
            case = REVIEW_STORE.update_case(case_id, body)
        except ReviewValidationError as exc:
            return self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
        except ReviewNotFoundError as exc:
            return self.send_api_error(HTTPStatus.NOT_FOUND, str(exc))
        except ReviewStoreError:
            return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save investigation changes.")
        return self.send_json({"case": case})

    def do_PATCH(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/cases/"):
            return self.send_api_error(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        case_id, is_history = self.case_route(parsed.path)
        if case_id is None or is_history:
            return self.send_api_error(HTTPStatus.NOT_FOUND, "Case endpoint not found.")
        body = self.read_json_body()
        if body is None:
            return self.send_api_error(HTTPStatus.BAD_REQUEST, "Malformed JSON request body.")
        try:
            case = REVIEW_STORE.update_case(case_id, body)
        except ReviewValidationError as exc:
            return self.send_api_error(HTTPStatus.BAD_REQUEST, str(exc))
        except ReviewNotFoundError as exc:
            return self.send_api_error(HTTPStatus.NOT_FOUND, str(exc))
        except ReviewStoreError:
            return self.send_api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not save investigation changes.")
        return self.send_json({"case": case})

    @staticmethod
    def case_route(path: str) -> tuple[str | None, bool]:
        parts = path.removeprefix("/api/cases/").split("/")
        if len(parts) == 1 and parts[0]:
            return parts[0], False
        if len(parts) == 2 and parts[0] and parts[1] == "history":
            return parts[0], True
        return None, False

    def read_json_body(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            return None
        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_api_error(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)


def run() -> None:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), FrontendHandler)
    print(f"ProcureMind app available at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
