import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_FILE = ROOT_DIR / "backend" / "data" / "sample_data.json"


def load_data() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_data(data: dict) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


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
            return self.send_json(build_dashboard_payload(load_data()))

        if parsed.path == "/api/transactions":
            data = load_data()
            filtered = filter_transactions(data["transactions"], parse_qs(parsed.query))
            return self.send_json({"transactions": filtered})

        if parsed.path == "/api/health":
            return self.send_json({"status": "ok"})

        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        else:
            self.path = parsed.path
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/cases/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
            return

        case_id = parsed.path.removeprefix("/api/cases/")
        body = self.read_json_body()
        if body is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON body")
            return

        data = load_data()
        for case in data["cases"]:
            if case["id"] != case_id:
                continue
            case["status"] = body.get("status", case["status"])
            case["notes"] = body.get("notes", case["notes"])
            save_data(data)
            return self.send_json({"case": case})

        self.send_error(HTTPStatus.NOT_FOUND, "Case not found")

    def read_json_body(self) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None

        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run() -> None:
    host = "127.0.0.1"
    port = 8000
    server = ThreadingHTTPServer((host, port), FrontendHandler)
    print(f"ProcureMind app available at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
