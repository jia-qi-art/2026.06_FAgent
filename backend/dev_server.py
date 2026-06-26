from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import data_service as svc


def send_json(handler: BaseHTTPRequestHandler, payload, status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        send_json(self, {})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        dataset = q.get("dataset", "WaDI_A2_ds10")
        event_id = int(q["event_id"]) if q.get("event_id") else None
        try:
            if parsed.path == "/api/health":
                payload = svc.health()
            elif parsed.path == "/api/datasets":
                payload = {"datasets": svc.available_datasets()}
            elif parsed.path == "/api/overview":
                payload = svc.overview(dataset)
            elif parsed.path == "/api/timeseries":
                start = int(q["start"]) if q.get("start") else None
                end = int(q["end"]) if q.get("end") else None
                payload = svc.timeseries(dataset, start, end)
            elif parsed.path == "/api/relation-graph":
                payload = svc.relation_graph(dataset, event_id)
            elif parsed.path == "/api/root-cause":
                payload = svc.root_cause(dataset, event_id)
            elif parsed.path == "/api/report":
                payload = svc.report(dataset, event_id)
            elif parsed.path.startswith("/api/jobs/"):
                payload = svc.get_job(parsed.path.rsplit("/", 1)[-1])
            else:
                send_json(self, {"detail": f"Not found: {parsed.path}"}, 404)
                return
            send_json(self, payload)
        except Exception as exc:
            send_json(self, {"detail": str(exc)}, 500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        payload = json.loads(body or "{}")
        try:
            if parsed.path == "/api/jobs/train":
                job = svc.create_train_job(payload.get("dataset", "WaDI_A2_ds10"), payload)
                send_json(self, {"job_id": job.job_id, "status": job.status, "dataset": job.dataset})
            elif parsed.path == "/api/agent/ask":
                send_json(self, svc.agent_answer(payload.get("dataset", "WaDI_A2_ds10"), payload.get("question", ""), payload.get("event_id")))
            else:
                send_json(self, {"detail": f"Not found: {parsed.path}"}, 404)
        except Exception as exc:
            send_json(self, {"detail": str(exc)}, 500)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Dev API server running at http://127.0.0.1:8000")
    server.serve_forever()
