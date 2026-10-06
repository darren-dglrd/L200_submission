#!/usr/bin/env python3
"""Lightweight Web Server for the Process Hypothesis Multi-Agent App & A2UI v0.9 Canvas.

Runs out-of-the-box on Python standard library (`http.server`) with zero external dependencies,
while also exposing a clean API contract for:
  - `POST /api/preprocess`          -> Invokes `data_preprocess` skill (obfuscation + batch selection)
  - `POST /api/chat_generate`       -> Invokes `hypothesis_generator_agent` (returns A2UI v0.9 cards)
  - `POST /api/queue/add`           -> Enqueues a candidate hypothesis for 1-by-1 testing
  - `POST /api/queue/evaluate_next` -> Invokes `hypothesis_evaluator_agent` on next queued hypothesis
  - `GET  /api/export_csv`          -> Exports evaluated results as Looker Studio / Data Studio CSV
  - `GET  /api/observability`       -> Returns OpenTelemetry spans, system metrics & quality metrics
"""

import csv
import http.server
import io
import json
from pathlib import Path
import socketserver
from typing import Any, Dict
from urllib.parse import urlparse

from app.agent import ProcessHypothesisMultiAgentApp

STATIC_DIR = Path(__file__).resolve().parent / "static"
GLOBAL_APP = ProcessHypothesisMultiAgentApp()


def build_looker_studio_csv(session_id: str = "default_session") -> str:
    """Export evaluated hypotheses in a flat CSV ready for Google Looker Studio (Data Studio)."""
    session = GLOBAL_APP.session_service.get_or_create_session(session_id=session_id)
    evaluated = session["state"].get("evaluated_results", [])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "hypothesis_id",
        "hypothesis_type",
        "title",
        "parameters",
        "verdict_badge",
        "p_value",
        "fdr_q_value",
        "pearson_r",
        "ml_variance_explained",
        "plain_english_summary",
        "recommended_action",
    ])
    for item in evaluated:
        d = item.get("data", item)
        writer.writerow([
            d.get("hypothesis_id", ""),
            d.get("hypothesis_type", ""),
            d.get("title", ""),
            ";".join(d.get("parameters", [])),
            d.get("verdict_badge", ""),
            d.get("p_value", ""),
            d.get("fdr_q_value", ""),
            d.get("pearson_r", ""),
            d.get("ml_variance_explained", ""),
            d.get("plain_english_summary", ""),
            d.get("recommended_action", ""),
        ])
    return buf.getvalue()


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class HypothesisRequestHandler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html_bytes = (STATIC_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(html_bytes)
            return

        if parsed.path == "/api/observability":
            self._send_json(GLOBAL_APP.observability.get_dashboard_metrics())
            return

        if parsed.path == "/api/export_csv":
            csv_text = build_looker_studio_csv().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="evaluated_hypotheses_looker_studio.csv"'
            )
            self.send_header("Content-Length", str(len(csv_text)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(csv_text)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length > 0 else {}
        session_id = body.get("session_id", "default_session")

        if self.path == "/api/preprocess":
            batches = body.get("batches", ["ALL"])
            custom_cols = body.get("custom_columns")
            res = GLOBAL_APP.preprocess_dataset(
                batches=batches, custom_columns=custom_cols, session_id=session_id
            )
            self._send_json(res)
            return

        if self.path == "/api/chat_generate":
            msg = body.get("message", "")
            res = GLOBAL_APP.run_hypothesis_generator_agent(user_message=msg, session_id=session_id)
            self._send_json(res)
            return

        if self.path == "/api/queue/add":
            h_id = body.get("hypothesis_id", "")
            custom = body.get("custom_hypothesis")
            res = GLOBAL_APP.enqueue_hypothesis(
                hypothesis_id=h_id, custom_hypothesis=custom, session_id=session_id
            )
            self._send_json(res)
            return

        if self.path == "/api/queue/evaluate_next":
            res = GLOBAL_APP.run_hypothesis_evaluator_agent_next(session_id=session_id)
            self._send_json(res)
            return

        self._send_json({"error": f"Unknown endpoint {self.path}"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep console output clean


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    with ThreadingHTTPServer((host, port), HypothesisRequestHandler) as httpd:
        print(f"Process Hypothesis Agent & A2UI v0.9 Canvas listening on http://{host}:{port}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
