"""Protocol-level coverage through LiteLLM's real Files and Batches clients."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from docetl.execution import BatchRequest, execute_batch


def _output_row(custom_id: str) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"provider-{custom_id}",
            "body": {
                "id": f"completion-{custom_id}",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"answer-{custom_id}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        },
        "error": None,
    }


def test_litellm_files_and_batches_protocol_end_to_end(tmp_path):
    """Exercise HTTP upload, submit, retrieve, download, and ID reordering."""
    requests_seen: list[tuple[str, str]] = []
    output_text = "".join(
        json.dumps(_output_row(custom_id)) + "\n" for custom_id in ("second", "first")
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            return

        def send_json(self, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            requests_seen.append(("POST", self.path))
            if self.path == "/v1/files":
                self.send_json(
                    {
                        "id": "file-input",
                        "object": "file",
                        "bytes": length,
                        "created_at": 0,
                        "filename": "input.jsonl",
                        "purpose": "batch",
                        "status": "processed",
                    }
                )
                return
            if self.path == "/v1/batches":
                self.send_json(
                    {
                        "id": "batch-local",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "input_file_id": "file-input",
                        "completion_window": "24h",
                        "status": "validating",
                        "created_at": 0,
                    }
                )
                return
            self.send_error(404)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            requests_seen.append(("GET", self.path))
            if self.path == "/v1/batches/batch-local":
                self.send_json(
                    {
                        "id": "batch-local",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "input_file_id": "file-input",
                        "completion_window": "24h",
                        "status": "completed",
                        "created_at": 0,
                        "completed_at": 1,
                        "output_file_id": "file-output",
                    }
                )
                return
            if self.path == "/v1/files/file-output/content":
                encoded = output_text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        results = execute_batch(
            [
                BatchRequest("first", {"model": "test-model", "messages": []}),
                BatchRequest("second", {"model": "test-model", "messages": []}),
            ],
            {
                "backend": "litellm",
                "provider": "hosted_vllm",
                "provider_kwargs": {"api_base": base_url, "api_key": "test"},
                "poll_interval_seconds": 0,
                "work_dir": str(tmp_path),
            },
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert [result.custom_id for result in results] == ["first", "second"]
    assert requests_seen == [
        ("POST", "/v1/files"),
        ("POST", "/v1/batches"),
        ("GET", "/v1/batches/batch-local"),
        ("GET", "/v1/files/file-output/content"),
    ]
