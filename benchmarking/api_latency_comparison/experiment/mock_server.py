# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MOCK_TOOL_CALL_COUNT = int(os.environ.get("MOCK_TOOL_CALL_COUNT", "1"))

MOCK_CONTENT = "This is a mock response from the benchmark server."

MOCK_RESPONSE = {
    "id": "chatcmpl-mock123",
    "object": "chat.completion",
    "created": 0,
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": MOCK_CONTENT,
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    },
}
MOCK_RESPONSE_BYTES = json.dumps(MOCK_RESPONSE).encode()


MOCK_SEARCH_RESPONSE = {
    "mixed": {"main": [{"type": "web", "index": 0}]},
    "web": {
        "results": [
            {
                "type": "web",
                "title": "Mock search result",
                "url": "https://example.com/mock-benchmark",
                "description": "Mock search result for benchmarking.",
                "date": "2026-01-01",
                "extra_snippets": [],
            }
        ]
    },
}
MOCK_SEARCH_RESPONSE_BYTES = json.dumps(MOCK_SEARCH_RESPONSE).encode()


class MockHandler(BaseHTTPRequestHandler):
    _counter_lock = threading.Lock()
    call_counter = 0
    search_count = 0

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length else b""
            stream = False
            if body:
                try:
                    stream = json.loads(body).get("stream", False)
                except (json.JSONDecodeError, AttributeError):
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error":"invalid JSON body"}')
                    return

            if stream:
                self._handle_streaming()
            else:
                self._handle_non_streaming()
        elif self.path == "/reset":
            with MockHandler._counter_lock:
                prev_calls = MockHandler.call_counter
                prev_searches = MockHandler.search_count
                MockHandler.call_counter = 0
                MockHandler.search_count = 0
            self._json_response({"reset": True, "prev_post_count": prev_calls, "prev_search_count": prev_searches})
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data: dict[str, Any] | bytes, code: int = 200) -> None:
        body = json.dumps(data).encode() if isinstance(data, dict) else data
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _should_return_tool_call(self) -> bool:
        """Decide whether the next response should include a tool call."""
        if MOCK_TOOL_CALL_COUNT <= 0:
            return False
        with MockHandler._counter_lock:
            MockHandler.call_counter += 1
            if MockHandler.call_counter <= MOCK_TOOL_CALL_COUNT:
                return True
            MockHandler.call_counter = 0
            return False

    def _handle_non_streaming(self) -> None:
        self._json_response(MOCK_RESPONSE_BYTES)

    def _handle_streaming(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        chunk_id = "chatcmpl-mock-stream"
        created = int(time.time())
        model = "mock-model"

        if self._should_return_tool_call():
            self._stream_tool_call(chunk_id, created, model)
        else:
            self._stream_text(chunk_id, created, model)

    def _stream_text(self, chunk_id: str, created: int, model: str) -> None:
        initial = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
        self.wfile.write(f"data: {json.dumps(initial)}\n\n".encode())

        content = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": MOCK_CONTENT}}],
        }
        self.wfile.write(f"data: {json.dumps(content)}\n\n".encode())

        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _stream_tool_call(self, chunk_id: str, created: int, model: str) -> None:
        with MockHandler._counter_lock:
            counter_val = MockHandler.call_counter
        initial = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}],
        }
        self.wfile.write(f"data: {json.dumps(initial)}\n\n".encode())

        tool_call = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call_benchmark_{counter_val}",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "benchmark test"}',
                                },
                            }
                        ]
                    },
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(tool_call)}\n\n".encode())

        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/health":
            self._json_response(b'{"status":"ok"}')
        elif self.path == "/v1/models":
            self._json_response(
                {"object": "list", "data": [{"id": "mock-model", "object": "model", "owned_by": "mock"}]}
            )
        elif self.path.startswith("/res/v1/web/search"):
            with MockHandler._counter_lock:
                MockHandler.search_count += 1
            self._json_response(MOCK_SEARCH_RESPONSE_BYTES)
        elif self.path == "/stats":
            with MockHandler._counter_lock:
                stats = {"post_count": MockHandler.call_counter, "get_search_count": MockHandler.search_count}
            self._json_response(stats)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("127.0.0.1", port), MockHandler)
    print(f"Mock server listening on port {port} (MOCK_TOOL_CALL_COUNT={MOCK_TOOL_CALL_COUNT})")
    server.serve_forever()
