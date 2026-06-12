# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import os
import sys

from locust import HttpUser, events, task
from locust.contrib.csv_request_logger import CsvRequestLogger

BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "straight-through")
if BENCHMARK_MODE not in ("straight-through", "agentic"):
    print(f"Error: BENCHMARK_MODE must be 'straight-through' or 'agentic', got '{BENCHMARK_MODE}'")
    sys.exit(1)

_request_log_path = os.environ.get("REQUEST_LOG")
if not _request_log_path:
    print("Error: REQUEST_LOG environment variable is required")
    sys.exit(1)
_request_logger = CsvRequestLogger(_request_log_path)


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    _request_logger.register(environment)


class ResponsesAPIUser(HttpUser):
    """Benchmarks /v1/responses with optional tool calls (set BENCHMARK_MODE=agentic)."""

    def on_start(self):
        self.model_id = "openai/mock-model"
        self.headers = {"Content-Type": "application/json"}

    @task
    def responses_api(self):
        payload = {
            "model": self.model_id,
            "input": "Hello, how are you?",
            "store": True,
            "stream": False,
        }

        if BENCHMARK_MODE == "agentic":
            payload["tools"] = [{"type": "web_search"}]

        with self.client.post(
            "/v1/responses",
            data=json.dumps(payload),
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                return
            try:
                data = response.json()
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
                return
            if data.get("status") != "completed":
                response.failure(f"Expected status 'completed', got '{data.get('status')}'")
                return
            output = data.get("output", [])
            if not output:
                response.failure("Empty output array")
                return
            has_assistant = any(
                item.get("role") == "assistant" and item.get("content")
                for item in output
                if item.get("type") == "message"
            )
            if not has_assistant:
                response.failure("No assistant message with content in output")
                return
            if BENCHMARK_MODE == "agentic":
                has_tool_output = any(item.get("type") in ("web_search_call", "function_call") for item in output)
                if not has_tool_output:
                    response.failure("No tool call output in agentic response")
                    return
            response.success()
