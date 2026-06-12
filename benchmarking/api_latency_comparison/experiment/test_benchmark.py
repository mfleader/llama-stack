# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end benchmark self-test.

Runs the experiment pipeline (setup → generate → experiment) with
2 replicates and 3s runs, then asserts artifacts were produced.

Run with:
  uv run python -m pytest benchmarking/api_latency_comparison/experiment/test_benchmark.py -v -s
"""

import csv
import socket
import tempfile
from pathlib import Path

import pytest

from benchmarking.api_latency_comparison.experiment.benchmark import _default_config, cleanup_worktrees, run_benchmark
from benchmarking.api_latency_comparison.experiment.generate_design_matrix import generate_matrix, resolve_version_hash
from benchmarking.api_latency_comparison.experiment.preflight import PreflightError, check_ports


@pytest.fixture(scope="class")
def benchmark_results():
    results_dir = Path(tempfile.mkdtemp(prefix="benchmark-test-"))
    cfg = _default_config()
    cfg["run_duration"] = 3

    run_benchmark(cfg, results_dir, baseline_ref="", comparison_ref="HEAD", replicates=2)

    yield results_dir

    cleanup_worktrees(cfg)


class TestRCBDInvariant:
    VERSIONS = [
        {"label": "baseline", "hash": "aaa"},
        {"label": "comparison", "hash": "bbb"},
        {"label": "comparison_ctrl", "hash": "bbb"},
    ]

    def test_each_block_has_all_versions(self):
        rows = generate_matrix(self.VERSIONS, replicates=5, seed=42)
        labels = [v["label"] for v in self.VERSIONS]
        for block in range(1, 6):
            block_labels = sorted(r["version_label"] for r in rows if r["block"] == block)
            assert block_labels == sorted(labels), f"Block {block}: {block_labels}"


class TestErrorPaths:
    def test_bad_ref_raises_valueerror(self):
        with pytest.raises(ValueError, match="could not resolve git ref"):
            resolve_version_hash("nonexistent-ref-abc123")

    def test_port_conflict_raises_preflight_error(self):
        s = socket.socket()
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        s.listen(1)
        try:
            with pytest.raises(PreflightError, match=f"Port {port} already in use"):
                check_ports([port])
        finally:
            s.close()


class TestExperimentPipeline:
    EXPECTED_COLUMNS = {
        "version_hash",
        "version_label",
        "run",
        "replicate",
        "block",
        "benchmark_mode",
        "mock_tool_call_count",
    }

    def test_experiment_matrix_has_correct_schema(self, benchmark_results):
        results_dir = benchmark_results
        with open(results_dir / "experiment-matrix.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 6
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == self.EXPECTED_COLUMNS
        versions = {r["version_label"] for r in rows}
        assert versions == {"baseline", "comparison", "comparison_ctrl"}

    def test_run_log_records_all_runs(self, benchmark_results):
        results_dir = benchmark_results
        with open(results_dir / "run-log.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 6
        assert all(r["status"] == "ok" for r in rows)
        assert all(int(r["requests_completed"]) > 0 for r in rows)

    def test_each_run_has_request_data(self, benchmark_results):
        results_dir = benchmark_results
        run_dirs = [d for d in (results_dir / "runs").iterdir() if d.name != "preflight" and d.is_dir()]
        assert len(run_dirs) == 6
        for rd in run_dirs:
            req_csv = rd / "requests.csv"
            assert req_csv.exists(), f"Missing {req_csv}"
            with open(req_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 3, f"Expected >= 3 requests in {req_csv}, got {len(rows)}"
            assert reader.fieldnames is not None
            assert "response_time_ms" in reader.fieldnames

    def test_environment_recorded(self, benchmark_results):
        results_dir = benchmark_results
        env_file = results_dir / "environment.txt"
        assert env_file.exists()
        content = env_file.read_text()
        assert "hostname:" in content
        assert "cpu_pinning:" in content
        assert "seed:" in content
