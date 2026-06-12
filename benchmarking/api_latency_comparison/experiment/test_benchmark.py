# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""End-to-end benchmark self-test.

Runs the full pipeline (setup → generate → experiment → fit) with
2 replicates and 3s runs, then asserts artifacts were produced.

Run with:
  uv run python -m pytest benchmarking/api_latency_comparison/experiment/test_benchmark.py -v -s
"""

import json
import tempfile
from pathlib import Path

import pytest

from benchmarking.api_latency_comparison.experiment.benchmark import _default_config, cleanup_worktrees, run_benchmark
from benchmarking.api_latency_comparison.experiment.generate_design_matrix import generate_matrix


@pytest.fixture(scope="class")
def benchmark_results():
    results_dir = Path(tempfile.mkdtemp(prefix="benchmark-test-"))
    cfg = _default_config()
    cfg["run_duration"] = 3

    fp_results = run_benchmark(cfg, results_dir, baseline_ref="", comparison_ref="HEAD", replicates=2)

    yield results_dir, fp_results

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


class TestBenchmarkPipeline:
    def test_fp_results_returned(self, benchmark_results):
        _, fp_results = benchmark_results
        assert "false_positive_detected" in fp_results

    def test_decisions_csv_exists(self, benchmark_results):
        results_dir, _ = benchmark_results
        assert (results_dir / "analysis" / "fits" / "decisions.csv").exists()

    def test_fp_results_json_exists(self, benchmark_results):
        results_dir, _ = benchmark_results
        fp_path = results_dir / "analysis" / "fits" / "fp-results.json"
        assert fp_path.exists()
        fp = json.loads(fp_path.read_text())
        assert "false_positive_detected" in fp

    def test_idata_exists(self, benchmark_results):
        results_dir, _ = benchmark_results
        assert (results_dir / "analysis" / "fits" / "idata.nc").exists()

    def test_run_log_exists(self, benchmark_results):
        results_dir, _ = benchmark_results
        assert (results_dir / "run-log.csv").exists()

    def test_experiment_matrix_exists(self, benchmark_results):
        results_dir, _ = benchmark_results
        assert (results_dir / "experiment-matrix.csv").exists()
