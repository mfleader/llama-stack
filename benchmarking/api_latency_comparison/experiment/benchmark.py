#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Benchmark experiment orchestrator.

Iterates a design matrix, starts servers per run, runs Locust,
collects latency data. Each run is fully independent.

Usage:
  uv run python experiment/benchmark.py --baseline-ref v1.0.2 --comparison-ref HEAD
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from mirakuru import HTTPExecutor

from benchmarking.api_latency_comparison.analysis.fit_resp_latency_model import load_and_fit
from benchmarking.api_latency_comparison.experiment.generate_design_matrix import (
    find_latest_release_tag,
    generate_matrix,
    resolve_version_hash,
)
from benchmarking.api_latency_comparison.experiment.preflight import (
    PreflightError,
    log,
    record_environment,
    run_all_checks,
)
from benchmarking.api_latency_comparison.experiment.runlog import RunLogEntry, append_run_log, load_completed_runs

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = (PROJECT_DIR / ".." / "..").resolve()
BASELINE_LABEL = "baseline"
COMPARISON_LABEL = "comparison"


class PinnedHTTPExecutor(HTTPExecutor):
    def __init__(self, *args: Any, cpu_affinity: set[int] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cpu_affinity = cpu_affinity

    @property
    def _popen_kwargs(self) -> dict[str, Any]:
        kwargs = super()._popen_kwargs
        if self._cpu_affinity is not None:
            affinity = self._cpu_affinity

            def pinned_preexec():
                os.setsid()
                os.sched_setaffinity(0, affinity)

            kwargs["preexec_fn"] = pinned_preexec
        return kwargs


def _get_worktree_dir(cfg: dict[str, Any], version_label: str) -> Path:
    if version_label == BASELINE_LABEL:
        return Path(cfg["worktree_base"]) / BASELINE_LABEL
    return Path(cfg["worktree_base"]) / COMPARISON_LABEL


def _clear_ogx_state() -> None:
    """Remove SQLite DBs and file cache from /tmp/ogx-benchmark."""
    for f in ["/tmp/ogx-benchmark/kvstore.db", "/tmp/ogx-benchmark/sql_store.db"]:  # noqa: S108
        try:
            os.unlink(f)
        except FileNotFoundError:
            pass
    shutil.rmtree("/tmp/ogx-benchmark/files", ignore_errors=True)  # noqa: S108


def _verify_cpu_pinning(pid: int, expected_core: int, name: str) -> None:
    try:
        actual = os.sched_getaffinity(pid)
        if actual == {expected_core}:
            log(f"  {name} (PID {pid}) pinned to core {expected_core}")
        else:
            log(f"  WARNING: {name} (PID {pid}) expected core {{{expected_core}}}, got {actual}")
    except OSError:
        pass


@contextmanager
def _servers(cfg: dict[str, Any], version: str, ogx_log_path: Path) -> Generator[None, None, None]:
    """Context manager that starts mock + OGX servers for one run."""
    with _mock_server_executor(cfg, version) as mock_exec:
        if cfg["taskset_available"] and mock_exec.process is not None:
            _verify_cpu_pinning(mock_exec.process.pid, cfg["cpu_mock"], "mock")
        with _ogx_server_executor(cfg, version, ogx_log_path) as ogx_exec:  # noqa: SIM117
            if cfg["taskset_available"] and ogx_exec.process is not None:
                _verify_cpu_pinning(ogx_exec.process.pid, cfg["cpu_ogx"], "OGX")
            _reset_mock(cfg)
            yield


def _make_executor(cmd: list[str], kwargs: dict[str, Any], cfg: dict[str, Any], cpu_key: str) -> HTTPExecutor:
    """Create an HTTPExecutor, optionally with CPU pinning."""
    if cfg["taskset_available"]:
        return PinnedHTTPExecutor(cmd, cpu_affinity={cfg[cpu_key]}, **kwargs)
    return HTTPExecutor(cmd, **kwargs)


def _mock_server_executor(cfg: dict[str, Any], version_label: str) -> HTTPExecutor:
    wt_dir = _get_worktree_dir(cfg, version_label)
    cmd = [sys.executable, str(SCRIPT_DIR / "mock_server.py"), str(cfg["mock_port"])]
    kwargs = dict(
        url=f"http://localhost:{cfg['mock_port']}/v1/health",
        method="GET",
        timeout=10,
        cwd=str(wt_dir),
        expected_returncode=-signal.SIGTERM,
    )
    return _make_executor(cmd, kwargs, cfg, "cpu_mock")


@contextmanager
def _ogx_server_executor(
    cfg: dict[str, Any], version_label: str, log_path: Path
) -> Generator[HTTPExecutor, None, None]:
    wt_dir = _get_worktree_dir(cfg, version_label)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "ogx.core.server.server:create_app",
        "--port",
        str(cfg["stack_port"]),
        "--factory",
    ]
    with open(log_path, "a") as log_file:
        kwargs = dict(
            url=f"http://localhost:{cfg['stack_port']}/v1/health",
            method="GET",
            timeout=30,
            cwd=str(wt_dir),
            envvars={
                "OPENAI_BASE_URL": f"http://localhost:{cfg['mock_port']}/v1",
                "OPENAI_API_KEY": "fake-token",
                "OGX_CONFIG": str(PROJECT_DIR / "configs" / "stack-config-benchmark.yaml"),
            },
            stdout=log_file,
            stderr=subprocess.STDOUT,
            expected_returncode=-signal.SIGTERM,
        )
        with _make_executor(cmd, kwargs, cfg, "cpu_ogx") as executor:
            yield executor


def _reset_mock(cfg: dict[str, Any]) -> None:
    """POST /reset to the mock server to zero its counters."""
    req = urllib.request.Request(f"http://localhost:{cfg['mock_port']}/reset", method="POST")
    try:
        urllib.request.urlopen(req, timeout=2)  # noqa: S310
    except Exception:
        time.sleep(0.5)
        urllib.request.urlopen(req, timeout=2)  # noqa: S310


def _run_locust(cfg: dict[str, Any], row: dict[str, Any], results_dir: Path) -> tuple[int, Path]:
    """Run a single Locust session, return (exit_code, request_csv_path)."""
    run_id = row["run"]
    rd = results_dir / "runs" / str(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    request_csv = rd / "requests.csv"
    locust_log = rd / "locust.log"

    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(SCRIPT_DIR / "locustfile_responses.py"),
        "--headless",
        "-u",
        "1",
        "-r",
        "1",
        "--run-time",
        f"{cfg['run_duration']}s",
        "--host",
        f"http://localhost:{cfg['stack_port']}",
    ]

    env = os.environ.copy()
    env["BENCHMARK_MODE"] = row.get("benchmark_mode", "agentic")
    env["REQUEST_LOG"] = str(request_csv)

    run_kwargs = {}
    if cfg["taskset_available"]:
        affinity = {cfg["cpu_locust"]}

        def pinned_preexec():
            os.sched_setaffinity(0, affinity)

        run_kwargs["preexec_fn"] = pinned_preexec

    with open(locust_log, "w") as lf:
        result = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, **run_kwargs)  # noqa: S603

    return result.returncode, request_csv


def _validate_agentic_loop(cfg: dict[str, Any]) -> None:
    """Verify the agentic loop fires (web_search_call in response)."""
    log("Validating agentic loop...")
    _reset_mock(cfg)

    payload = json.dumps(
        {
            "model": "openai/mock-model",
            "input": "agentic loop check",
            "tools": [{"type": "web_search"}],
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{cfg['stack_port']}/v1/responses",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)  # noqa: S310
    data = json.loads(resp.read())
    types = [item.get("type") for item in data.get("output", [])]
    if "web_search_call" not in types:
        raise RuntimeError(f"Agentic loop not firing: output types = {types}")

    stats = json.loads(urllib.request.urlopen(f"http://localhost:{cfg['mock_port']}/stats", timeout=2).read())
    search_count = stats.get("get_search_count", 0)
    if search_count >= 1:
        log(f"  Agentic loop validated: search_count={search_count}")
    else:
        log(f"WARNING: Mock search endpoint not called (search_count={search_count})")


def run_preflight_benchmark(cfg: dict[str, Any], results_dir: Path) -> None:
    """Start servers, run a short benchmark, validate agentic loop."""
    log("=== Pre-Flight Benchmark ===")

    baseline = BASELINE_LABEL
    _clear_ogx_state()

    preflight_dir = results_dir / "runs" / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    with _servers(cfg, baseline, preflight_dir / "ogx.log"):
        _validate_agentic_loop(cfg)
        log("  Pre-flight benchmark passed")


def _record_run(
    results_dir: Path,
    row: dict[str, Any],
    run_id: int,
    version: str,
    start: int,
    end: int,
    n_requests: int,
    status: str,
) -> None:
    entry = RunLogEntry(
        run=run_id,
        version_label=version,
        version_hash=row["version_hash"],
        replicate=row["replicate"],
        benchmark_mode=row.get("benchmark_mode", "agentic"),
        mock_tool_call_count=row.get("mock_tool_call_count", 1),
        start_time=start,
        end_time=end,
        duration_s=end - start,
        requests_completed=n_requests,
        status=status,
    )
    append_run_log(results_dir / "run-log.csv", entry)


def run_experiment(cfg: dict[str, Any], results_dir: Path | str, matrix_csv: Path) -> None:
    """Main experiment loop."""
    results_dir = Path(results_dir)
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)
    (results_dir / "analysis" / "fits").mkdir(parents=True, exist_ok=True)
    taskset = run_all_checks(
        str(results_dir),
        str(matrix_csv),
        cfg["worktree_base"],
        BASELINE_LABEL,
        COMPARISON_LABEL,
        cfg["mock_port"],
        cfg["stack_port"],
    )
    cfg["taskset_available"] = taskset

    cfg["repo_root"] = str(REPO_ROOT)
    cfg["baseline_label"] = BASELINE_LABEL
    cfg["comparison_label"] = COMPARISON_LABEL
    record_environment(cfg, str(results_dir))

    run_preflight_benchmark(cfg, results_dir)

    matrix = pd.read_csv(matrix_csv).to_dict("records")
    run_log = results_dir / "run-log.csv"
    completed = load_completed_runs(run_log) if run_log.exists() else set()
    total_runs = len(matrix)
    failed_runs = 0
    start_time = time.time()

    log(f"Matrix: {total_runs} runs ({len(completed)} already completed)")

    interrupted = False

    def handle_signal(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        log("\n=== INTERRUPTED ===")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for i, row in enumerate(matrix, 1):
        if interrupted:
            break

        run_id = row["run"]
        if run_id in completed:
            continue

        version = row["version_label"]
        log(f"--- Run {i}/{total_runs}: run={run_id} version={version} rep={row['replicate']} ---")

        _clear_ogx_state()
        run_start = int(time.time())

        try:
            rd = results_dir / "runs" / str(run_id)
            rd.mkdir(parents=True, exist_ok=True)
            with _servers(cfg, version, rd / "ogx.log"):
                exit_code, request_csv = _run_locust(cfg, row, results_dir)
                if request_csv.exists():
                    with open(request_csv) as f:
                        n_requests = sum(1 for _ in f) - 1
                else:
                    n_requests = 0

            run_end = int(time.time())

            if n_requests == 0:
                status = "locust_crash" if exit_code != 0 else "error"
                log(f"  WARNING: {status} (exit={exit_code}, 0 requests)")
                failed_runs += 1
            else:
                status = "ok"
                if exit_code != 0:
                    log(f"  WARNING: Locust exited {exit_code} but produced {n_requests} requests")

            _record_run(results_dir, row, run_id, version, run_start, run_end, n_requests, status)
            log(f"  Result: {n_requests} reqs, status={status}")

        except Exception as e:
            run_end = int(time.time())
            log(f"  ERROR: {e}\n{traceback.format_exc()}")
            failed_runs += 1
            _record_run(results_dir, row, run_id, version, run_start, run_end, 0, "error")

    total_duration = int(time.time() - start_time)
    total_ok = len(load_completed_runs(run_log) if run_log.exists() else set())

    log("")
    log("=== Benchmark Complete ===")
    log(f"  Total elapsed: {total_duration // 60}m {total_duration % 60}s")
    log(f"  Runs: {total_ok} ok, {failed_runs} error")

    if interrupted:
        log("\nExperiment was interrupted. Resume with the same command.")
        raise KeyboardInterrupt


def setup_and_generate(results_dir: Path, baseline_ref: str | None, comparison_ref: str, replicates: int) -> Path:
    """Resolve refs, setup worktrees, generate design matrix. Returns matrix CSV path."""
    baseline_ref = baseline_ref or find_latest_release_tag()

    log("Setting up worktrees...")
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "bash",
            str(SCRIPT_DIR / "setup-worktree.sh"),
            "--repo",
            str(REPO_ROOT),
            "--baseline",
            baseline_ref,
            "--baseline-label",
            BASELINE_LABEL,
            "--comparison",
            comparison_ref,
            "--comparison-label",
            COMPARISON_LABEL,
        ],
        check=True,
    )

    log("Generating design matrix...")
    baseline_hash = resolve_version_hash(baseline_ref)
    comparison_hash = resolve_version_hash(comparison_ref)
    versions = [
        {"label": BASELINE_LABEL, "hash": baseline_hash},
        {"label": COMPARISON_LABEL, "hash": comparison_hash},
        {"label": f"{COMPARISON_LABEL}_ctrl", "hash": comparison_hash},
    ]
    rows = generate_matrix(versions, replicates, seed=42)
    matrix_path = results_dir / "experiment-matrix.csv"
    pd.DataFrame(rows).to_csv(matrix_path, index=False)
    log(f"  Wrote: {matrix_path} ({len(rows)} runs)")

    return matrix_path


def fit_model(results_dir: Path, baseline_label: str) -> dict[str, Any]:
    log("Fitting model...")
    return load_and_fit([str(results_dir)], baseline_label)


def cleanup_worktrees(cfg: dict[str, Any]) -> None:
    for label in [BASELINE_LABEL, COMPARISON_LABEL]:
        shutil.rmtree(f"{cfg['worktree_base']}/{label}", ignore_errors=True)
    subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "prune"], capture_output=True)  # noqa: S603, S607


def run_benchmark(
    cfg: dict[str, Any], results_dir: Path | str, baseline_ref: str | None, comparison_ref: str, replicates: int
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    matrix_csv = setup_and_generate(results_dir, baseline_ref, comparison_ref, replicates)
    run_experiment(cfg, results_dir, matrix_csv)
    return fit_model(results_dir, BASELINE_LABEL)


def _default_config() -> dict[str, Any]:
    return {
        "worktree_base": os.environ.get("WORKTREE_BASE", "/tmp/ci-validation"),  # noqa: S108
        "run_duration": int(os.environ.get("RUN_DURATION", "10")),
        "mock_port": int(os.environ.get("MOCK_PORT", "8080")),
        "stack_port": int(os.environ.get("STACK_PORT", "8321")),
        "seed": os.environ.get("SEED", "42"),
        "cpu_ogx": int(os.environ.get("CPU_OGX", "0")),
        "cpu_locust": int(os.environ.get("CPU_LOCUST", "1")),
        "cpu_mock": int(os.environ.get("CPU_MOCK", "2")),
        "taskset_available": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark experiment orchestrator")
    parser.add_argument("--baseline-ref", default=None, help="Baseline git ref (default: latest release tag)")
    parser.add_argument("--comparison-ref", default="HEAD", help="Comparison git ref (default: HEAD)")
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per group (default: 3)")
    parser.add_argument("--run-duration", type=int, default=10, help="Seconds per run (default: 10)")
    parser.add_argument("--results-dir", help="Results directory (default: auto-timestamped)")
    args = parser.parse_args()

    cfg = _default_config()
    cfg["run_duration"] = args.run_duration

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        results_dir = REPO_ROOT / "results" / f"api-latency-{ts}"

    log("=== Benchmark Experiment ===")
    log(f"  Results: {results_dir}")
    log(f"  Duration: {cfg['run_duration']}s per run")

    fp_results = {}
    try:
        fp_results = run_benchmark(cfg, results_dir, args.baseline_ref, args.comparison_ref, args.replicates)
    except (PreflightError, ValueError) as e:
        log(f"ERROR: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        cleanup_worktrees(cfg)

    if fp_results.get("false_positive_detected"):
        sys.exit(1)


if __name__ == "__main__":
    main()
