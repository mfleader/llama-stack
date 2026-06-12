#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pre-flight checks for the benchmark experiment."""

import csv
import os
import shutil
import socket
import subprocess
from datetime import UTC
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    pass


def log(msg: str) -> None:
    from datetime import datetime

    ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def check_ports(ports: list[int]) -> None:
    log("Checking port availability...")
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) == 0:
                raise PreflightError(f"Port {port} already in use")


def check_disk_space(results_dir: str) -> None:
    usage = shutil.disk_usage(results_dir)
    avail_mb = usage.free // (1024 * 1024)
    if avail_mb < 500:
        raise PreflightError(f"Insufficient disk space: {avail_mb}MB (need 500MB)")
    log(f"  Disk space: {avail_mb}MB available")


def check_cpu_pinning() -> bool:
    log("Checking CPU pinning...")
    try:
        os.sched_setaffinity(0, {0})
        actual = os.sched_getaffinity(0)
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
        if actual == {0}:
            log("  CPU pinning verified (sched_setaffinity works)")
            return True
    except (OSError, AttributeError):
        pass
    log("WARNING: CPU pinning unavailable, running without pinning")
    return False


def check_matrix(matrix_csv: str) -> None:
    log("Checking experiment matrix...")
    if not Path(matrix_csv).exists():
        raise PreflightError(f"Matrix CSV not found: {matrix_csv}")
    with open(matrix_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        n_runs = sum(1 for _ in reader)
    log(f"  Matrix: {n_runs} runs, {n_cols} columns")
    if n_cols != 7:
        raise PreflightError(f"Expected 7 columns, got {n_cols}")


def check_worktrees(worktree_base: str, baseline_label: str, comparison_label: str) -> None:
    log("Checking worktrees...")
    for label in [baseline_label, comparison_label]:
        wt = Path(worktree_base) / label
        if not (wt / ".git").exists():
            raise PreflightError(f"Worktree not found: {wt}. Run experiment/setup-worktree.sh first")
    log(f"  Worktrees OK ({baseline_label}, {comparison_label})")


def run_all_checks(
    results_dir: str,
    matrix_csv: str,
    worktree_base: str,
    baseline_label: str,
    comparison_label: str,
    mock_port: int,
    stack_port: int,
) -> bool:
    """Run all pre-flight checks. Returns taskset_available bool.

    Raises PreflightError on any failure.
    """
    log("=== Pre-Flight Checks ===")

    check_ports([mock_port, stack_port])
    check_disk_space(results_dir)
    taskset_available = check_cpu_pinning()
    check_matrix(matrix_csv)
    check_worktrees(worktree_base, baseline_label, comparison_label)

    log("=== Pre-Flight Complete ===")

    return taskset_available


def _read_proc(path: str, prefix: str) -> str:
    """Read a colon-delimited value from a /proc file by prefix."""
    try:
        for line in open(path):
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def _cmd(cmd: list[str]) -> str:
    """Run a command and return stripped stdout, or 'unknown' on failure."""
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()  # noqa: S603
    except Exception:
        return "unknown"


def record_environment(cfg: dict[str, Any], results_dir: str) -> None:
    """Write hardware and software environment details to environment.txt."""
    from datetime import datetime

    wb = cfg["worktree_base"]
    bl, cl = cfg["baseline_label"], cfg["comparison_label"]
    env = {
        "date": datetime.now(tz=UTC).isoformat(),
        "hostname": socket.gethostname(),
        "kernel": os.uname().release,
        "python": _cmd(["uv", "run", "--project", cfg["repo_root"], "python", "--version"]),
        "baseline_label": bl,
        "baseline_hash": _cmd(["git", "-C", str(Path(wb) / bl), "rev-parse", "HEAD"]),
        "comparison_label": cl,
        "comparison_hash": _cmd(["git", "-C", str(Path(wb) / cl), "rev-parse", "HEAD"]),
        "cpu_model": _read_proc("/proc/cpuinfo", "model name"),
        "cpu_cores": os.cpu_count(),
        "memory_mb": int(_read_proc("/proc/meminfo", "MemTotal").split()[0]) // 1024
        if _read_proc("/proc/meminfo", "MemTotal") != "unknown"
        else "unknown",
        "seed": cfg["seed"],
        "run_duration": cfg["run_duration"],
        "cpu_pinning": f"ogx={cfg['cpu_ogx']} locust={cfg['cpu_locust']} mock={cfg['cpu_mock']}"
        if cfg["taskset_available"]
        else "disabled",
    }
    env_file = Path(results_dir) / "environment.txt"
    log(f"Recording environment to {env_file}")
    env_file.write_text("\n".join(f"{k}: {v}" for k, v in env.items()) + "\n")
