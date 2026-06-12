#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Generate randomized experiment matrices for CI regression benchmarking.

Generates a CSV with one row per run, randomized across version groups.
Each version group gets the specified number of replicates.
"""

import argparse
import os
import random
import subprocess
import sys
from typing import Any

import pandas as pd


def _git(args: list[str], git_dir: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the CompletedProcess result."""
    cmd = ["git"]
    if git_dir:
        cmd.extend(["-C", git_dir])
    cmd.extend(args)
    return subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603


def find_latest_release_tag(git_dir: str | None = None) -> str:
    result = _git(["tag", "--sort=-v:refname"], git_dir)
    for line in result.stdout.splitlines():
        tag = line.strip()
        if tag and tag[0] == "v" and tag.count(".") == 2:
            print(f"  Latest release tag: {tag}", flush=True)
            return tag
    print("Error: no release tags found (vX.Y.Z)", file=sys.stderr)
    sys.exit(1)


def resolve_version_hash(label: str, git_dir: str | None = None) -> str:
    result = _git(["rev-parse", label], git_dir)
    if result.returncode == 0:
        return result.stdout.strip()
    result = _git(["rev-parse", f"origin/{label}"], git_dir)
    if result.returncode == 0:
        return result.stdout.strip()
    print(f"Error: could not resolve git ref '{label}'", file=sys.stderr)
    if git_dir:
        print(f"  (in repo: {git_dir})", file=sys.stderr)
    sys.exit(1)


def generate_matrix(versions: list[dict[str, str]], replicates: int, seed: int) -> list[dict[str, Any]]:
    """Generate RCBD experiment design matrix.

    Randomized Complete Block Design: each block contains exactly one
    run of each version in a random order. Blocks are in temporal
    sequence (block 1 runs first). Within-block randomization prevents
    position effects from aliasing with treatment effects.

    Args:
        versions: list of {"label": str, "hash": str}
        replicates: number of blocks (one replicate per version per block)
        seed: random seed for within-block randomization
    """
    rng = random.Random(seed)
    all_rows = []
    for block in range(1, replicates + 1):
        block_rows = []
        for ver in versions:
            block_rows.append(
                {
                    "version_hash": ver["hash"],
                    "version_label": ver["label"],
                    "run": 0,
                    "replicate": block,
                    "block": block,
                    "benchmark_mode": "agentic",
                    "mock_tool_call_count": 1,
                }
            )
        rng.shuffle(block_rows)
        all_rows.extend(block_rows)

    for i, row in enumerate(all_rows):
        row["run"] = i

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment matrices for CI regression benchmarking")
    parser.add_argument(
        "--versions", default=None, help="Comma-separated version labels (omit to use --baseline-ref/--comparison-ref)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--replicates", type=int, default=1, help="Replicates per cell (default: 1)")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: .)")
    parser.add_argument("--repo", default=None, help="Path to OGX git repo for resolving git refs")
    parser.add_argument("--baseline-ref", default=None, help="Baseline git ref (empty or omitted = latest release tag)")
    parser.add_argument("--comparison-ref", default="HEAD", help="Comparison git ref (default: HEAD)")
    args = parser.parse_args()

    hash_map = {}

    if args.replicates < 1:
        print("WARNING: --replicates < 1, using 1", file=sys.stderr)
        args.replicates = 1

    os.makedirs(args.output_dir, exist_ok=True)

    if args.versions:
        labels = [v.strip() for v in args.versions.split(",")]
    else:
        baseline_ref = args.baseline_ref or find_latest_release_tag(args.repo)
        comparison_ref = args.comparison_ref
        labels = ["baseline", "comparison", "comparison_ctrl"]
        baseline_hash = resolve_version_hash(baseline_ref, git_dir=args.repo)
        comparison_hash = resolve_version_hash(comparison_ref, git_dir=args.repo)
        hash_map = {
            "baseline": baseline_hash,
            "comparison": comparison_hash,
            "comparison_ctrl": comparison_hash,
        }

    versions = []
    for label in labels:
        if label in hash_map:
            version_hash = hash_map[label]
        else:
            version_hash = resolve_version_hash(label, git_dir=args.repo)
        versions.append({"label": label, "hash": version_hash})

    all_rows = generate_matrix(versions, args.replicates, args.seed)
    matrix_path = os.path.join(args.output_dir, "experiment-matrix.csv")
    pd.DataFrame(all_rows).to_csv(matrix_path, index=False)
    print(f"  Wrote: {matrix_path} ({len(all_rows)} runs)")


if __name__ == "__main__":
    main()
