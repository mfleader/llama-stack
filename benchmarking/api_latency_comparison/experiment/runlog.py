# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Run log and design matrix I/O."""

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class RunLogEntry:
    run: int
    version_label: str
    version_hash: str
    replicate: int
    benchmark_mode: str
    mock_tool_call_count: int
    start_time: int
    end_time: int
    duration_s: int
    requests_completed: int
    status: str


RUN_LOG_FIELDS = [f.name for f in dataclasses.fields(RunLogEntry)]


def load_completed_runs(run_log: Path | str) -> set[int]:
    """Return the set of run IDs with status 'ok' from the run log CSV."""
    df = pd.read_csv(run_log)
    return set(df.loc[df["status"] == "ok", "run"].astype(int))


def append_run_log(run_log: Path | str, entry: RunLogEntry) -> None:
    """Append a run log entry to the CSV, writing the header if needed."""
    path = Path(run_log)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(dataclasses.asdict(entry))
