# API Latency Comparison Benchmark

Measures per-request latency of two OGX versions under a controlled
agentic workload. Compares an older release against a newer commit by
running both through a mocked agentic workload and recording
per-request response times.

Analysis and model fitting are added in a follow-up PR.

## Overview

The experiment is a randomized complete block design with three trials
(treatment combinations). Each trial is replicated multiple times, and
each replicate is a **run**: one row in the design matrix. The design
matrix generator randomizes run order to guard against temporal
confounding.

The three trials are:

- **Baseline**: the older version (e.g., latest release tag)
- **Comparison**: the newer version under test
- **Comparison control**: same commit as comparison, run independently as
  a negative control for false positive detection

Each run starts a fresh OGX server against a mock backend, sends
agentic requests (with web_search tool calls) via Locust for a fixed
duration, and records per-request latencies. The false positive
detection runs the negative control (same code as comparison, run
independently) to verify the experiment isn't producing spurious
differences.

Components:

- **Mock server** (`experiment/mock_server.py`): canned OpenAI + Brave Search responses
- **Locust** (`experiment/locustfile_responses.py`): load generator, 1 concurrent user
- **Experiment orchestrator** (`experiment/benchmark.py`): run execution with CPU pinning
- **Worktree setup** (`experiment/setup-worktree.sh`): isolated git worktrees per version
- **Design matrix** (`experiment/generate_design_matrix.py`): randomized experiment design

## Prerequisites

```bash
# Benchmark experiment dependencies (Locust, mirakuru)
uv sync --group api-latency-comparison
```

## Quick Start

The orchestrator handles worktree setup, matrix generation, and
experiment execution in one command:

```bash
uv run python -m benchmarking.api_latency_comparison.experiment.benchmark \
  --baseline-ref v1.1.0 --comparison-ref HEAD --replicates 5
```

Output lands in an auto-timestamped directory under `results/`.

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `RESULTS_DIR` | auto-timestamped | Where to write results |
| `MATRIX_CSV` | `$RESULTS_DIR/experiment-matrix.csv` | Experiment matrix |
| `RUN_DURATION` | 10 | Seconds per run |
| `MOCK_PORT` | 8080 | Mock server port |
| `STACK_PORT` | 8321 | OGX server port |
| `CPU_OGX` | 0 | Core for OGX server |
| `CPU_LOCUST` | 1 | Core for Locust |
| `CPU_MOCK` | 2 | Core for mock server |

## Implementation Notes

**CPU pinning**: Processes are pinned via `os.sched_setaffinity()` in
`preexec_fn` callbacks, applied at fork before exec. Pinning is verified
per run via `os.sched_getaffinity(pid)` after each server start.

**Brave Search patching**: Older OGX versions don't have the `base_url`
field on `BraveSearchToolConfig`. The setup script patches it via `sed`
so the mock server can serve search results locally.
