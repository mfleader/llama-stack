# API Latency Comparison Benchmark

Detects latency regressions between two OGX versions using a Bayesian
hierarchical model. Compares an older release against a newer commit by
running both through a mocked agentic workload and fitting a Wald (Inverse Gaussian) latency
model to the per-request response times.

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
  a false positive control

Each run starts a fresh OGX server against a mock backend, sends
agentic requests (with web_search tool calls) via Locust for a fixed
duration, and records per-request latencies. A Bayesian model estimates
the version effect on mean latency. If the false positive control fires
(same code shows a difference), the experiment is unreliable.

Components:

- **Mock server** (`experiment/mock_server.py`): canned OpenAI + Brave Search responses
- **Locust** (`experiment/locustfile_responses.py`): load generator, 1 concurrent user
- **Experiment orchestrator** (`experiment/benchmark.py`): run execution with CPU pinning
- **Worktree setup** (`experiment/setup-worktree.sh`): isolated git worktrees per version
- **Design matrix** (`experiment/generate_design_matrix.py`): randomized experiment design
- **Model fitting** (`analysis/fit_resp_latency_model.py`): Wald (Inverse Gaussian) model + diagnostics

## Prerequisites

```bash
# Benchmark dependencies (PyMC, nutpie, ArviZ, Locust, etc.)
uv sync --group benchmark-regression

# Rust toolchain (required for nutpie compilation)
rustup show  # or: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Quick Start

```bash
# 1. Setup worktrees
bash benchmarking/api_latency_comparison/experiment/setup-worktree.sh \
  --repo /path/to/ogx \
  --baseline v1.0.2 --baseline-label baseline \
  --comparison HEAD --comparison-label comparison

# 2. Generate experiment matrix
uv run python -m benchmarking.api_latency_comparison.experiment.generate_design_matrix \
  --versions "baseline,comparison,comparison_ctrl" \
  --version-hashes "baseline:$(git rev-parse v1.0.2),comparison:$(git rev-parse HEAD),comparison_ctrl:$(git rev-parse HEAD)" \
  --replicates 10 \
  --output-dir /tmp/benchmark-results

# 3. Run experiment
RUN_DURATION=10 \
uv run python -m benchmarking.api_latency_comparison.experiment.benchmark \
  --baseline baseline --comparison comparison \
  --results-dir /tmp/benchmark-results \
  --matrix-csv /tmp/benchmark-results/experiment-matrix.csv

# 4. Fit model and run diagnostics
uv run python -m benchmarking.api_latency_comparison.analysis.fit_resp_latency_model \
  --data-dirs /tmp/benchmark-results \
  --baseline baseline \
  --save-idata
```

## GitHub Actions

The workflow at `.github/workflows/response-latency-regression-benchmark.yml`
runs daily comparing the latest release tag against main. Manual dispatch
accepts custom refs, replicates, and run duration.

```bash
gh workflow run response-latency-regression-benchmark.yml \
  -f replicates=10 \
  -f run_duration=10
```

## Interpreting Results

The fit script prints parameter estimates, diagnostics, and a summary:

```
============================================================
RESULTS (regression threshold: 1.0ms)
============================================================
comparison vs baseline
  mean latency: +1.7ms  [+1.5, +2.0]
  p50:  +1.7ms  [+1.4, +2.1]  REGRESSION
  p95:  +2.0ms  [+1.3, +2.7]  REGRESSION
  p99:  +2.0ms  [+0.7, +3.3]  no regression
============================================================
```

- **mean latency**: posterior mean shift with 99% HDI
- **p50/p95/p99**: posterior predictive quantile contrasts
- **REGRESSION**: P(contrast <= 1ms) < 0.05
- **False positive check**: same-code control must show no difference

Diagnostics include MCMC health (divergences, E-BFMI, ESS), posterior
correlations, prior-to-posterior contraction, Pareto k analysis, and
residual autocorrelation checks.

Output files: `decisions.csv`, `fp-results.json`, and optionally
`idata.nc` (full InferenceData for offline analysis).

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

**Data filtering**: The first and last observation of each run are
dropped before fitting. The first is a client warmup artifact (Locust
connection setup). The last is frequently elevated (edge-of-window effect).

**CPU pinning**: Processes are pinned via `os.sched_setaffinity()` in
`preexec_fn` callbacks, applied at fork before exec. Pinning is verified
per run via `os.sched_getaffinity(pid)` after each server start.

**Brave Search patching**: Older OGX versions don't have the `base_url`
field on `BraveSearchToolConfig`. The setup script patches it via `sed`
so the mock server can serve search results locally.
