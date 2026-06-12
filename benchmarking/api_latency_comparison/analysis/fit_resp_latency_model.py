#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Fit Wald (Inverse Gaussian) latency model with GP temporal adjustment.

HSGP(m=20, c=1.5) Gaussian process absorbs within-run autocorrelation.
Matern32 kernel. Per-group ZeroSumNormal run intercepts. Centered
run-level lambda. mu_version parameterization with beta_v derived as contrast.

Usage:
  python fit_resp_latency_model.py \
    --data-dirs results/my-experiment \
    --baseline baseline
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import nutpie
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from benchmarking.api_latency_comparison.analysis import wald_numba  # noqa: F401 — registers numba dispatch for Wald (Inverse Gaussian) RV
from benchmarking.api_latency_comparison.analysis.decisions import compute_decisions
from benchmarking.api_latency_comparison.analysis.diagnostics import run_all_diagnostics


def load_data(data_dirs: list[str], baseline: str) -> dict[str, Any]:
    """Load experiment data from one or more result directories.

    Returns a dict with arrays and metadata needed by build_model and
    compute_decisions.
    """
    all_dfs = []
    all_meta = []

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        meta = pd.read_csv(data_dir / "experiment-matrix.csv")
        meta["group"] = meta["version_label"]
        all_meta.append(meta)

        for _, row in meta.iterrows():
            f = data_dir / "runs" / str(row["run"]) / "requests.csv"
            if not f.exists():
                f = data_dir / "requests" / f"run-{row['run']}.csv"
            df = pd.read_csv(
                f, usecols=["timestamp", "response_time_ms", "status_code", "exception"], dtype={"exception": str}
            )
            df = df[
                (df["response_time_ms"] > 0)
                & (df["status_code"] == 200)
                & (df["exception"].str.strip().eq("") | df["exception"].isna())
            ]
            df["run"] = row["run"]
            df["group"] = row["group"]
            all_dfs.append(df)

    meta_combined = pd.concat(all_meta, ignore_index=True)
    requests_all = pd.concat(all_dfs, ignore_index=True)
    requests_all = requests_all.sort_values(["run", "timestamp"]).reset_index(drop=True)
    requests_all["time_idx"] = requests_all.groupby("run").cumcount() + 1
    groups = sorted(requests_all["group"].unique())
    n_runs = int(requests_all["run"].nunique())

    if baseline not in groups:
        print(f"  ERROR: baseline '{baseline}' not found in groups {groups}", flush=True)
        sys.exit(1)

    run_to_version = {row["run"]: row["group"] for _, row in meta_combined.iterrows()}

    return {
        "requests": requests_all,
        "meta_combined": meta_combined,
        "groups": groups,
        "n_runs": n_runs,
        "run_to_version": run_to_version,
        "baseline": baseline,
    }


def _filter_observations(requests: pd.DataFrame) -> pd.DataFrame:
    """Drop first and last observation per run (see MODEL.md Data Filtering)."""
    max_idx = requests.groupby("run")["time_idx"].transform("max")
    filtered = requests[(requests["time_idx"] > 1) & (requests["time_idx"] < max_idx)].copy()
    filtered["time_idx"] = filtered.groupby("run").cumcount() + 1
    n_dropped = len(requests) - len(filtered)
    print(f"  Dropped {n_dropped} observations (first + last per run)", flush=True)
    return filtered


def _prepare_arrays(
    requests: pd.DataFrame, groups: list[str], n_runs: int, run_to_version: dict[int, str], baseline: str
) -> dict[str, Any]:
    """Convert filtered DataFrame to arrays for model building."""
    y = requests["response_time_ms"].values.astype(np.float64)
    group_labels = requests["group"].values
    time_idx = requests["time_idx"].values
    run = requests["run"].values.astype(int)
    group_to_idx = {g: i for i, g in enumerate(groups)}
    group_idx = np.array([group_to_idx[g] for g in group_labels])

    group_run_indices = {}
    for g in groups:
        group_run_indices[g] = sorted([t for t, v in run_to_version.items() if v == g])

    run_counts = requests.groupby("run").size()
    print(
        f"  Observations per run: min={run_counts.min()}, median={int(run_counts.median())}, mean={run_counts.mean():.0f}, max={run_counts.max()}",
        flush=True,
    )
    for g in groups:
        mask = group_labels == g
        n_t = requests.loc[mask, "run"].nunique()
        print(f"    {g}: {n_t} runs, {mask.sum()} obs, median={np.median(y[mask]):.1f}ms", flush=True)

    return {
        "y": y,
        "group_labels": group_labels,
        "time_idx": time_idx,
        "run": run,
        "group_idx": group_idx,
        "groups": groups,
        "n_runs": n_runs,
        "run_to_version": run_to_version,
        "group_run_indices": group_run_indices,
        "baseline": baseline,
    }


def build_model(raw_data: dict[str, Any]) -> tuple[pm.Model, dict[str, Any]]:
    """Build the PyMC model. See MODEL.md for the full specification.

    Filters observations (drops first/last per run) and builds the
    model. Section comments match MODEL.md headings. The code defines
    priors before assembling mu and the likelihood (bottom-up), while
    MODEL.md presents likelihood first (top-down). Search by heading.
    """
    # Data Filtering (see MODEL.md)
    filtered = _filter_observations(raw_data["requests"])
    data = _prepare_arrays(
        filtered,
        raw_data["groups"],
        raw_data["n_runs"],
        raw_data["run_to_version"],
        raw_data["baseline"],
    )

    groups = data["groups"]
    n_runs = data["n_runs"]
    y = data["y"]
    group_run_indices = data["group_run_indices"]

    # scale run order to [0, 1] for the drift covariate (global chronological position; RCBD blocking is in the design, not the model)
    run_order_scaled = data["run"].astype(np.float64) / max(n_runs - 1, 1)

    # coords label posterior output axes (e.g. mu_version[baseline] instead of mu_version[0])
    coords = {"group": groups, "obs_id": np.arange(len(y)), "run": np.arange(n_runs)}
    for g in groups:
        coords[f"run_{g}"] = np.arange(len(group_run_indices[g]))

    with pm.Model(coords=coords) as model:
        x_time = pm.Data("x_time", data["time_idx"].astype(np.float64)[:, None])
        x_drift = pm.Data("x_drift", run_order_scaled, dims="obs_id")
        x_run = pm.Data("x_run", data["run"], dims="obs_id")
        x_group = pm.Data("x_group", data["group_idx"], dims="obs_id")

        # Mean Structure
        mu_version = pm.Normal("mu_version", mu=25, sigma=10, dims="group")
        beta_drift = pm.Normal("beta_drift", mu=0, sigma=2)

        # Run Intercepts
        sigma_run = pm.Exponential("sigma_run", lam=1)
        delta_blocks = []
        for _gidx, g in enumerate(groups):
            delta_g = pm.ZeroSumNormal(f"delta_{g}", sigma=sigma_run, dims=f"run_{g}")
            delta_blocks.append(delta_g)
        delta_run_full = pt.zeros(n_runs)
        for gidx, g in enumerate(groups):
            for pos, t in enumerate(group_run_indices[g]):
                delta_run_full = pt.set_subtensor(delta_run_full[t], delta_blocks[gidx][pos])

        # Shape Structure
        log_lambda_bar = pm.Normal("log_lambda_bar", mu=7, sigma=2)
        sigma_lambda = pm.Exponential("sigma_lambda", lam=1)
        log_lambda_run = pm.Normal("log_lambda_run", mu=log_lambda_bar, sigma=sigma_lambda, dims="run")
        lambda_obs = pt.exp(log_lambda_run[x_run])

        # GP Temporal Adjustment
        eta_gp = pm.HalfNormal("eta_gp", sigma=0.25)
        ell_gp = pm.InverseGamma("ell_gp", mu=6, sigma=3)
        cov_func = eta_gp**2 * pm.gp.cov.Matern32(input_dim=1, ls=ell_gp)
        gp = pm.gp.HSGP(m=[20], c=1.5, cov_func=cov_func, parametrization="noncentered", drop_first=True)
        f = gp.prior("f", X=x_time)

        # Likelihood
        mu = mu_version[x_group] + beta_drift * x_drift + delta_run_full[x_run] + f
        mu = pt.maximum(mu, 1e-6)  # Wald (Inverse Gaussian) requires mu > 0; clamp if GP/intercepts go negative
        pm.Wald("y", mu=mu, lam=lambda_obs, observed=y, dims="obs_id")

        # Derived Quantities
        baseline_idx = groups.index(data["baseline"])
        pm.Deterministic("beta_v", mu_version - mu_version[baseline_idx])

    return model, data


def fit_and_diagnose(model: pm.Model, data: dict[str, Any], out_dir: Path) -> az.InferenceData:
    """Sample, compute LOO, run diagnostics. Returns idata."""
    ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
    print(f"[{ts}] --- Fitting (4 chains, 1000+1000, nutpie) ---", flush=True)

    compiled = nutpie.compile_pymc_model(model)
    t0 = time.time()
    idata = nutpie.sample(
        compiled,
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.95,
        seed=42,
        progress_bar=True,
    )
    fit_time = time.time() - t0
    ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
    print(f"[{ts}]   Fit time: {fit_time:.1f}s", flush=True)

    summary = az.summary(
        idata,
        var_names=[
            "mu_version",
            "beta_v",
            "beta_drift",
            "sigma_run",
            "log_lambda_bar",
            "sigma_lambda",
            "eta_gp",
            "ell_gp",
        ],
        kind="all",
        ci_kind="hdi",
        ci_prob=0.89,
    )
    print(summary.to_string(), flush=True)

    with model:
        pm.compute_log_likelihood(idata)
    loo_result = az.loo(idata, pointwise=True)

    run_all_diagnostics(
        idata,
        data["y"],
        data["run"],
        data["group_idx"],
        data["groups"],
        data["n_runs"],
        data["run_to_version"],
        summary=summary,
        loo_result=loo_result,
    )

    idata.to_netcdf(str(out_dir / "idata.nc"))
    return idata


def load_and_fit(data_dirs: list[str], baseline: str) -> dict[str, Any]:
    """Full pipeline: load, build, fit, diagnose, decide, save."""
    raw_data = load_data(data_dirs, baseline)
    model, data = build_model(raw_data)
    out_dir = Path(data_dirs[0]) / "analysis" / "fits"
    out_dir.mkdir(parents=True, exist_ok=True)
    idata = fit_and_diagnose(model, data, out_dir)
    decisions, fp_results = compute_decisions(model, idata, data)

    pd.DataFrame(decisions).to_csv(out_dir / "decisions.csv", index=False)
    with open(out_dir / "fp-results.json", "w") as f:
        json.dump(fp_results, f, indent=2)
    print(f"  Saved: {out_dir}", flush=True)

    return fp_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Wald (Inverse Gaussian) latency model with GP temporal adjustment")
    parser.add_argument("--data-dirs", required=True, help="Comma-separated experiment results directories")
    parser.add_argument("--baseline", required=True, help="Baseline group label for contrasts (e.g., v1.0.2)")
    args = parser.parse_args()
    dirs = [d.strip() for d in args.data_dirs.split(",")]
    fp_results = load_and_fit(dirs, args.baseline)
    if fp_results.get("false_positive_detected"):
        sys.exit(1)
