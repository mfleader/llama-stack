#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Post-fit diagnostics for the latency model.

Organized into three categories (Betancourt/Gelman Bayesian workflow):
  1. Computational faithfulness: did the sampler work?
  2. Calibration: are the priors well-specified?
  3. Retrodictive adequacy: does the model fit the data?
"""

from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all_diagnostics(
    idata: az.InferenceData,
    y: np.ndarray,
    run: np.ndarray,
    group_idx: np.ndarray,
    groups: list[str],
    n_runs: int,
    run_to_version: dict[int, str],
    summary: pd.DataFrame | None = None,
    loo_result: Any = None,
) -> None:
    run_computational(idata, summary)
    run_calibration(idata, groups)
    run_retrodictive(idata, y, run, group_idx, groups, n_runs, run_to_version, loo_result)


# ---------------------------------------------------------------------------
# 1. Computational faithfulness: did the sampler work?
# ---------------------------------------------------------------------------


def run_computational(idata: az.InferenceData, summary: pd.DataFrame | None = None) -> None:
    """MCMC health checks and effective sample size analysis."""
    print("\n  === Computational Diagnostics ===", flush=True)

    print("\n  --- MCMC Health ---", flush=True)
    _, diag = az.diagnose(idata, show_diagnostics=False, return_diagnostics=True)

    print(f"    Divergences: {diag['divergent']['n_divergent']}", flush=True)

    for c, val in enumerate(diag["bfmi"]["bfmi_values"]):
        status = "OK" if val > 0.2 else "LOW"
        print(f"    Chain {c} E-BFMI: {val:.3f} [{status}]", flush=True)

    try:
        ns = idata.sample_stats["n_steps"].values
        for c in range(ns.shape[0]):
            l2 = np.log2(ns[c] + 1)
            print(f"    Chain {c} log2(n_steps): mean={l2.mean():.1f}, max={l2.max():.0f}", flush=True)
    except Exception:
        pass

    print("\n  --- ESS Ratio ---", flush=True)
    if summary is None:
        var_names = ["mu_version", "beta_drift", "sigma_run", "log_lambda_bar", "sigma_lambda", "eta_gp", "ell_gp"]
        present = [v for v in var_names if v in idata.posterior]
        if not present:
            return
        summary = az.summary(idata, var_names=present, kind="diagnostics")

    best_ess = summary["ess_bulk"].max()
    rows = []
    for idx, row in summary.iterrows():
        if "ess_bulk" not in row or pd.isna(row["ess_bulk"]):
            continue
        ratio = row["ess_bulk"] / best_ess
        flag = "<-- LOW" if ratio < 0.1 else ""
        rows.append([str(idx), f"{row['ess_bulk']:.0f}", f"{ratio:.3f}", flag])
    print(tabulate(rows, headers=["Parameter", "ESS", "Ratio", ""], tablefmt="plain"), flush=True)


# ---------------------------------------------------------------------------
# 2. Calibration: are the priors well-specified?
# ---------------------------------------------------------------------------

PRIOR_WIDTHS = {
    "mu_version": 4 * 10,
    "beta_drift": 4 * 2,
    "sigma_run": 3 / 1,
    "log_lambda_bar": 4 * 2,
    "sigma_lambda": 3 / 1,
    "eta_gp": 2.5 * 0.25,
    "ell_gp": 20,
}


def _extract_draws(idata: az.InferenceData, var_names: list[str], groups: list[str]) -> dict[str, np.ndarray]:
    """Extract flattened posterior draws, expanding vectorized params."""
    draws = {}
    for v in var_names:
        try:
            d = idata.posterior[v].values
        except KeyError:
            continue
        if d.ndim == 3:
            for i in range(d.shape[2]):
                label = groups[i] if v == "mu_version" else str(i)
                draws[f"{v}[{label}]"] = d[:, :, i].flatten()
        else:
            draws[v] = d.flatten()
    return draws


def run_calibration(idata: az.InferenceData, groups: list[str]) -> None:
    """Prior-to-posterior contraction and posterior correlations."""
    print("\n  === Calibration Diagnostics ===", flush=True)

    draws = _extract_draws(idata, list(PRIOR_WIDTHS.keys()), groups)

    print("\n  --- Prior-to-Posterior Contraction ---", flush=True)
    rows = []
    for name, flat in draws.items():
        base_var = name.split("[")[0]
        prior_w = PRIOR_WIDTHS.get(base_var)
        if prior_w is None:
            continue
        c = 1 - (np.percentile(flat, 97.5) - np.percentile(flat, 2.5)) / prior_w
        status = "GOOD" if c > 0.5 else "WEAK" if c > 0.1 else "UNINFORMATIVE"
        rows.append([name, f"{c:.3f}", status])
    print(tabulate(rows, headers=["Parameter", "Contraction", "Status"], tablefmt="plain"), flush=True)

    print("\n  --- Posterior Correlations (|r| > 0.3) ---", flush=True)
    if len(draws) < 2:
        print("    None", flush=True)
        return

    corr = pd.DataFrame(draws).corr()
    cols = list(corr.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.3:
                flag = "***" if abs(r) > 0.7 else "**" if abs(r) > 0.5 else "*"
                rows.append([cols[i], cols[j], f"{r:+.3f}", flag])
    if rows:
        print(tabulate(rows, headers=["Parameter A", "Parameter B", "r", ""], tablefmt="plain"), flush=True)
    else:
        print("    None", flush=True)


# ---------------------------------------------------------------------------
# 3. Retrodictive adequacy: does the model fit the data?
# ---------------------------------------------------------------------------


def run_retrodictive(
    idata: az.InferenceData,
    y: np.ndarray,
    run: np.ndarray,
    group_idx: np.ndarray,
    groups: list[str],
    n_runs: int,
    run_to_version: dict[int, str],
    loo_result: Any = None,
) -> None:
    """Pareto k analysis and residual autocorrelation checks."""
    print("\n  === Retrodictive Diagnostics ===", flush=True)
    _report_pareto_k(y, run, loo_result)
    _report_residual_acf(idata, y, run, group_idx, groups, n_runs, run_to_version)


def _report_pareto_k(y: np.ndarray, run: np.ndarray, loo_result: Any) -> None:
    print("\n  --- Pareto k ---", flush=True)

    pk = loo_result.pareto_k.values
    high_k = np.where(pk > 0.7)[0]
    p95 = np.percentile(y, 95)
    p99 = np.percentile(y, 99)

    print(f"    k > 0.7: {len(high_k)}/{len(pk)}", flush=True)
    print(f"    Max k: {pk.max():.3f}", flush=True)
    if len(high_k) > 0:
        above_p95 = sum(y[i] > p95 for i in high_k)
        above_p99 = sum(y[i] > p99 for i in high_k)
        print(f"    Above p95 ({p95:.1f}ms): {above_p95}/{len(high_k)}", flush=True)
        print(f"    Above p99 ({p99:.1f}ms): {above_p99}/{len(high_k)}", flush=True)
        n_runs_affected = len({run[i] for i in high_k})
        print(f"    Spread across {n_runs_affected} runs", flush=True)


def _acf_per_run(resid: np.ndarray, run: np.ndarray, n_runs: int, max_lag: int = 5) -> dict[int, list[float]]:
    """Per-run ACF at lags 1..max_lag. Returns {lag: [values]}."""
    acfs = {lag: [] for lag in range(1, max_lag + 1)}
    for t in range(n_runs):
        mask = run == t
        r = resid[mask]
        if len(r) < 10:
            continue
        r = r - r.mean()
        n = len(r)
        c0 = np.sum(r**2) / n
        if c0 == 0:
            continue
        for lag in range(1, max_lag + 1):
            acfs[lag].append(np.sum(r[:-lag] * r[lag:]) / (n * c0))
    return acfs


def _compute_standardized_residuals(
    idata: az.InferenceData,
    y: np.ndarray,
    run: np.ndarray,
    group_idx: np.ndarray,
    groups: list[str],
    n_runs: int,
    run_to_version: dict[int, str],
) -> np.ndarray:
    """Compute (y - mu_hat) / sqrt(Var_Wald) using posterior means.  Wald = Inverse Gaussian."""
    mu_version_mean = idata.posterior["mu_version"].values.mean(axis=(0, 1))
    beta_drift_mean = idata.posterior["beta_drift"].values.mean()
    run_order_scaled = run.astype(float) / max(n_runs - 1, 1)

    delta_run_mean = np.zeros(n_runs)
    for g in groups:
        delta_g = idata.posterior[f"delta_{g}"].values.mean(axis=(0, 1))
        run_indices = sorted([t for t, v in run_to_version.items() if v == g])
        for pos, t in enumerate(run_indices):
            delta_run_mean[t] = delta_g[pos]

    log_lambda_run_mean = idata.posterior["log_lambda_run"].values.mean(axis=(0, 1))
    mu_hat = mu_version_mean[group_idx] + beta_drift_mean * run_order_scaled + delta_run_mean[run]
    lambda_hat = np.exp(log_lambda_run_mean[run])
    inverse_gaussian_variance = mu_hat**3 / lambda_hat
    return (y - mu_hat) / np.sqrt(inverse_gaussian_variance)


def _report_residual_acf(
    idata: az.InferenceData,
    y: np.ndarray,
    run: np.ndarray,
    group_idx: np.ndarray,
    groups: list[str],
    n_runs: int,
    run_to_version: dict[int, str],
) -> None:
    print("\n  --- Residual Autocorrelation ---", flush=True)

    std_resid = _compute_standardized_residuals(idata, y, run, group_idx, groups, n_runs, run_to_version)

    acf_std = _acf_per_run(std_resid, run, n_runs)
    print("    Standardized residual ACF:", flush=True)
    for lag in range(1, 4):
        vals = acf_std[lag]
        if vals:
            print(f"      ACF({lag}): mean={np.mean(vals):.3f}, median={np.median(vals):.3f}", flush=True)

    acf_sq = _acf_per_run(std_resid**2, run, n_runs)
    n_obs_typical = int(np.median([np.sum(run == t) for t in range(n_runs)]))
    threshold = 2 / np.sqrt(max(n_obs_typical, 1))
    print("    Squared residual ACF (volatility):", flush=True)
    for lag in range(1, 4):
        vals = acf_sq[lag]
        if vals:
            v = np.mean(vals)
            sig = "SIGNIFICANT" if abs(v) > threshold else "ok"
            print(f"      ACF({lag}): mean={v:.3f} [{sig}]", flush=True)
