#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Posterior predictive quantile decisions and false positive checks."""

from datetime import UTC, datetime
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
from tabulate import tabulate

QUANTILES = [50, 95, 99]
THRESHOLD_MS = 1.0


def compute_hdi(samples: np.ndarray, prob: float = 0.99) -> tuple[float, float]:
    """Highest density interval from 1D samples via az.hdi()."""
    result = az.hdi(np.asarray(samples), prob=prob)
    return float(result[0]), float(result[1])


def _make_decision(contrast: np.ndarray, block: str, q: int, version: str) -> dict[str, Any]:
    """Build a single quantile decision dict from posterior contrast draws."""
    pr_below = float(np.mean(contrast <= THRESHOLD_MS))
    hdi_lo, hdi_hi = compute_hdi(contrast, 0.99)
    return {
        "block": block,
        "quantile": q,
        "version": version,
        "diff_ms": float(np.mean(contrast)),
        "sd": float(np.std(contrast)),
        "pr_below_threshold": pr_below,
        "detected": pr_below < 0.05,
        "hdi_lo": float(hdi_lo),
        "hdi_hi": float(hdi_hi),
    }


def _format_decisions_table(decs: list[dict[str, Any]]) -> str:
    """Format a list of decision dicts as a tabulate table string."""
    rows = []
    for d in decs:
        verdict = "YES" if d["detected"] else "no"
        interval = f"[{d['hdi_lo']:+.1f}, {d['hdi_hi']:+.1f}]"
        rows.append(
            [
                d["version"],
                f"p{d['quantile']}",
                f"{d['diff_ms']:+.2f}",
                f"{d['sd']:.2f}",
                f"{d['pr_below_threshold']:.4f}",
                verdict,
                interval,
            ]
        )
    return tabulate(
        rows,
        headers=["Version", "Quantile", "Diff(ms)", "SD(ms)", "P(<=1ms)", "Verdict", "99% interval"],
        colalign=("left", "left", "right", "right", "right", "right", "left"),
    )


def _check_false_positive(
    pp: np.ndarray,
    group_masks: dict[str, np.ndarray],
    fp_base: str | None,
    fp_ctrl: str | None,
    decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run false positive check between two same-code groups.

    Returns (fp_decisions, fp_results) where fp_results is the JSON-serializable
    dict written to fp-results.json.
    """
    print(f"  === False Positive Check: {fp_ctrl} vs {fp_base} ===", flush=True)
    fp_results = {"false_positive_detected": False, "quantiles": {}}
    if not (fp_base and fp_ctrl and fp_base in group_masks and fp_ctrl in group_masks):
        print("  WARNING: Cannot run false positive check — no matching version_hash pair found", flush=True)
        return [], fp_results

    any_fp = False
    fp_decs = []
    for q in QUANTILES:
        contrast = np.percentile(pp[:, group_masks[fp_ctrl]], q, axis=1) - np.percentile(
            pp[:, group_masks[fp_base]], q, axis=1
        )
        dec = _make_decision(contrast, "false_positive", q, fp_ctrl)
        if dec["detected"]:
            any_fp = True
        decisions.append(dec)
        fp_decs.append(dec)
        fp_results["quantiles"][str(q)] = {
            "diff_ms": dec["diff_ms"],
            "pr_below_threshold": dec["pr_below_threshold"],
            "detected": dec["detected"],
        }
    print(_format_decisions_table(fp_decs), flush=True)
    fp_results["false_positive_detected"] = any_fp
    if any_fp:
        print("  RESULT: FALSE POSITIVE DETECTED — experiment apparatus is unreliable for this run", flush=True)
    else:
        print("  RESULT: clean", flush=True)
    return fp_decs, fp_results


def compute_decisions(
    model: pm.Model, idata: az.InferenceData, data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute posterior predictive quantile decisions and save results."""
    y = data["y"]
    group_labels = data["group_labels"]
    groups = data["groups"]
    baseline = data["baseline"]

    ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
    print(f"[{ts}] --- Posterior Predictive Quantile Decisions ---", flush=True)

    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    pp = idata.posterior_predictive["y"].values.reshape(-1, len(y))

    group_masks = {g: (group_labels == g) for g in groups}
    fp_base = "comparison" if "comparison" in groups else None
    fp_ctrl = "comparison_ctrl" if "comparison_ctrl" in groups else None
    non_ref = [g for g in groups if g != baseline]
    comparison_groups = [g for g in non_ref if g != fp_ctrl]
    beta_v_draws = idata.posterior["beta_v"].values.reshape(-1, len(groups))
    beta_v_summaries = {}

    for j, name in enumerate(groups):
        if name == baseline or name == fp_ctrl:
            continue
        draws = beta_v_draws[:, j]
        mean_val = float(np.mean(draws))
        hdi_lo, hdi_hi = compute_hdi(draws, 0.99)
        beta_v_summaries[name] = (mean_val, hdi_lo, hdi_hi)

    decisions = []

    for name in comparison_groups:
        for q in QUANTILES:
            contrast = np.percentile(pp[:, group_masks[name]], q, axis=1) - np.percentile(
                pp[:, group_masks[baseline]], q, axis=1
            )
            decisions.append(_make_decision(contrast, "comparison", q, name))

    _, fp_results = _check_false_positive(pp, group_masks, fp_base, fp_ctrl, decisions)

    print(f"  {'=' * 60}", flush=True)
    print(f"  RESULTS (regression threshold: {THRESHOLD_MS}ms)", flush=True)
    print(f"  {'=' * 60}", flush=True)
    if fp_results.get("false_positive_detected"):
        print("  FALSE POSITIVE DETECTED — results are unreliable", flush=True)
    summary_rows = []
    for name in comparison_groups:
        mean_shift, mean_hdi_lo, mean_hdi_hi = beta_v_summaries[name]
        summary_rows.append(
            [f"{name} vs {baseline}", "mean", f"{mean_shift:+.1f}", f"[{mean_hdi_lo:+.1f}, {mean_hdi_hi:+.1f}]", ""]
        )
        for dec in decisions:
            if dec["block"] == "comparison" and dec["version"] == name:
                verdict = "REGRESSION" if dec["detected"] else "no regression"
                summary_rows.append(
                    [
                        "",
                        f"p{dec['quantile']}",
                        f"{dec['diff_ms']:+.1f}",
                        f"[{dec['hdi_lo']:+.1f}, {dec['hdi_hi']:+.1f}]",
                        verdict,
                    ]
                )
    print(
        tabulate(summary_rows, headers=["Comparison", "Quantile", "Diff(ms)", "99% interval", ""], tablefmt="plain"),
        flush=True,
    )
    print(f"  {'=' * 60}", flush=True)

    return decisions, fp_results
