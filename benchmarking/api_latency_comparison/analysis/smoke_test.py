#!/usr/bin/env python3
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Pre-flight smoke test for the analysis modeling stack.

Validates the full chain used by fit_resp_latency_model.py:
nutpie compile+sample, NetCDF save/reload, and LOO computation.
"""

import tempfile
import time
import traceback
from pathlib import Path

import arviz as az
import numpy as np
import nutpie
import pymc as pm
import wald_numba  # noqa: F401 — registers numba dispatch for Wald (Inverse Gaussian) RV

if __name__ == "__main__":
    print("[smoke_test] Starting analysis stack smoke test...", flush=True)
    t0 = time.time()

    try:
        rng = np.random.default_rng(42)
        y = rng.wald(mean=25, scale=9000, size=20)

        with pm.Model() as model:
            mu = pm.Normal("mu", mu=25, sigma=10)
            lam = pm.HalfNormal("lam", sigma=100)
            pm.Wald("y", mu=mu, lam=lam, observed=y)

        compiled = nutpie.compile_pymc_model(model)
        idata = nutpie.sample(compiled, draws=50, tune=50, chains=2, seed=42, progress_bar=False)
        print(f"  nutpie sample: {idata.posterior.dims['draw'] * idata.posterior.dims['chain']} draws", flush=True)

        with model:
            pm.compute_log_likelihood(idata)

        with tempfile.TemporaryDirectory(prefix="smoke-") as tmp:
            nc = Path(tmp) / "smoke.nc"
            idata.to_netcdf(str(nc))
            reloaded = az.from_netcdf(str(nc))
            assert hasattr(reloaded, "posterior")

        print("  NetCDF save/reload: OK", flush=True)
        loo = az.loo(idata, pointwise=True)
        print(f"  LOO ELPD: {loo.elpd:.1f}", flush=True)
        print(f"\n[smoke_test] PASSED in {time.time() - t0:.1f}s", flush=True)

    except Exception as e:
        print(f"\n[smoke_test] FAILED: {e}", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from e
