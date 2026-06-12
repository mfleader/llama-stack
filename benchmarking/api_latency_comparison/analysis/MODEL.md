# Response Latency Model

[Wald (Inverse Gaussian)](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
hierarchical model with Hilbert Space Gaussian Process (HSGP) temporal
adjustment. Detects mean latency shifts between OGX versions at a 1ms
threshold.

Response times are strictly positive and right-skewed (occasional slow
requests). The Wald (Inverse Gaussian) captures this naturally, while a Normal would allow
impossible negative times.

## Likelihood

```text
y_i ~ Wald/InvGaussian(mu_i, lambda_i)
```

## Mean (Location) Structure

```text
mu_i = mu_version[g_i] + beta_drift * d_i + delta_run[t_i] + f(x_i)
```

## Shape Structure

Lambda is the Wald (Inverse Gaussian) shape parameter: higher values produce a narrower
peak and lighter tails. Spread scales as mu^3 / lambda, so higher
lambda means more consistent response times at the same mean latency.

```text
lambda_i = exp(log_lambda_run[t_i])
log_lambda_run[t] ~ Normal(log_lambda_bar, sigma_lambda)
```

## Run Intercepts

Each run gets its own intercept (delta_run) to account for
run-to-run variation (e.g., different thermal state, background load).
Within each version group, the intercepts are independently constrained
to sum to zero via [ZeroSumNormal](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html)
so that the version mean latency (mu_version) cleanly represents
each version's average response time.

## Experimental Design (RCBD)

The run order follows a Randomized Complete Block Design. Each block
contains exactly one run of each version (baseline, comparison,
comparison_ctrl) in a randomly permuted order. Blocks execute in
temporal sequence (block 1 first, block N last).

Blocking prevents temporal drift from aliasing with the treatment
effect: any monotonic trend affects all three versions within a block
approximately equally. Within-block randomization prevents position
effects (e.g., cold-start penalty for the first run in a block)
from systematically favoring one version.

beta_drift operates on the global chronological run position d_i,
scaled to [0, 1] across the full experiment. It captures smooth
linear drift that the block structure does not resolve (e.g.,
gradual thermal ramp within a block). The two mechanisms are
complementary: blocking removes arbitrary between-block shifts,
beta_drift removes smooth within-experiment trends.

## Gaussian Process Temporal Adjustment

Sequential observations within a run share transient system state
(GC pressure, connection pool warm-up, event loop saturation),
producing temporal autocorrelation. The GP captures this within-run
dependence so that credible intervals on the version effect (beta_v)
reflect the true effective sample size. Run intercepts (delta_run)
capture between-run level shifts; the GP captures within-run
temporal structure by operating on the sequence number of each
request within a run (1st request, 2nd request, ...).

The [HSGP (Hilbert Space Gaussian Process)](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.HSGP.html)
is an approximation to a full Gaussian Process that scales
linearly in the number of observations, making it practical for
the large datasets typical of automated performance experiments.
[Matern 3/2](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)
kernel: nearby observations are correlated, correlation decays
smoothly with distance.

```text
f ~ HSGP(Matern32, m=20, c=1.5, noncentered, drop_first=True)
```

## Priors

```text
mu_version[g]     ~ Normal(25, 10),       for each version group
beta_drift        ~ Normal(0, 2)
sigma_run       ~ Exponential(1)
delta_run[t]    ~ ZeroSumNormal(sigma_run),  independently per group
log_lambda_bar    ~ Normal(7, 2)
sigma_lambda      ~ Exponential(1)
eta_gp            ~ HalfNormal(0.25)
ell_gp            ~ InverseGamma(mu=6, sigma=3)
```

## Derived Quantities

```text
beta_v[g] = mu_version[g] - mu_version[baseline]
```

## Definitions

A bare index like `g` or `t` ranges over all values (used in priors).
A subscripted index like `g_i` or `t_i` is a lookup: it maps
observation `i` to its group or run.

The `_bar` suffix denotes a population average: the center of the
distribution that individual values are drawn from.

| Symbol | Known/Estimated | Code variable | Description |
|---|---|---|---|
| i | index | — | Observation index |
| g | index | — | Version group index (baseline, comparison, comparison_ctrl) |
| t | index | — | Run index |
| y_i | known | `y` | Response time for observation i (ms, positive) |
| g_i | known | `x_group` | Version group that observation i belongs to |
| t_i | known | `x_run` | Run that observation i belongs to |
| d_i | known | `x_drift` | Run's chronological position in the experiment, scaled to [0, 1] |
| x_i | known | `x_time` | Observation sequence number within a run |
| mu_version[g] | estimated | `mu_version` | Mean latency for version group g (baseline, comparison, comparison_ctrl) (ms) |
| beta_drift | estimated | `beta_drift` | Linear drift over run ordering (ms) |
| sigma_run | estimated | `sigma_run` | Hierarchical standard deviation for run intercepts (ms) |
| delta_run[t] | estimated | `delta_run_full` | Run-level deviation from group mean (ms), ZeroSumNormal within each group |
| log_lambda_bar | estimated | `log_lambda_bar` | Population average log shape across runs |
| sigma_lambda | estimated | `sigma_lambda` | Between-run standard deviation in log shape |
| log_lambda_run[t] | estimated | `log_lambda_run` | Per-run log shape (centered parameterization) |
| lambda_i | deterministic | `lambda_obs` | exp(log_lambda_run[t_i]). Wald (Inverse Gaussian) shape parameter |
| eta_gp | estimated | `eta_gp` | How far the GP can shift latency within a run (ms) |
| ell_gp | estimated | `ell_gp` | How many observations the GP effect stays correlated over |
| f(x) | estimated | `f` | HSGP Matern32 within-run temporal adjustment |
| beta_v[g] | deterministic | `beta_v` | How much slower (positive) or faster (negative) version g is compared to baseline, in ms |

## Data Filtering

First and last observation per run are dropped. The first is a
Locust client warmup artifact (+20ms). The last is an edge-of-window
effect.

## Code Reference

| What | File | Function |
|---|---|---|
| Model definition and data filtering | `fit_resp_latency_model.py` | `build_model()` |
| Fitting and LOO | `fit_resp_latency_model.py` | `fit_and_diagnose()` |
