# Noise Model Policy

This policy defines the lightweight noise model contracts introduced in MES-158.

## Uncertainty terms and roles

- **Measurement error**: uncertainty in the recorded observable from instrumentation or
  assay repeatability. It belongs to the data domain and is required for all primary
  supported models unless explicitly marked optional.
- **Model discrepancy**: structural mismatch between surrogate model assumptions and the
  physical model family. This is a separate additive contribution and is only attached
  where the model declares support for it.
- **Surrogate uncertainty**: uncertainty from the surrogate predictor itself
  (training or approximation error). This is optional and may be deterministic.
- **Posterior uncertainty**: uncertainty induced by finite inference posterior spread in
  latent parameters, represented here as an additional independent variance term for
  toy numeric composition tests.

The policy keeps these components distinct and additive. Each component contributes its own
variance, and the toy likelihood composes them as:

`total_variance = measurement^2 + discrepancy^2 + surrogate^2 + posterior^2`

This preserves the conceptual distinction even when one or more components is absent.

## Model support contracts

Each registered model declares:

- the model identifier and family label,
- supported observables and their units,
- whether measurement error is required and the accepted measurement-error kinds,
- whether discrepancy, surrogate uncertainty, or posterior uncertainty are supported.

Configurations must match the model support metadata. Missing required measurement error
or attaching a component that a model does not support is treated as a validation error.

## Legacy compatibility layer

M1 keeps existing EMB and GV likelihood behavior recoverable through shared wrapper functions before adding richer covariance terms. The wrapper preserves the current runtime conventions:

- EMB compression surrogate DNN mode uses `sigma * force` directly.
- EMB compression surrogate BNN mode combines surrogate predictive standard deviation and `sigma * abs(force_mean)` in quadrature.
- EMB indentation surrogate mode applies the `d0` displacement offset, clips at zero, then uses `sigma * adjusted_displacement`; BNN mode adds predictive standard deviation in quadrature.
- Direct Mirheo compression keeps constant absolute `sigma`; direct Mirheo indentation keeps proportional `sigma * final_distance`.
- GV Phase 1 uses the declared multiplicative `sigma` noise model and emits Korali `Standard Deviation` through the shared legacy wrapper.

The staged composite interface names rollout stages `M0` through `M7`. `M0` and `M1` are legacy/baseline compatibility stages; later stages add noise primitives, measurement uncertainty, surrogate covariance, discrepancy, validation diagnostics, and integrated EMB comparison without duplicating legacy variance math in runtime call sites.


## M2 noise primitives

M2 adds executable primitives for richer observation-noise and likelihood composition while leaving the M1 legacy wrappers stable.

### Additive plus relative observation noise

The additive/relative primitive is nonnegative by construction and exposes named variance components:

`total_variance_i = additive_sigma^2 + (relative_sigma * max(abs(prediction_i), prediction_scale_floor))^2 + floor_i`

`floor_i` is zero unless `minimum_total_variance` is larger than the raw additive-plus-relative variance. Relative-only configurations with zero predictions may assemble zero variance, but likelihood evaluation must fail unless an additive term or explicit floor makes the total variance positive. This keeps legacy zero-standard-deviation behavior quarantined in M1 while giving M2 a stable finite-likelihood path.

### Correlated curve noise

Correlated curve noise is an optional covariance contribution for curve-indexed residuals. The first supported kernel is squared exponential. Kernel amplitude is a standard deviation; covariance uses `amplitude^2`. Multi-curve inputs are block diagonal in M2, with no cross-curve covariance. Builders record covariance shape, diagonal and correlation ranges, eigenvalue range, condition number, jitter applied, Cholesky status, and curve identifiers.

A disabled or zero-amplitude correlated component returns a zero covariance contribution. Total covariance composition remains responsible for adding diagonal variance and for failing clearly if the final covariance is singular and no jitter policy is supplied.

### Heavy-tail robust likelihood

Robust likelihood is opt-in. The default remains Gaussian and preserves the existing diagonal Normal likelihood when heavy-tail mode is absent. The first robust mode is scaled Student-t with `2 < degrees_of_freedom <= 100`. Diagonal Student-t uses pointwise standard deviations; full-covariance Student-t uses the covariance Cholesky factor and Mahalanobis distance. Robust likelihood rejects nonpositive scales and non-positive-definite covariance instead of silently repairing invalid measurement models.

Korali `Bayesian/Reference` Normal remains the M1 runtime boundary. Full-covariance and scaled heavy-tail likelihoods should attach through an explicit custom likelihood adapter in later integration work rather than overloading the legacy reference fields.
