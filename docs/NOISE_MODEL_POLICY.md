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
