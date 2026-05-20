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


## M3 measurement uncertainty primitives

M3 adds measurement-uncertainty covariance terms that are named separately from observation noise. These terms consume legacy-adjusted predictions and control axes; they do not mutate legacy `Reference Evaluations`, `Standard Deviation`, or indentation `d0` handling.

### Contact and alignment uncertainty

Contact/alignment uncertainty is represented as rank-structured covariance over a curve:

- `contact_offset`: `sigma_contact^2 * ones * ones.T`
- `alignment_tilt`: `sigma_alignment^2 * centered_controls * centered_controls.T`
- `displacement_scale_calibration`: `sigma_displacement_scale^2 * predictions * predictions.T`
- `force_scale_calibration`: `sigma_force_scale^2 * (controls * sensitivity) * (controls * sensitivity).T`
- `minimum_variance_floor`: diagonal variance floor

`enabled=false` returns a named zero contribution even if sigma fields are present. `enabled=true` with all zero sigmas is numerically inactive but still reports the named components. A nonzero force-scale term requires an explicit force sensitivity vector. Legacy indentation `d0` remains a legacy-wrapper concern; M3 contact/alignment covariance consumes the already adjusted prediction curve to avoid double applying offsets.

### Geometry uncertainty

Geometry uncertainty uses a Jacobian covariance model:

`C_geometry = J_geometry * Sigma_geometry * J_geometry.T`

where `J_geometry[i, k]` is the sensitivity of prediction point `i` to geometry parameter `k`. Parameters carry names, units, optional nominal values, and either diagonal sigmas or an explicit positive semidefinite parameter covariance matrix. The resulting contribution is reported as `geometry_jacobian` plus per-parameter diagonal covariance components such as `geometry:radius` or `geometry:diameter_um`. Explicit off-diagonal parameter covariance is reported as signed cross components named `geometry_cross:<left>:<right>` so named components reconstruct the total geometry covariance instead of hiding covariance terms in the aggregate.

Disabled geometry uncertainty returns a named zero covariance. Enabled geometry uncertainty requires declared parameters, matching sensitivity vectors, finite values, and valid units. It remains distinct from M2 additive/relative observation noise and correlated curve noise to avoid double counting. Diagnostics must include an enabled-versus-disabled measurement-uncertainty comparison, not only component heatmaps.

## M4 surrogate covariance propagation

M4 adds a stable covariance interface for surrogate predictive uncertainty. It keeps the existing legacy BNN standard-deviation path recoverable while allowing richer covariance payloads for later runtime adapters.

Supported representations:

- `diagonal`: pointwise surrogate predictive standard deviations, stored as `surrogate_predictive_diagonal`.
- `full`: a dense positive-semidefinite predictive covariance, stored as `surrogate_predictive_full`.
- `low_rank`: a point-by-rank factor matrix `F` with covariance `F * F.T`, optionally plus a diagonal residual, stored as `surrogate_predictive_low_rank` and `surrogate_predictive_diagonal`.

All forms also report `surrogate_covariance_total`. Disabled surrogate covariance returns a named zero contribution and preserves deterministic DNN/direct-simulation behavior. Enabled surrogate covariance fails early when the selected representation is missing, has the wrong shape, contains nonfinite values, is asymmetric, or is not positive semidefinite.

Legacy EMB BNN predictive standard deviation remains a diagonal surrogate covariance term. Composing M2 relative observation variance with the M4 diagonal surrogate covariance must reproduce the M1 legacy BNN quadrature rule:

`total_std_i = sqrt((relative_sigma * abs(prediction_i))^2 + surrogate_std_i^2)`

M4 does not introduce a general full-hierarchy assembler. Full observation, measurement, surrogate, and discrepancy assembly remains the M5 total-covariance milestone.

## M5 model discrepancy and total covariance assembly

M5 introduces the first shared full-hierarchy covariance assembler. Runtime adapters remain outside this milestone: EMB, GV, Korali reference fields, and legacy standard-deviation paths are not rewritten automatically. The M5 API is invoked explicitly by tests, diagnostics, or later integration adapters.

### Low-rank model discrepancy

Low-rank model discrepancy is disabled by default. When enabled, it uses an explicit basis matrix `B` with shape `[observation_points, rank]` and a marginal coefficient covariance `K`, producing:

`C_discrepancy = B * K * B.T + C_floor`

The builder also supports diagonal coefficient scales with deterministic shrinkage as a convenience for synthetic fixtures. Full coefficient covariance may include cross terms; those are named as `model_discrepancy_cross:<left>:<right>` so named components reconstruct `model_discrepancy_total`. Non-intercept polynomial helper columns are mean-centered to avoid silently absorbing a constant protected offset in fixture diagnostics.

Enabled discrepancy fails early for missing or inconsistent basis rank, nonfinite basis or coefficient covariance, non-symmetric or non-PSD coefficient covariance, zero prior scale, and overflow during covariance assembly. Disabled discrepancy returns a named zero covariance and does not alter lower-stage behavior.


### Discrepancy identifiability diagnostics

M5 discrepancy is useful only when it remains visible as an uncertainty layer rather than silently replacing physical parameters. The identifiability diagnostics evaluate saved run artifacts with:

- discrepancy magnitude relative to response scale, residual scale, and total predictive standard deviation;
- residual variance explained by discrepancy;
- physical parameter shift in posterior-standard-deviation and relative units;
- physical-sensitivity versus discrepancy-basis correlation, including a first canonical correlation;
- prior-to-posterior discrepancy coefficient shrinkage and active coefficient counts;
- covariance trace and diagonal shares for observation, surrogate, measurement, and model-discrepancy components.

A run with discrepancy enabled must record that it was explicitly opted in. Missing component decomposition is a fail-gate condition because reports must distinguish observation noise, surrogate covariance, measurement covariance, and model discrepancy. Null/noise-only controls pass only when discrepancy remains zero and coefficients shrink toward zero. Negative controls fail when discrepancy can mimic protected physical sensitivities or hide missing physics by explaining systematic residual structure.

The reproducible diagnostic entry point is:

`python scripts/qa/noise_m5_discrepancy_identifiability_diagnostics.py --output-root <run-root>`

The script writes a manifest, metrics JSON, fixture summary CSV, residual decomposition, parameter-shift, theta-beta-correlation, covariance-share, shrinkage, and fixture-gate plots. The manifest records the executed command, Git branch/SHA/status, thresholds, expected-versus-actual fixture gate statuses, artifact paths, and residual-risk notes so the diagnostics can be audited from saved run artifacts.

### Total covariance assembly

The M5 assembler accepts explicit `CovarianceTerm` objects. Only terms marked `included=True` are numerically summed. Child components from M2-M4 builders are preserved in diagnostics, but are not summed when a parent total is included. This prevents double counting patterns such as including both `measurement:geometry` and `geometry:radius`/`geometry_cross:*` as independent summands.

Canonical included terms for the full hierarchy are:

- `observation:additive_relative`
- `observation:correlated_curve`
- `measurement:contact_alignment`
- `measurement:geometry`
- `surrogate:predictive`
- `discrepancy:low_rank`

Duplicate term names, duplicate aliases, child/parent identifier collisions, shape mismatches, nonfinite values, asymmetric parent matrices, and non-PSD parent covariance contributions are hard errors. Signed child diagnostics such as geometry cross terms may be non-PSD because they are never summed independently.

The final active total covariance must be positive definite for likelihood use. Singular low-rank terms are allowed as components, but the assembled total must either be positive definite or provide an explicit final jitter policy. Child jitter is retained as diagnostics and is not added a second time.

## M6 synthetic recovery

M6 synthetic recovery is validation/reporting over the M1-M5 hierarchy, not a change to inference semantics. The lightweight recovery harness uses deterministic linear-Gaussian fixtures with named covariance components for:

- legacy/noise-only behavior;
- measurement uncertainty;
- surrogate predictive covariance;
- opt-in discrepancy covariance.

Recovery metrics include parameter bias, absolute error, posterior-standard-deviation z-error, interval coverage, residual RMSE, finite-observable checks, design/covariance conditioning, and covariance group trace shares. CI-scale thresholds are intentionally conservative plumbing checks; they do not claim final EMB posterior calibration.

The reproducible diagnostic entry point is:

`python scripts/qa/noise_m6_synthetic_recovery_diagnostics.py --output-root <run-root>`

The script writes a root synthetic manifest, metrics JSON, summary CSV, per-scenario configs/truth/observations/covariance summaries/recovery reports, and plots for parameter intervals, observable overlays, whitened residuals, and covariance heatmaps.

## M6 predictive checks and SBC

M6 predictive checks are validation/reporting over the assembled hierarchy. They compare observed fixture summaries against posterior predictive distributions and run SBC-style rank calibration checks on lightweight deterministic records. They do not change the likelihood API or make production EMB calibration claims.

The first supported diagnostic scenarios are:

- `baseline_legacy`, covering the legacy/noise-only predictive surface;
- `full_hierarchy`, covering observation, measurement, surrogate, and discrepancy covariance terms together.

Predictive-check reports intentionally separate runtime/numerical failures from calibration failures. Runtime/numerical failures cover nonfinite predictive payloads or invalid input shapes. Calibration failures cover posterior predictive summary z-scores, pointwise predictive interval coverage, predictive rank edge concentration, SBC rank histogram distance, SBC mean rank quantile error, SBC edge concentration, and SBC posterior interval coverage.

The CI-scale thresholds in `configs/noise/predictive_checks.example.yaml` are conservative fixture gates. They are strict enough to catch broken uncertainty plumbing, swapped dimensions, missing rank records, and collapsed predictive intervals, but MES-37 remains responsible for production EMB comparison evidence.

The reproducible diagnostic entry point is:

`python scripts/qa/noise_m6_predictive_checks_diagnostics.py --output-root <run-root>`

The script writes a root predictive manifest, metrics JSON, summary CSV, per-scenario configs/observations/rank records/reports, and plots for PPC observable overlays, PPC summary intervals, SBC rank histograms, and calibration coverage summaries.

## M7 integrated EMB comparison

M7 compares paired legacy and upgraded EMB likelihood behavior on tracked EMB reference-data fixtures before any production posterior claim is made. The local diagnostic keeps legacy and upgraded modes in the same command surface: legacy uses the M1 compatibility wrappers, while upgraded mode uses the full-hierarchy total covariance terms from M5.

The required validation scenarios are:

- `emb_compression_reference`, using `emb/compression/evalkit/data/data_1.csv`;
- `emb_indentation_reference`, using `emb/indentation/evalkit/data/data_morris_3.40.csv`.

The diagnostic reports posterior interval width ratios, posterior mean shifts in pooled posterior-standard-deviation units, predictive band width ratios, legacy/upgraded residual RMSE, interval coverage, standardized residuals, and covariance trace-share attribution. The report interpretation classifies whether upgraded uncertainty broadened, shifted, or stabilized the comparison relative to legacy behavior.

The reproducible diagnostic entry point is:

`python scripts/qa/noise_m7_emb_comparison_diagnostics.py --output-root <run-root>`

The script writes an EMB comparison manifest, metrics JSON, Markdown report, summary CSV, per-scenario configs/predictions/posterior samples/covariance summaries/reports, and plots for posterior intervals, predictive bands, residual diagnostics, and the metrics table.

M7 evidence carries an explicit evidence class. The default local diagnostic is `validation_fixture` with `production_claim=false`; it is validation evidence for paired workflow plumbing and reporting, not a production EMB posterior campaign.

Gate 06 is a reader over existing evidence, not another generator. Run it with:

`python scripts/qa/noise_gate06_integrated_emb.py --synthetic-manifest <MES-34 manifest> --predictive-manifest <MES-35 manifest> --emb-manifest <MES-37 manifest> --output-root <gate-root>`

Use `--require-production` only when closing a production-posterior claim. Validation fixture evidence must fail that stricter gate rather than masquerading as production evidence.

## M8 release readiness and reporting

M8 adds the release-readiness surface for the hierarchy without changing inference semantics. The public mode names are:

- `legacy`, preserving the M1 compatibility wrappers and default behavior;
- `noise_primitives`, exposing M2 observation-noise and robust-likelihood primitives;
- `measurement_uncertainty`, exposing M3 contact/alignment and geometry terms;
- `surrogate_covariance`, exposing M4 surrogate predictive covariance;
- `discrepancy` and `full_hierarchy`, exposing the M5 total-covariance assembly;
- `synthetic_recovery` and `predictive_checks`, exposing the M6 validation diagnostics;
- `emb_comparison`, exposing the M7 paired EMB comparison diagnostic.

The config validator accepts the historical `kind: noise` examples and the newer `family: noise_hierarchy` validation configs. It rejects missing schema metadata, unsupported families, absolute path literals, and private HPC path literals so checked-in examples remain portable across Vega, Karolina, and local CI.

Run the release-readiness report with existing M6/M7/Gate06 evidence:

`python scripts/qa/noise_hierarchy_release_readiness.py --mode full_hierarchy --synthetic-manifest <MES-34 manifest> --predictive-manifest <MES-35 manifest> --emb-manifest <MES-37 manifest> --gate06-manifest <Gate06 manifest> --output-root <release-root>`

The command writes:

- `noise_release_readiness_manifest.json`, including command, seed-free deterministic evidence references, commit, branch, Python environment, config validation, mode surface, merge boundary, skips, and residual risk;
- `noise_artifact_index.json`, indexing the synthetic recovery, predictive check, and EMB comparison manifests and their sidecars;
- `noise_config_validation.json`, recording per-config pass/fail/warning state;
- `noise_release_readiness_report.md`, a human-readable report that distinguishes legacy, staged, and full-hierarchy templates and links the evidence paths.

Gate 07 reads the release manifest and verifies the closeout-facing claims:

`python scripts/qa/noise_gate07_release_checks.py --release-manifest <release-root>/noise_release_readiness_manifest.json --output-root <gate07-root>`

Gate 07 passes when configs pass, required evidence entries exist and report clean scenario gates, Gate 06 passes, the project is either merged or explicitly at a human review/merge boundary, and the manifest records that no active Karolina worktree or session was touched.
