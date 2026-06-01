# Active Learning Final Gate

This document is the final-gate contract for the active-learning production slice described by
[`ACTIVE_LEARNING_ENGINE.md`](ACTIVE_LEARNING_ENGINE.md).

## Run scope and execution policy

- **Domain:** EMB indentation workflow.
- **Indentation diameter:** `3.4um`.
- **Prior data state:** start from **no prior training data**.
- **Active candidate controls:** `ka` and `kb` only.
  - `ka` in log-space: `[1e2, 6e5]`.
  - `kb` in log-space: `[400, 7e4]`.
- **Legacy compatibility:** `Yt` is derived only from `ka` for legacy Mirheo payload compatibility; it is never an active selected candidate dimension.
- **Sampling policy:** 3-round dynamic strategy:
  - Round 1: `initial_sobol_maximin` (30 curves).
  - Round 2: `ensemble_disagreement_diversity` with `6` exploration + `24` acquisition curves, then greedy diversity selection.
  - Round 3: `ensemble_disagreement_diversity` with `6` exploration + `24` acquisition curves, then greedy diversity selection.
- **Budget:** `3 rounds × 30 curves / round`, each curve is a full force sweep.
- **Comparator:** `LHS 90-curve comparator` baseline for `30/60/90` prefix comparison against AL prefixes.
- **Canary:** `1 curve × 3 force points` before full campaign admission.
- **Platform:** `Karolina` with `30 concurrent jobs`.
- **Retry policy:** retry limit `3`.
- **Final gate evidence shape:** AL prefixes `30`, `60`, `90` curves; LHS prefixes `30`, `60`, `90` curves.
- **Labeling policy:** fresh DPD labels only.
- **Force grid input:** `samples_all.dat` only.
- **Runtime preflight:** every fresh-DPD production array must pass the scratch-backed DPD production preflight described in
  [`DPD_PRODUCTION_PREFLIGHT.md`](DPD_PRODUCTION_PREFLIGHT.md).
- **Dual-HPC closeout:** Karolina production may proceed under Karolina preflight evidence, but project closeout remains blocked until the same preflight canary passes on Vega.

## Required validation plots (every step)

At every active-learning step, the workflow must produce and record:

- initial round-1 candidate/selected sample coverage (`initial_round1_samples`),
- runtime-per-curve
- selected samples ka/kb overlays
- per-round additions
- exploration vs acquisition split
- disagreement-acquisition map
- force overlays
- AL-vs-LHS relative `L2` comparison
- failure/quarantine/replacement

These outputs must be kept as part of the run payload and included in Linear review links.

## Final pass/fail gate

Final acceptance requires all plots above to be present and the following gating checks to pass:

- `runtime_per_curve` must be complete and free from missing values.
- `samples_ka_kb` and `force_curve_overlays` must show full selected sets for each round.
- `per_round_additions` must match the 30/60/90 acceptance cadence.
- `exploration_vs_acquisition` and `disagreement_acquisition_map` must show non-starved acquisition in target regions.
- `failure_quarantine_replacement` must document each quarantine, retry action, and replacement decision.
- `al_vs_lhs_relative_l2` must show AL improvement against each LHS prefix (30, 60, 90) where possible.
- no failed or interrupted DPD candidate output may be used as training, scoring, or held-out evidence;
- Slurm-accounted `TIMEOUT` candidates must be quarantined/replaced; Slurm-accounted scheduler interruptions such as
  `CANCELLED`, `PREEMPTED`, `NODE_FAIL`, or `REVOKED` must be retried as the same candidate;
- replacement batches must be generated through the resume helper, resumed only for missing replacement indices, and recorded in the campaign manifests;
- the project cannot be marked dual-HPC complete until the Vega preflight canary passes.

## Production runtime fingerprint

The production run uses the fixed runtime envelope below:

```text
radp=6.80
L=25
fscale=0.0074
shell_th=5e-9
numsteps=5000
numsteps_eq=10000
```

Force axis for the `3.4um` dataset must be exactly:

- `0.0`
- `357.14285714285717`
- `714.2857142857143`
- `1071.4285714285716`
- `1428.5714285714287`
- `1785.7142857142858`
- `2142.857142857143`
- `2500.0`
- `2857.1428571428573`
- `3214.2857142857147`
- `3571.4285714285716`
- `3928.571428571429`
- `4285.714285714286`
- `4642.857142857143`
- `5000.0`

## PR and work sequencing (Linear/vault ready)

1. Open one PR with this gate doc and `configs/active_learning/final_gate_plan.example.yaml`
   plus updated evidence links.
2. Attach the Linear task IDs for:
   - gate setup validation
   - per-step plot inventory
   - retry/quarantine evidence
3. Keep this gate as GitHub-check-blocking until plot evidence and runtime
   fingerprint matches are present.
4. Archive vault-ready evidence package paths (`payload + gate manifest`) in the PR
   description and close with a no-open-item checklist.
