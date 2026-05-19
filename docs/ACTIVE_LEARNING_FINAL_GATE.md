# Active Learning Final Gate

This document is the final-gate contract for the active-learning production slice described by
[`ACTIVE_LEARNING_ENGINE.md`](ACTIVE_LEARNING_ENGINE.md).

## Run scope and execution policy

- **Domain:** EMB indentation workflow.
- **Indentation diameter:** `3.4um`.
- **Prior data state:** start from **no prior training data**.
- **Candidate dimensions:** `Yt` and `kb` only.
  - `Yt` bounds: `1.0e5` to `1.0e9`.
  - `kb` bounds: `400.0` to `70000.0`.
- **Budget:** `3 rounds × 30 curves / round`, each curve is a full force sweep.
- **Comparator:** `LHS 90-curve comparator` for baseline coverage checks.
- **Canary:** `1 curve × 3 force points` before full campaign admission.
- **Platform:** `Karolina` with `30 concurrent jobs`.
- **Retry policy:** retry limit `3`.

## Required validation plots (every step)

At every active-learning step, the workflow must produce and record:

- grouped-holdout median curve relative `L2` primary
- mean curve relative `L2`
- max curve relative `L2`
- residuals
- predicted/reference curves
- coverage/acquisition
- failure/quarantine status
- model-selection diagnostics
- `AL-vs-LHS` summary

These outputs must be kept as part of the run payload and included in Linear review links.

## Final pass/fail gate

Final acceptance requires all plots above to be present and the following gating checks to pass:

- grouped-holdout median curve relative `L2` improves for AL versus the 90-curve LHS comparator.
- mean/max curve relative `L2` supports the grouped-holdout median result.
- residual diagnostics show no unbounded drift across rounds.
- predicted/reference overlays remain physically consistent and monotone where expected.
- coverage/acquisition indicates no acquisition starvation in high-likelihood regions.
- failure/quarantine counts are explained with retry actions and quarantine resolution.
- model-selection selects a production-ready surrogate and optimizer path.
- AL-vs-LHS summary confirms active-learning gain over random/comparator baseline.

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
