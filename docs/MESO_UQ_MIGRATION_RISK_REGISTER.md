# MesoUQ migration risk register

This register tracks the risks that matter for the Wave 0 migration closeout.

## Register

| Risk | Why it matters | Current controls | Evidence to watch |
|---|---|---|---|
| Paper reproducibility drift | Public figure and replay claims break if manifests, paper assets, or exact replay inputs move out of sync | Keep `papers/` for now, keep paper-facing artifacts manifest-backed, preserve exact replay docs | `docs/HUQ_EMB_EXACT_FIGURE_REPLAY.md`, `docs/RELEASE_EVIDENCE_CHECKLIST_v0.1.0.md`, `tests/test_gv_paper_replay_*.py` |
| HPC operation regressions | Site-specific launch behavior can fail on Karolina or Vega if output roots, partitions, or env bootstrap drift | Keep site-aware launchers, doctor flows, and canonical output-root rules | `docs/KAROLINA_FULL_PLATFORM.md`, `docs/VEGA_BOOTSTRAP.md`, `docs/VEGA_ACCEPTANCE_COMMAND.md`, `tests/test_karolina_validation_matrix.py` |
| Public API breakage | Breaking `import meso_uq` or the public workflow commands would invalidate downstream users | Keep one release cycle of shims and warnings, keep import tests green | `tests/test_bootstrap.py`, `tests/test_package_import.py`, `tests/test_validation_matrix.py`, `tests/test_vega_acceptance_smoke.py` |
| Artifact movement | Moving generated data out of git can orphan scripts that still read old paths | Use manifests, preserve compatibility roots temporarily, and migrate callers before removing legacy trees | `docs/RELEASE_SCOPE.md`, `tests/test_gv_runtime_governance.py`, `tests/test_gv_pipeline_governance.py` |
| Serialized artifact incompatibility | Pickles and saved model states can become unreadable across version or schema changes | Record commit, runtime, and artifact checksums; keep compatibility tests for serialization round trips | `tests/test_surrogate_pickle_compat.py`, `tests/test_release_assets_smoke.py`, `tests/test_script_manifests.py` |
| Validation matrix breakage | The GPU closeout gate is only useful if the validation matrix still executes end to end | Keep Karolina/Vega matrix wrappers, site-neutral dispatch, and output-root checks stable | `tests/test_validation_matrix.py`, `tests/test_karolina_validation_matrix.py`, `tests/test_validation_runner_smoke.py` |
| GV staging drift | GV currently relies on a compatibility bridge between provenance roots and legacy import roots | Move toward manifest-driven staging without dropping the bridge early | `tests/test_gv_pipeline_governance.py`, `tests/test_gv_runtime_governance.py`, `docs/GV_EXTENSION_CLOSEOUT.md` |

## Required response pattern

When a risk turns into a real failure:

1. capture the exact command and output root,
2. identify the commit and platform/GPU context,
3. describe the failure mode in plain terms,
4. record the fix or adaptation,
5. rerun the same gate and keep both failure and rerun evidence.

## Closeout rule

Do not declare the migration issue closed until the relevant risk row has:

- a current owner decision,
- a current validation reference,
- and an explicit note that the failure mode was exercised or ruled out.
