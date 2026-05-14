# MesoUQ migration risk register

This register tracks the risks that matter for the Wave 0 migration closeout.

## Register

Snapshot date: 2026-05-13.

| Risk | Evidence | Likelihood | Impact | Mitigation | Rollback plan | Validation command | Phase blocker | Owner decision needed |
|---|---|---|---|---|---|---|---|---|
| Paper scripts and `papers/` reproduction paths drift | `docs/HUQ_EMB_EXACT_FIGURE_REPLAY.md`; `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py`; `scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py`; paper replay tests | Medium | High | Keep campaign orchestration and exact asset port split explicit; preserve `paper_data` layout contracts; keep hard-fail checks for manifests and workflow gates | Re-run with fixed `--campaign-id` and roots; use `--skip-workflow` or `--skip-asset-graph` to isolate failure; keep previous campaign artifacts until replacement validates | `python scripts/workflows/emb/huq_emb/run_paper_data_campaign.py --paper-data-root paper_data --campaign-id <id> --skip-workflow`; `python scripts/workflows/emb/huq_emb/run_exact_uqdpd_asset_port.py --paper-data-root paper_data --campaign-id <id>`; `python -m pytest tests/test_run_exact_uqdpd_asset_port.py tests/test_gv_paper_replay_*.py` | Phase 2 | Should the legacy replay split remain supported past Phase 2? |
| Karolina, Vega, and workstation platform behavior mismatch | `scripts/platforms/hpc/run_validation_matrix.py`; `scripts/platforms/vega/run_validation_matrix.py`; `scripts/platforms/karolina/run_validation_matrix.py`; `scripts/platforms/workstation/run_local_validation_matrix.py`; `docs/PLATFORM_POLICY.md` | Medium | High | Use the HPC dispatcher as canonical selector where possible; keep compatibility shims explicit; document local/workstation backend expectations | Route recovery through canonical dispatcher with explicit site env; regenerate run roots; compare expected outputs | `python -m pytest tests/test_validation_matrix.py tests/test_karolina_validation_matrix.py tests/test_workstation_validate_local_outputs.py`; GPU matrix through `scripts/platforms/karolina/sbatch/validation_matrix.sbatch` | Phase 2 | Keep direct site scripts as supported shims, or deprecate after one release cycle? |
| `extern/korali` vendoring/provenance risk | `src/meso_uq/site_runtime.py`; `scripts/platforms/vega/bootstrap_korali.sh`; `scripts/platforms/karolina/bootstrap_korali.sh`; `src/meso_uq/artifacts/policy.py`; `src/meso_uq/vega.py` | Low-Medium | High | Keep `extern/korali` protected from artifact cleanup; prioritize site runtime env over ambient env; keep bootstrap-generated environment scripts | Re-bootstrap Korali into the site runtime root, regenerate `env.sh`, and rerun strict imports | `bash scripts/platforms/vega/bootstrap_korali.sh --skip-python-build-deps`; `python -m pytest tests/test_import_boundaries.py tests/unit/test_artifact_manifest_policy.py` | Phase 2 | Can vendored Korali ever be relocated or removed, and under what compatibility window? |
| Mirheo and GV source staging reproducibility | `src/meso_uq/vega.py`; `scripts/platforms/vega/bootstrap_mirheo.sh`; `scripts/platforms/karolina/bootstrap_mirheo.sh`; `src/meso_uq/site_runtime.py`; GV replay/governance tests | Medium | High | Keep lock/env/override precedence and source snapshots; enforce snapshot checks before runtime; move toward manifest-driven staging without dropping legacy bridge early | Re-bootstrap Mirheo from explicit source or lock path; restore snapshot; rerun doctor and GV replay/governance tests | `bash scripts/platforms/vega/bootstrap_mirheo.sh --source <path> --python-bin $(command -v python)`; `python scripts/platforms/vega/doctor_vega.py --with-mirheo --with-gv-runtime --strict`; `python -m pytest tests/test_gv_pipeline_governance.py tests/test_gv_paper_replay_*.py` | Phase 2 / Phase 3 | Approve removal timing for the old GV source compatibility bridge |
| Legacy and central config path ambiguity | `docs/CONFIGURATION_POLICY.md`; `docs/VALIDATION_CONFIGS.md`; `tests/unit/test_config_loader.py`; `tests/unit/test_phase3b_config_bootstrap.py`; `tests/unit/test_evalkit_surrogate_runtime.py` | Medium | Medium-High | Keep additive migration model; preserve explicit precedence for `HUQ_INFERENCE_CONFIG` and `CONFIG_PATH`; keep legacy path compatibility tested | Pin explicit config overrides in workflows until config taxonomy migration is complete | `python -m pytest tests/unit/test_config_loader.py tests/unit/test_phase3b_config_bootstrap.py tests/unit/test_config_policy.py tests/unit/test_evalkit_surrogate_runtime.py` | Phase 2 | Approve hard cutover date and compatibility scope for `inference/configs` and `reduced/configs` |
| BNN/Pyro optional dependency behavior | `pyproject.toml`; `docs/DEPENDENCY_EXTRAS.md`; `src/meso_uq/surrogate/bnn.py`; `tests/test_import_boundaries.py`; `tests/test_optional_dependency_markers.py`; `codecov.yml` | Medium | Medium | Keep BNN optional by contract; avoid importing Pyro paths in default workflows; keep BNN canaries separate from non-BNN baseline | Install BNN extras and rerun BNN matrix; keep non-BNN tests as baseline gate while optional dependencies are absent | `python -m pytest tests/test_import_boundaries.py tests/test_optional_dependency_markers.py`; `pip install -e ".[bnn]" && python -m pytest tests/test_bnn_training_matrix.py tests/test_bnn_certification_matrix.py` | Phase 2 / Phase 3 | Decide when, if ever, BNN support becomes mandatory for acceptance |
| Pickled model compatibility regressions | `src/meso_uq/surrogate/model.py`; `tests/unit/test_surrogate_pickle_compat.py`; release asset smoke tests | Low-Medium | High | Preserve compatibility loader aliases and regression tests before artifact cleanup | Validate legacy pickle fixtures before deleting or moving model artifacts; hold artifact cleanup until pass | `python -m pytest tests/unit/test_surrogate_pickle_compat.py tests/test_release_assets_smoke.py` | Phase 2 | Approve supported legacy pickle formats and removal window |
| CI and Codecov relaxed versus strict gates mismatch | `.github/workflows/ci.yml`; `codecov.yml`; `scripts/qa/ci/check_coverage_increase.py`; `tests/test_coverage_delta_gate.py` | High | Medium | Keep strictness explicit through labels/manual strict mode; document skipped canaries and exceptions in PR closeout | For risky merges, run strict coverage check manually and attach report before merge | `python -m pytest tests/test_coverage_delta_gate.py`; `python scripts/qa/ci/check_coverage_increase.py --base-json <base> --head-json <head> --strict` | Phase 3 | Approve when strict coverage gate becomes mandatory baseline again |
| Artifact movement and owner-approved deletion | `docs/ARTIFACT_POLICY.md`; `docs/MESO_UQ_MIGRATION_INVENTORY.md`; `src/meso_uq/artifacts/policy.py`; `tests/unit/test_artifact_manifest_policy.py`; GV governance tests | Medium | High | Protect vendor/runtime roots; require owner-approved cleanup list, manifest/checksum evidence, and post-action GPU validation | Recreate deleted artifacts through bootstrap/regeneration scripts; restore from archive if validation fails | `python -m pytest tests/unit/test_artifact_manifest_policy.py tests/test_gv_pipeline_governance.py`; GPU validation matrix after cleanup | Phase 3 / Phase 6 | Approve deletion inventory, external storage target, retention window, and rollback archive |

## High-risk gaps

1. Ownership and decision tracking must stay explicit for every risk because migration closeout depends on owner-approved retention or deprecation.
2. Site script support must be labeled by phase: supported as canonical, supported as shim, deprecated, or removed.
3. Optional dependency behavior must remain linked to CI strictness so a relaxed feedback loop does not hide BNN/Pyro regressions.
4. Artifact deletion must include commandable evidence: inventory row, owner approval, cleanup command, validation command, and rollback path.
5. Compatibility shims can mask broken canonical paths, so every removal candidate needs both old-path and new-path tests before one-release-cycle removal begins.

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
