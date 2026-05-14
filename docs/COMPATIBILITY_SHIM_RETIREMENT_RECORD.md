# Compatibility shim retirement record

MES-166 records whether MesoUQ compatibility shims can be removed after one
release cycle. This checkout does not contain evidence that a full release cycle
has elapsed after the architecture-migration compatibility inventory was added.
The retained status below is therefore intentional: the shims remain in place
until the release-cycle blocker is cleared by release evidence and green tests.

Review baseline: 2026-05-14.

## Retirement policy

- Do not remove a compatibility surface until the replacement path is documented,
  the legacy path emits or carries a deprecation/compatibility notice where
  practical, the relevant tests are green, and one full release cycle has passed.
- Treat `docs/MESO_UQ_COMPATIBILITY_SURFACE.md` as the human-readable inventory.
- Treat `src/meso_uq/workflows/legacy.py`,
  `src/meso_uq/surrogate/compat.py`, and
  `src/meso_uq/config/aliases.py` as the machine-readable inventories.
- Keep paper-specific and generated cleanup records out of this retirement
  record.

## Retained package and artifact shims

| Surface | Replacement path | Evidence | Reason retained | Retirement blocker | Review trigger/date |
|---|---|---|---|---|---|
| `import meso_uq` | N/A; remains the package API | `tests/test_bootstrap.py`, `tests/test_package_import.py`, `docs/RELEASE_SCOPE.md` | Public package import is not a retirement candidate. | None; keep stable. | Every release readiness review. |
| `learning` | `meso_uq.surrogate` | `src/learning/__init__.py`, `tests/unit/test_surrogate_pickle_compat.py` | Top-level package shim supports historical surrogate imports and pickle module resolution. | Release-critical pickles still need the alias path to remain loadable. | Review after the first release following 2026-05-14 and after artifact migration/regeneration evidence lands. |
| `learning.model.MLP` | `meso_uq.surrogate.model.MLP` | `src/learning/model.py`, `src/meso_uq/surrogate/compat.py`, `tests/unit/test_surrogate_pickle_compat.py`, `tests/test_release_assets_smoke.py` | Older deterministic EMB surrogate pickles can reference `learning.model.MLP`. | No evidence yet that all release-critical pickles were migrated or regenerated and validated for one release cycle. | Review after the first release following 2026-05-14 and after `SURROGATE_SERIALIZATION_ALIASES` no longer lists the legacy class path. |

## Retained workflow shims

| Surface | Replacement path | Evidence | Reason retained | Retirement blocker | Review trigger/date |
|---|---|---|---|---|---|
| `scripts/platforms/hpc/run_validation_matrix.py` | Shared workflow matrix runner | `tests/test_validation_matrix.py`, `docs/VALIDATION_MATRIX.md` | Site-neutral dispatcher remains part of the documented validation command shape. | Replacement command and callers have not been proven through one full release cycle. | Review after validation matrix docs and callers point only at the replacement for one release. |
| `scripts/platforms/vega/run_validation_matrix.py` | `scripts/platforms/hpc/run_validation_matrix.py` and shared workflow matrix runner | `tests/test_validation_matrix.py`, `docs/VEGA_VALIDATION_MATRIX.md` | Vega-facing command shape is still documented for operators. | Vega docs and launchers still carry the compatibility entrypoint. | Review after the first release following 2026-05-14 with green Vega matrix evidence. |
| `scripts/run_vega_acceptance.py` | `scripts/platforms/vega/run_validation_suite.py` and documented Vega bootstrap flow | `tests/test_vega_acceptance_smoke.py`, `docs/VEGA_ACCEPTANCE_COMMAND.md` | Acceptance command remains a documented operator entrypoint. | Operator docs have not completed a release cycle on the replacement-only path. | Review after `docs/VEGA_ACCEPTANCE_COMMAND.md` and `docs/VEGA_BOOTSTRAP.md` stop referencing the legacy entrypoint for one release. |
| `scripts/platforms/vega/run_validation_suite.py` | Shared validation/orchestration package APIs when available | `tests/test_vega_acceptance_smoke.py`, `docs/VEGA_ACCEPTANCE_CHECKLIST.md` | Runner is still the maintained Vega acceptance implementation behind compatibility commands. | No package-level replacement API has completed release validation. | Review when a package orchestration API replaces the script in docs and tests for one release. |
| `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py` | Future maintained HUQ-EMB workflow package entrypoint | `tests/test_huq_emb_campaign_orchestrator.py`, `docs/WORKFLOWS.md` | Maintained campaign runner is still the target for the legacy ops shim. | Paper-data workflow callers have not moved to a replacement package entrypoint. | Review after the HUQ-EMB campaign has replacement docs and green replay tests for one release. |
| `ops/huq_emb/run_paper_data_campaign.py` | `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py` | `tests/test_ops_huq_emb_shim.py`, `docs/MESO_UQ_COMPATIBILITY_SURFACE.md` | Legacy operations path delegates to the maintained campaign runner. | Ops callers may still use the old path; no release-cycle evidence supports removal. | Review after ops docs and tests stop requiring the shim for one release. |
| `compression/src/generate.py` | `meso_uq.simulation.generate_emb_simulation` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_compression_static_geometry.py`, `tests/unit/test_compute_indentation.py` | Historical EMB compression generation command remains a compatibility entrypoint. | CLI callers and generated-file expectations have not completed the migration window. | Review after generation docs and callers use the package API for one release. |
| `indentation/src/generate.py` | `meso_uq.simulation.generate_emb_simulation` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_compression_static_geometry.py`, `tests/unit/test_compute_indentation.py` | Historical EMB indentation generation command remains a compatibility entrypoint. | CLI callers and generated-file expectations have not completed the migration window. | Review after generation docs and callers use the package API for one release. |
| `compression/evalkit/posterior_compression.py` | `meso_uq.workflows.legacy + compression.evalkit compatibility functions` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_evalkit_surrogate_runtime.py`, `tests/unit/test_evalkit_surrogate_import_resolution.py` | Korali, notebooks, and serialized workflow references may import the legacy evalkit names. | No replacement evalkit API has completed release validation. | Review after evalkit imports and workflow references migrate for one release. |
| `indentation/evalkit/posterior_indentation.py` | `meso_uq.workflows.legacy + indentation.evalkit compatibility functions` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_evalkit_surrogate_runtime.py`, `tests/unit/test_evalkit_surrogate_import_resolution.py` | Korali, notebooks, and serialized workflow references may import the legacy evalkit names. | No replacement evalkit API has completed release validation. | Review after evalkit imports and workflow references migrate for one release. |
| `inference/scripts/run_phase_1.py` | `meso_uq.workflow_acceleration` and `meso_uq.experiments` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_phase1_burn_in_config.py` | Phase launcher remains the documented/callable inference entrypoint while orchestration moves into package code. | Package orchestration replacement has not completed one release cycle. | Review after phase-1 docs and callers use package orchestration for one release. |
| `inference/scripts/run_phase_2.py` | `meso_uq.workflow_acceleration` and `meso_uq.experiments` | `src/meso_uq/workflows/legacy.py`, phase runtime tests | Phase launcher remains callable while runtime plumbing migrates. | Package orchestration replacement has not completed one release cycle. | Review after phase-2 docs and callers use package orchestration for one release. |
| `inference/scripts/run_phase_3b.py` | `meso_uq.workflow_acceleration` and `meso_uq.experiments` | `src/meso_uq/workflows/legacy.py`, `tests/unit/test_inference_phase3b_driver.py`, `tests/unit/test_inference_phase3b_runtime.py` | Phase launcher remains callable while runtime plumbing migrates. | Package orchestration replacement has not completed one release cycle. | Review after phase-3b docs and callers use package orchestration for one release. |
| `reduced/scripts/run_phase_1.py` | `inference/scripts/run_phase_1.py` with reduced config, then future orchestration API | `src/meso_uq/workflows/legacy.py`, `tests/test_reduced_phase_wrappers.py` | Reduced wrapper keeps legacy reduced-model command shape. | Reduced callers still rely on wrapper forwarding and config defaults. | Review after reduced phase-1 callers use the replacement for one release. |
| `reduced/scripts/run_phase_2.py` | `inference/scripts/run_phase_2.py` with reduced config, then future orchestration API | `src/meso_uq/workflows/legacy.py`, `tests/test_reduced_phase_wrappers.py` | Reduced wrapper keeps legacy reduced-model command shape. | Reduced callers still rely on wrapper forwarding and config defaults. | Review after reduced phase-2 callers use the replacement for one release. |
| `reduced/scripts/run_phase_3b.py` | `inference/scripts/run_phase_3b.py` with reduced config, then future orchestration API | `src/meso_uq/workflows/legacy.py`, `tests/test_reduced_phase_wrappers.py` | Reduced wrapper keeps legacy reduced-model command shape. | Reduced callers still rely on wrapper forwarding and config defaults. | Review after reduced phase-3b callers use the replacement for one release. |
| `propagation/scripts/run_phase1_propagation.py` | `meso_uq.postprocess.propagation` | `src/meso_uq/workflows/legacy.py`, `tests/test_script_manifests.py`, `src/meso_uq/postprocess/propagation.py` | Propagation launcher remains callable while shared postprocess APIs stabilize. | Replacement API has not been the only documented path for one release. | Review after propagation phase-1 docs and callers use package APIs for one release. |
| `propagation/scripts/run_phase3b_propagation.py` | `meso_uq.postprocess.propagation` | `src/meso_uq/workflows/legacy.py`, `tests/test_script_manifests.py`, `src/meso_uq/postprocess/propagation.py` | Propagation launcher remains callable while shared postprocess APIs stabilize. | Replacement API has not been the only documented path for one release. | Review after propagation phase-3b docs and callers use package APIs for one release. |

## Retained config roots and aliases

| Surface | Replacement path | Evidence | Reason retained | Retirement blocker | Review trigger/date |
|---|---|---|---|---|---|
| `inference/configs/production` | Current canonical root until config APIs replace it | `src/meso_uq/config/aliases.py`, `tests/unit/test_config_aliases.py`, `docs/WORKFLOWS.md` | Production inference configs are still canonical files and legacy-alias records. | No replacement config API and release-cycle evidence for renamed roots. | Review after config docs and launchers use a replacement API for one release. |
| `inference/configs/validation` | Current canonical root until config APIs replace it | `src/meso_uq/config/aliases.py`, `tests/unit/test_config_aliases.py`, `docs/VALIDATION_CONFIGS.md` | Validation configs are still documented and tested in place. | Validation matrix still expects these roots. | Review after validation config docs and tests use a replacement API for one release. |
| `reduced/configs/production` | Current canonical root until config APIs replace it | `src/meso_uq/config/aliases.py`, `tests/unit/test_config_aliases.py`, `docs/WORKFLOWS.md` | Reduced production configs are still documented and tested in place. | Reduced launchers still default to these configs. | Review after reduced launchers and docs use a replacement API for one release. |
| `reduced/configs/validation` | Current canonical root until config APIs replace it | `src/meso_uq/config/aliases.py`, `tests/unit/test_config_aliases.py`, `docs/VALIDATION_CONFIGS.md` | Reduced validation configs are still documented and tested in place. | Validation matrix still expects these roots. | Review after reduced validation docs and tests use a replacement API for one release. |
| `examples/configs` | Refresh, archive, or replacement examples location when decided | `src/meso_uq/config/aliases.py`, `tests/unit/test_config_aliases.py`, `docs/RELEASE_SCOPE.md` | Example config refresh/archive decision is still pending. | No recorded final decision or one-release validation for a replacement location. | Review when examples config policy is finalized; no earlier than the first release after 2026-05-14. |
| `HUQ_INFERENCE_CONFIG` and `CONFIG_PATH` override behavior | Future config loader/orchestration API | `src/meso_uq/workflows/legacy.py`, `src/meso_uq/config/loader.py`, `tests/unit/test_config_loader.py`, `tests/unit/test_evalkit_surrogate_runtime.py` | Environment overrides remain part of runtime compatibility for legacy phase/evalkit callers. | Callers and batch templates still use the override behavior. | Review after launchers stop depending on these overrides for one release. |

## Retained data/staging surfaces

| Surface | Replacement path | Evidence | Reason retained | Retirement blocker | Review trigger/date |
|---|---|---|---|---|---|
| `gv/<experiment>/src` | Manifest-driven GV staging | `docs/MESO_UQ_COMPATIBILITY_SURFACE.md`, `tests/test_gv_pipeline_governance.py`, `tests/test_gv_runtime_governance.py` | Legacy import roots bridge existing GV provenance and runtime staging. | Manifest-only staging has not completed one release cycle. | Review after GV staging docs and tests require manifest-only delivery for one release. |
| `gv_simulation_files/...` | Manifest-driven GV staging | `docs/MESO_UQ_COMPATIBILITY_SURFACE.md`, `tests/test_gv_pipeline_governance.py`, `tests/test_gv_runtime_governance.py` | Legacy staging roots are still part of the documented compatibility surface. | Manifest-only staging has not completed one release cycle. | Review after GV staging docs and tests require manifest-only delivery for one release. |

## Closeout decision

No compatibility shim is approved for removal in this slice. All listed surfaces
remain retained because the repository evidence still describes an active
migration window and does not prove that one full release cycle has passed.
