# MesoUQ compatibility surface

This page records the legacy paths, imports, configs, and workflows that must remain
available for one release cycle while the architecture migration settles.

Snapshot date: 2026-05-13.

## Compatibility rules

- Keep `import meso_uq` stable.
- Allow `mesouq` branding at the project and CLI level.
- Preserve documented launchers as shims or wrappers until the replacement docs and
  callers are updated together.
- Keep warnings or deprecation notices visible so the migration is measurable.

## Legacy surfaces to preserve

| Surface | Preserve for one release cycle | Evidence now | Later update source |
|---|---|---|---|
| `import meso_uq` | Package import remains the public API | `tests/test_bootstrap.py`, `tests/test_package_import.py`, `docs/RELEASE_SCOPE.md` | `docs/GETTING_STARTED.md`, `docs/RELEASE_NOTES_v0.1.0.md` |
| `learning.model` | Keep as a deprecated installed-package shim for legacy deterministic surrogate pickles and historical imports | `src/learning/model.py`, `src/meso_uq/surrogate/compat.py`, `tests/unit/test_surrogate_pickle_compat.py`, `tests/test_release_assets_smoke.py` | `meso_uq.surrogate.model`, Phase 6 artifact migration/regeneration issue |
| `scripts/platforms/hpc/run_validation_matrix.py` and site wrappers | Keep a site-neutral dispatcher plus Karolina/Vega shims | `tests/test_validation_matrix.py`, `tests/test_karolina_validation_matrix.py`, `tests/test_vega_operator_helpers.py` | `docs/VALIDATION_MATRIX.md`, `docs/VEGA_VALIDATION_MATRIX.md`, `docs/KAROLINA_FULL_PLATFORM.md` |
| `scripts/platforms/vega/run_validation_matrix.py` | Keep the Vega-facing command shape while it forwards to the shared workflow matrix runner | `tests/test_validation_matrix.py`, `docs/VEGA_VALIDATION_MATRIX.md` | `docs/WORKFLOWS.md`, `scripts/platforms/hpc/run_validation_matrix.py` |
| `scripts/run_vega_acceptance.py` and `scripts/platforms/vega/run_validation_suite.py` | Keep the Vega acceptance entrypoint and its richer runner | `tests/test_vega_acceptance_smoke.py`, `docs/VEGA_ACCEPTANCE_COMMAND.md`, `docs/VEGA_ACCEPTANCE_CHECKLIST.md` | `docs/VEGA_BOOTSTRAP.md`, `docs/VEGA_WORKFLOW_HELPERS.md` |
| `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py` and `ops/huq_emb/run_paper_data_campaign.py` | Keep the legacy paper-data/postprocess path as a compatibility shim | `tests/test_huq_emb_campaign_orchestrator.py`, `tests/test_ops_huq_emb_shim.py`, `docs/WORKFLOWS.md` | `docs/HUQ_EMB_EXACT_FIGURE_REPLAY.md`, `docs/RELEASE_SCOPE.md` |
| `gv/<experiment>/src` and `gv_simulation_files/...` legacy staging roots | Keep the old import-root contract while GV moves to manifest-driven staging | `tests/test_gv_pipeline_governance.py`, `tests/test_gv_runtime_governance.py`, `docs/GV_EXTENSION_CLOSEOUT.md` | `docs/KAROLINA_FULL_PLATFORM.md`, `docs/WORKFLOWS.md` |
| Validation configs under `inference/configs/validation` and `reduced/configs/validation` | Keep the current validation-profile YAMLs intact | `docs/VALIDATION_CONFIGS.md`, `docs/VALIDATION_MATRIX.md` | `docs/WORKFLOWS.md`, `tests/test_validation_matrix.py` |
| Production configs under `inference/configs/production` and `reduced/configs/production` | Keep the canonical production configs in place while launchers migrate | `docs/WORKFLOWS.md`, `docs/RELEASE_SCOPE.md` | `docs/VEGA_ACCEPTANCE_CHECKLIST.md`, `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md` |
| `examples/configs` | Keep only while refresh/archive decisions are being made | `docs/RELEASE_SCOPE.md`, `docs/GETTING_STARTED.md` | `docs/README.md`, `docs/RELEASE_NOTES_v0.1.0.md` |
| `emb/compression/src/generate.py` and `emb/indentation/src/generate.py` | Keep as EMB simulation-generation shims while callers migrate to package APIs | `src/meso_uq/simulation/emb_generation.py`, `tests/unit/test_compression_static_geometry.py`, `tests/unit/test_compute_indentation.py` | `meso_uq.simulation.generate_emb_simulation`, `docs/EMB_WORKFLOW_EXTRACTION_PLAN.md` |
| `emb/compression/evalkit/posterior_compression.py` and `emb/indentation/evalkit/posterior_indentation.py` | Keep import and function names stable for Korali, notebooks, and serialized workflow references | `tests/unit/test_evalkit_surrogate_runtime.py`, `tests/unit/test_evalkit_surrogate_import_resolution.py`, `tests/unit/test_surrogate_backend_switch.py` | `src/meso_uq/workflows/legacy.py`, future inference/surrogate package APIs |
| `inference/scripts/run_phase_1.py`, `run_phase_2.py`, and `run_phase_3b.py` | Keep phase launchers callable while runtime plumbing moves behind package contracts | `tests/unit/test_phase1_burn_in_config.py`, `tests/unit/test_inference_phase3b_driver.py`, `tests/unit/test_inference_phase3b_runtime.py` | `meso_uq.workflow_acceleration`, `meso_uq.experiments`, future orchestration package |
| `reduced/scripts/run_phase_1.py`, `run_phase_2.py`, and `run_phase_3b.py` | Keep reduced-model launchers as thin delegates to inference phase launchers | `tests/test_reduced_phase_wrappers.py` | central config aliases and future orchestration package |
| `propagation/scripts/run_phase1_propagation.py` and `run_phase3b_propagation.py` | Keep propagation launchers callable while shared path/config/backend mechanics move to package code | `tests/test_script_manifests.py`, `src/meso_uq/postprocess/propagation.py`, `src/meso_uq/workflows/legacy.py` | `meso_uq.postprocess.propagation`, future reporting/orchestration package |

## Compatibility inventory source

`src/meso_uq/workflows/legacy.py` is the machine-readable inventory for the legacy compression, indentation, inference, reduced, and propagation workflow surfaces. New compatibility shims must be added there with:

- legacy path;
- replacement API;
- purpose;
- workflow family.

The same module owns compatibility-only project-root probing, inference-config fallback resolution, surrogate runtime parsing, evalkit path setup, surrogate trained-directory paths, and once-per-surface warning text. Scientific behavior should remain in the existing workflow modules until a dedicated migration issue moves it behind a replacement package API.

Serialized surrogate artifact compatibility is tracked separately in `src/meso_uq/surrogate/compat.py`. That manifest is intentionally import-light and is exposed through `meso_uq.public_api` so restructuring work can check legacy pickle class paths without importing Torch. The only installed top-level compatibility package is `learning`, and it should remain a warning-emitting shim rather than a place for new implementation.

## Reproducible compatibility inventory commands

These commands were used to count references and confirm symlink targets:

```bash
find scripts -maxdepth 2 -type l | sort
readlink scripts/ci scripts/vega scripts/karolina scripts/hpc
rg -n "^(from|import)\s+compression\." src tests scripts docs .github examples --glob '!venv/**'
rg -n "^(from|import)\s+indentation\." src tests scripts docs .github examples --glob '!venv/**'
rg -n "^(from|import)\s+inference\.scripts\." src tests scripts docs .github examples --glob '!venv/**'
rg -n "^(from|import)\s+reduced\." src tests scripts docs .github examples --glob '!venv/**'
rg -n "^(from|import)\s+propagation\." src tests scripts docs .github examples --glob '!venv/**'
rg -n "^(from|import)\s+meso_uq\.surrogate" src tests scripts docs .github examples --glob '!venv/**'
rg -n "inference/configs/|reduced/configs/|examples/configs/" docs tests .github scripts examples --glob '!venv/**'
rg -n "HUQ_INFERENCE_CONFIG|CONFIG_PATH|MESOUQ_RUNS_ROOT|HPC_SITE|MESOUQ_SITE" src tests docs scripts .github examples --glob '!venv/**'
rg -n "sys\.modules\.setdefault\(\"learning|learning\.model\"|_install_legacy_pickle_aliases" src/meso_uq/surrogate/model.py
find scripts -name '*.sbatch' -print0 | xargs -0 rg -n "python\s+([A-Za-z0-9_./-]+\.py|[A-Za-z0-9_./-]+)"
```

## MES-129 compatibility matrix

| Surface | Current reference count and evidence | Replacement path or API | Owner or status | Shim plan | Warning plan | Removal release | Risk |
|---|---|---|---|---|---|---|---|
| `emb.compression.*` imports | 9 refs / 8 files, mostly tests and evaluation runtime coverage | No complete replacement yet; route future reusable calls to package APIs under `meso_uq` and shared EMB extraction modules | Legacy EMB modality boundary, active | Keep tolerated import compatibility while runtime extraction proceeds | Add deprecation warning only when a canonical package API exists, not before | One full release after canonical APIs and wrapper tests land | Medium |
| `emb.indentation.*` imports | 9 refs / 6 files, mostly tests and evalkit runtime coverage | Same as compression | Legacy EMB modality boundary, active | Same as compression | Same as compression | Same as compression | Medium |
| `inference.scripts.*` Python imports | 0 static Python imports found | None required | Already absent as import surface | No shim needed for Python import form | No warning needed | Not applicable | Low |
| `reduced.*` Python imports | 0 static Python imports found | None required | Already absent as import surface | No shim needed for Python import form | No warning needed | Not applicable | Low |
| `propagation.*` Python imports | 0 static Python imports found | None required | Already absent as import surface | No shim needed for Python import form | No warning needed | Not applicable | Low |
| `meso_uq.surrogate.*` imports | 28 refs / 18 files in scripts, tests, and workflow helpers | Canonical package API surface | Active canonical API | Keep stable and covered by public import tests | Treat breaking changes as API changes; no deprecation unless replacement exists | Not applicable | Low |
| Direct `inference/scripts/*` paths | 11 refs / 6 files in docs, CI canaries, tests, and acceptance examples | Prefer workflow runners under `scripts/platforms/*` where possible | Active legacy script surface | Keep files and command shape; add wrappers if implementation moves | Document replacements before emitting runtime warnings | One full release after docs/tests use replacement paths | High |
| Direct `reduced/scripts/*` paths | 3 refs / 1 docs file | Prefer shared workflow runners when reduced phases move | Active low-volume legacy script surface | Keep wrappers and explicit docs while reduced workflow remains public | Document replacement commands in workflow docs | One full release after replacement docs land | Medium |
| Direct `propagation/scripts/*` paths | 8 refs / 5 files in docs and CI backend canary | Prefer package/reporting APIs when propagation moves | Active legacy script surface | Keep scripts until propagation/reporting boundary is extracted | Document replacement path and keep CI canary reachable | One full release after replacement path validates | High |
| `scripts/ci`, `scripts/vega`, `scripts/karolina`, `scripts/hpc` symlinks | `scripts/ci -> qa/ci`, `scripts/vega -> platforms/vega`, `scripts/karolina -> platforms/karolina`, `scripts/hpc -> platforms/hpc` | Canonical paths are `scripts/qa/ci` and `scripts/platforms/*` | Compatibility-only aliases | Keep symlinks during migration and assert they resolve | Docs should label aliases as compatibility paths | One full release after canonical references replace alias references | High |
| `scripts/platforms/vega` runner surface | High-reference compatibility surface in docs, tests, CI, and wrappers | `scripts/platforms/hpc` dispatcher with explicit `HPC_SITE` where possible | Active compatibility runner | Keep wrappers delegating to shared workflow code | Existing warnings stay visible; expand only where logs remain parseable | Follow-up release after green compatibility matrix | High |
| `scripts/platforms/karolina` runner surface | Active references in platform docs, tests, and launch templates | Shared `scripts/platforms/hpc` dispatcher where possible | Active site runner | Keep Karolina-specific sbatch and env bootstrap while shared runner grows | Mark replacement paths in platform docs before runtime warnings | Follow-up release after Karolina matrix parity evidence | Medium |
| `scripts/platforms/hpc` runner surface | Canonical shared runner used by matrix dispatch and site wrappers | Canonical | Active canonical surface | Keep as primary dispatcher | No deprecation warning | Not applicable | Low |
| `inference/configs/*` | 26 refs / 13 files in docs, tests, and scripts | Keep current tree until Phase 3 central config migration | Active legacy/canonical mixed config surface | Preserve resolver and env override support | Warn only for deprecated implicit defaults after replacement schema exists | Not before Phase 3 migration and one release cycle | Medium |
| `reduced/configs/*` | 18 refs / 8 files | Keep current tree until Phase 3 central config migration | Active legacy/canonical mixed config surface | Preserve wrapper defaults and validation configs | Same as inference configs | Not before Phase 3 migration and one release cycle | Medium |
| `examples/configs` | Example configs present; no direct filename references in the compatibility scan | Refresh under new taxonomy or archive as historical examples | Needs Phase 3 owner decision | Keep until examples refresh/archive issue closes | Docs should say examples are not durable source of truth | After examples are refreshed or archived | Medium |
| Serialized artifact aliases `learning`, `learning.model` | Explicit shim in `src/meso_uq/surrogate/model.py`; covered by `tests/unit/test_surrogate_pickle_compat.py` | Canonical loader remains `meso_uq.surrogate.model` | Active critical serialization shim | Keep `_install_legacy_pickle_aliases()` in load path | Keep regression tests; avoid noisy warnings during model loading | Remove only with staged artifact migration and owner-approved compatibility pack | High |
| Env and CLI path layer | `HUQ_INFERENCE_CONFIG`, `CONFIG_PATH`, `MESOUQ_RUNS_ROOT`, `HPC_SITE`, `MESOUQ_SITE`, and common flags such as `--config`, `--output-root`, `--selection` remain widely referenced | `meso_uq.config.loader`, `meso_uq.hpc_paths`, and platform dispatcher policy | Active runtime/path governance surface | Preserve precedence and normalize output roots | Keep warnings for non-canonical roots where they do not corrupt machine-readable logs | Dedicated migration PR plus one release cycle after policy docs update | High |

## High-risk compatibility surfaces

1. Symlinked launch aliases and Vega wrappers are high risk because CI, docs, and platform workflows still reference them.
2. Serialized model aliasing is high risk because old pickles may fail to load if `learning.model` compatibility is removed.
3. Environment and CLI path controls are high risk because they silently change config and output roots.
4. Direct `inference/scripts`, `reduced/scripts`, and `propagation/scripts` calls are high risk where CI canaries or operational docs invoke literal paths.
5. Slurm templates are high risk because they encode both script paths and cluster environment assumptions.

## Update triggers

Update this page when any of the following happen together:

- the launcher surface moves from compatibility wrapper to canonical site-neutral entrypoint
- the GV staging model changes from legacy roots to manifest-only delivery
- the validation configs are renamed, archived, or regenerated
- the paper replay flow stops using the compatibility shim
- a legacy workflow path starts delegating to `src/meso_uq/workflows/legacy.py` or to a new package API

## Removal rule

Do not remove a compatibility surface until:

1. the new primary path is documented,
2. the old path is covered by a deprecation notice or warning,
3. the relevant tests are green,
4. one full release cycle has passed.
