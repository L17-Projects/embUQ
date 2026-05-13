# MesoUQ compatibility surface

This page records the legacy paths, imports, configs, and workflows that must remain
available for one release cycle while the architecture migration settles.

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
| `scripts/platforms/hpc/run_validation_matrix.py` and site wrappers | Keep a site-neutral dispatcher plus Karolina/Vega shims | `tests/test_validation_matrix.py`, `tests/test_karolina_validation_matrix.py`, `tests/test_vega_operator_helpers.py` | `docs/VALIDATION_MATRIX.md`, `docs/VEGA_VALIDATION_MATRIX.md`, `docs/KAROLINA_FULL_PLATFORM.md` |
| `scripts/platforms/vega/run_validation_matrix.py` | Keep the Vega-facing command shape while it forwards to the shared workflow matrix runner | `tests/test_validation_matrix.py`, `docs/VEGA_VALIDATION_MATRIX.md` | `docs/WORKFLOWS.md`, `scripts/platforms/hpc/run_validation_matrix.py` |
| `scripts/run_vega_acceptance.py` and `scripts/platforms/vega/run_validation_suite.py` | Keep the Vega acceptance entrypoint and its richer runner | `tests/test_vega_acceptance_smoke.py`, `docs/VEGA_ACCEPTANCE_COMMAND.md`, `docs/VEGA_ACCEPTANCE_CHECKLIST.md` | `docs/VEGA_BOOTSTRAP.md`, `docs/VEGA_WORKFLOW_HELPERS.md` |
| `scripts/workflows/emb/huq_emb/run_paper_data_campaign.py` and `ops/huq_emb/run_paper_data_campaign.py` | Keep the legacy paper-data/postprocess path as a compatibility shim | `tests/test_huq_emb_campaign_orchestrator.py`, `tests/test_ops_huq_emb_shim.py`, `docs/WORKFLOWS.md` | `docs/HUQ_EMB_EXACT_FIGURE_REPLAY.md`, `docs/RELEASE_SCOPE.md` |
| `gv/<experiment>/src` and `gv_simulation_files/...` legacy staging roots | Keep the old import-root contract while GV moves to manifest-driven staging | `tests/test_gv_pipeline_governance.py`, `tests/test_gv_runtime_governance.py`, `docs/GV_EXTENSION_CLOSEOUT.md` | `docs/KAROLINA_FULL_PLATFORM.md`, `docs/WORKFLOWS.md` |
| Validation configs under `inference/configs/validation` and `reduced/configs/validation` | Keep the current validation-profile YAMLs intact | `docs/VALIDATION_CONFIGS.md`, `docs/VALIDATION_MATRIX.md` | `docs/WORKFLOWS.md`, `tests/test_validation_matrix.py` |
| Production configs under `inference/configs/production` and `reduced/configs/production` | Keep the canonical production configs in place while launchers migrate | `docs/WORKFLOWS.md`, `docs/RELEASE_SCOPE.md` | `docs/VEGA_ACCEPTANCE_CHECKLIST.md`, `docs/VEGA_PHASE2_NATIVE_CUDA_CHECKLIST.md` |
| `examples/configs` | Keep only while refresh/archive decisions are being made | `docs/RELEASE_SCOPE.md`, `docs/GETTING_STARTED.md` | `docs/README.md`, `docs/RELEASE_NOTES_v0.1.0.md` |

## Update triggers

Update this page when any of the following happen together:

- the launcher surface moves from compatibility wrapper to canonical site-neutral entrypoint
- the GV staging model changes from legacy roots to manifest-only delivery
- the validation configs are renamed, archived, or regenerated
- the paper replay flow stops using the compatibility shim

## Removal rule

Do not remove a compatibility surface until:

1. the new primary path is documented,
2. the old path is covered by a deprecation notice or warning,
3. the relevant tests are green,
4. one full release cycle has passed.
