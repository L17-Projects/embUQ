# EMB Workflow Extraction Plan (Wave 1 Prep)

This document prepares Wave 1 extraction work for MES-140, MES-141, and MES-143.
It inventories current EMB workflow surfaces and identifies the smallest safe extraction steps without moving workflow code yet.

## Scope and constraints in this slice

- No broad runtime moves in `compression/`, `indentation/`, `reduced/`, `inference/`, `src/meso_uq/core/**`, or workflow runner scripts.
- Focus on evidence gathering, compatibility risks, and first extraction sequencing.
- Add inventory guardrails so path/entrypoint drift is caught early.

## Evidence-backed inventory

### 1) Legacy generation/runtime-prep surfaces

Shared behavior:
- Both modality generators implement the same parameter sweep loop shape, command generation, and `run_HPC.sbatch` creation.
  - `compression/src/generate.py` lines 29-139
  - `indentation/src/generate.py` lines 24-129
- Both parameter writers support EMB-only runtime prep and generate `parameters*.yaml`, `parameters.prms*.yaml`, and `posq.txt`.
  - `compression/src/parameters.py` lines 41-148
  - `indentation/src/parameters.py` lines 19-131

Divergence:
- Config resolution style differs:
  - compression scans many fallback paths (`compression/src/generate.py` lines 10-27, `compression/src/parameters.py` lines 13-35)
  - indentation primarily uses repo-root absolute resolution in `parameters.py` (`indentation/src/parameters.py` lines 14-17), but fallback scan in `generate.py` (`indentation/src/generate.py` lines 9-21)
- Execution method differs:
  - compression uses `os.system("cd ... && python3 sphere_icosphere.py ...")` (`compression/src/parameters.py` line 76)
  - indentation uses `subprocess.run(..., cwd=...)` (`indentation/src/parameters.py` lines 56-60)
- Numeric behavior diverges in at least one physics parameter:
  - indentation scales `mvert *= 5.0` (`indentation/src/parameters.py` line 70), compression does not.

### 2) Surrogate evaluate flows (DNN/BNN)

Shared behavior:
- DNN evaluators expose single-sample and batch APIs with shape checks and non-negative clipping.
  - compression DNN: `compression/surrogate/evaluate.py` lines 64-202
  - indentation DNN: `indentation/surrogate/evaluate.py` lines 74-210
- BNN evaluators expose parallel APIs returning `(mean, std)` for single and batch modes.
  - compression BNN: `compression/surrogate/evaluate_bnn.py` lines 50-133
  - indentation BNN: `indentation/surrogate/evaluate_bnn.py` lines 52-137

Divergence:
- Artifact name handling differs:
  - compression DNN expects `microbubble_force_BEST.pkl` directly (`compression/surrogate/evaluate.py` lines 32-34)
  - indentation DNN has dual-name fallback (`indentation/surrogate/evaluate.py` lines 33-41)
  - compression BNN candidate set is smaller than indentation BNN candidate set (`evaluate_bnn.py` candidate tuples at lines 12-15 vs 12-17)
- Batch correction semantics differ by modality axis:
  - compression subtracts `d0` from displacement before inference (`compression/surrogate/evaluate.py` line 166)
  - indentation adds `d0` after displacement prediction (`indentation/surrogate/evaluate.py` lines 184-186)

### 3) Train / holdout / BNN / multi-arch flows

Shared behavior:
- `emb_train.py` wrappers for both modalities are thin adapters over shared `meso_uq.surrogate.cli` + `train_tabular_surrogate`.
  - compression: lines 5-33
  - indentation: lines 5-37
- `emb_train_bnn.py` wrappers are thin adapters over shared `train_tabular_bnn_surrogate`.
  - compression: lines 8-77
  - indentation: lines 8-84
- Grouped holdout scripts share the same orchestration skeleton and output contract (`curve_split.csv`, per-curve CSVs, summary JSON, promoted best artifact copy).
  - compression: `run_group_holdout.py` lines 28-141
  - indentation: `run_group_holdout.py` lines 28-158

Divergence:
- Indentation training wrappers carry loader-specific knobs (`--disp-source`, `--rupture-ratio`) not present in compression.
  - `indentation/surrogate/scripts/emb_train.py` lines 19-23
  - `indentation/surrogate/scripts/emb_train_bnn.py` lines 41-54
  - `indentation/surrogate/scripts/run_group_holdout.py` lines 39-66
- Multi-arch implementations are asymmetrical:
  - compression multi-arch uses shared CLI loader/trainer (`compression/.../train_multi_arch.py` lines 21, 44-71)
  - indentation multi-arch has custom data cleaning + custom torch loop in script (`indentation/.../train_multi_arch.py` lines 68-206)

### 4) Validation-matrix and wrapper references

- Public docs define `run_validation_matrix.py` as validation entrypoint and `run_workflow_matrix.py` as delegated lower-level runner:
  - `docs/VEGA_VALIDATION_MATRIX.md` lines 35-45
- `run_validation_matrix.py` is a compatibility wrapper, hard-pinning profile `validation` and delegating to `run_workflow_matrix.py`:
  - `scripts/platforms/vega/run_validation_matrix.py` lines 72-91
- `run_workflow_matrix.py` defines the concrete per-selection command sequence and artifact layout used by matrix validation:
  - command assembly: lines 221-333
  - matrix output root/report setup: lines 454-500
- Additional Vega wrappers are thin pass-through shims to Karolina runners:
  - `run_dnn_rebaseline_matrix.py`, `run_bnn_sweep_matrix.py`, `run_bnn_roundtrip_check.py`, `promote_certified_bnn.py` lines 10-14 in each file
- Workflows doc explicitly marks these wrappers as compatibility surfaces:
  - `docs/WORKFLOWS.md` lines 254-274

## Legacy wrapper candidates and compatibility risks

### A) Current-working-directory (CWD) assumptions

Risk:
- Evalkit root/config discovery still probes `cwd`, parent, grandparent and relative config paths.
  - compression: `compression/evalkit/posterior_compression.py` lines 65-93 and 252-259
  - indentation: `indentation/evalkit/posterior_indentation.py` lines 61-85
- Legacy runtime `compute_compression()` writes beneath project-root-relative folders discovered from cwd (`_out/...`, `_init_...`).
  - `compression/evalkit/posterior_compression.py` lines 280-286

Compatibility implication:
- Moving scripts/modules without a stable path resolver contract can silently break runtime discovery in downstream wrapper usage.

### B) Output-root conventions (old and current)

Risk:
- Legacy compression runtime still uses `_out/compression_<d>um` and `_init_compression_<d>um`.
  - `compression/evalkit/posterior_compression.py` lines 280-286
- Validation matrix and modern wrappers enforce canonical `_runs/...` (or `paper_data/...`) roots.
  - docs policy: `docs/VEGA_VALIDATION_MATRIX.md` lines 103-108
  - wrapper enforcement path: `scripts/platforms/vega/run_validation_matrix.py` lines 61-70 and `run_workflow_matrix.py` lines 454-459

Compatibility implication:
- Extraction must keep old roots functional behind compatibility entrypoints until all callers move to canonical roots.

### C) Serialized surrogate artifact module-path compatibility

Risk:
- Existing pickles may contain legacy module references (`learning.model`).
- Compatibility aliasing is currently provided at load time:
  - `src/meso_uq/surrogate/model.py` lines 35-47
  - compatibility test: `tests/unit/test_surrogate_pickle_compat.py` lines 8-40

Compatibility implication:
- Any extraction that changes model class import paths must preserve this alias bridge or expand it before artifact migration.

### D) Validation matrix dependency chain risk

Risk:
- `run_validation_matrix.py` → `run_workflow_matrix.py` delegation and Vega→Karolina shim wrappers are still asserted by tests and docs.
  - docs: `docs/WORKFLOWS.md` lines 270-274
  - wrappers: vega shim files lines 10-14

Compatibility implication:
- Do not remove/rename compatibility wrappers in first extraction slices; keep command shape stable until migration window closes.

## Existing test coverage anchors for this surface

Key tests currently guarding the inventory:
- generation/parameters:
  - `tests/test_compression_generate.py`
  - `tests/test_indentation_generate_runtime.py`
  - `tests/test_compression_parameters_runtime.py`
  - `tests/test_indentation_parameters_runtime.py`
- train/BNN/loader contracts:
  - `tests/unit/test_compression_emb_train_script.py`
  - `tests/unit/test_compression_emb_train_bnn_script.py`
  - `tests/unit/test_indentation_emb_train_script.py`
  - `tests/unit/test_indentation_emb_train_bnn_script.py`
- evaluate and holdout orchestration:
  - `tests/test_surrogate_evaluators.py`
  - `tests/test_surrogate_group_holdout_orchestration.py`
- matrix/wrapper compatibility:
  - `tests/test_validation_matrix.py`
  - `tests/test_vega_matrix_sbatch.py`

## Minimal safe first extraction PR (after core contracts land)

Recommended first PR boundary:

1. Introduce a small EMB workflow contract module in `src/meso_uq/` (pure constants + path/spec resolvers only), no behavior changes yet.
2. Start with non-invasive consumers only:
   - inventory tests and docs references,
   - optionally loader-independent scripts that already call shared helpers (`emb_train.py`, `emb_train_bnn.py`, `run_group_holdout.py`) using imports/constant lookups only.
3. Keep multi-arch logic and legacy runtime-prep (`generate.py`, `parameters.py`, evalkit `compute_*`) untouched in this first PR.
4. Preserve all existing CLI flags and default artifact names.

Rationale:
- This isolates contract introduction from runtime behavior changes, minimizing break risk while enabling follow-up extraction PRs to move logic incrementally.

MES-140 implementation note:
- `src/meso_uq/agents/emb/workflows.py` now records the EMB compression/indentation generation contracts, config-resolution candidates, parameter-file naming, legacy script identities, and modality-specific runtime-prep constants such as indentation's mass multiplier.
- `src/meso_uq/simulation/emb_generation.py` now owns the shared parameter-sweep expansion, generated `parameters-default*.yaml` grid, `commands.txt` writing, and legacy `run_HPC.sbatch` text.
- `compression/src/generate.py` and `indentation/src/generate.py` remain compatibility entry points with the same CLI flags while delegating the duplicated generation loop to the package helper.
- `compression/src/parameters.py` and `indentation/src/parameters.py` still own the heavy runtime-preparation physics and mesh generation; their extraction remains a follow-up because it depends on preserving the documented compression/indentation numeric differences.

MES-141 implementation note:
- `src/meso_uq/surrogate/emb_workflows.py` now records the EMB compression/indentation surrogate workflow contracts for deterministic NN and BNN wrappers, checkpoint metadata, dataset split metadata, backend resolution, and grouped holdout orchestration.
- `compression/surrogate/scripts/emb_train.py`, `indentation/surrogate/scripts/emb_train.py`, `compression/surrogate/scripts/emb_train_bnn.py`, `indentation/surrogate/scripts/emb_train_bnn.py`, and both `run_group_holdout.py` wrappers remain callable compatibility entry points while delegating shared CLI/parser/orchestration behavior.
- `compression/surrogate/evaluate.py` and `indentation/surrogate/evaluate.py` remain untouched in this slice because evaluator extraction has higher coupling to serialized artifact names and runtime prediction semantics.

## Recommended next extraction issue order

1. MES-140: land shared EMB workflow contracts (path/spec constants, naming, modality metadata) + compatibility tests.
2. MES-141: migrate thin train/BNN/holdout wrappers to consume the shared contract module without changing CLI behavior.
3. MES-143: migrate evaluator/runtime path resolution to shared resolvers behind compatibility wrappers; keep legacy aliases/output-root behavior until validation-matrix parity is proven.

## Out of scope in this slice

- No movement of runtime simulation code from `compression/src` or `indentation/src`.
- No wrapper deletions.
- No GPU matrix execution; integrator still must run target-environment evidence flows.
