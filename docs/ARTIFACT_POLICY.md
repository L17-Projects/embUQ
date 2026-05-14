# Artifact Policy

Generated artifacts are treated as runtime/state and must not be source-controlled. Artifact manifests are the reproducibility contract for anything that must survive a run.

The example manifest lives at `configs/artifacts/artifact_manifest.example.json` and carries `schema_version`, `manifest_id`, `generated_at`, `cleanup_policy`, and `artifacts`.

## Generated-root policy (not tracked)

- `_out`
- `_runs`
- `_init_compression_*`
- `_init_indentation_*`
- `out_hierarchical`
- `_ci`
- `logs`
- repo-root `runtime`
- root Slurm logs `mesouq-*.out`, `mesouq-*.err`
- root Slurm logs `slurm-*.out`, `slurm-*.err`
- `.coverage`, `.pytest_cache`, `__pycache__`, `*.py[cod]`, `*.egg-info`
- `build`, `dist`

## Manifest artifact classes

The initial policy supports these `artifact_class` values:

- `raw`
- `reference`
- `processed`
- `generated`
- `simulation_output`
- `surrogate`
- `surrogate_checkpoint`
- `posterior`
- `posterior_sample`
- `config`
- `figure`
- `report`
- `log`
- `metadata`
- `training_manifest`
- `runtime_manifest`
- `run_manifest`

Manifest records also support policy-side metadata primitives:

- `retention_policy` (`curated` or `generated`)
- `storage_location` (`source_tree`, `generated_root`, `hpc_output`, `external`)
- `release_critical` (boolean override that allows curated exceptions)
- `checksum` metadata (`algorithm`, `value`)
- `provenance` metadata (`generated_by`, `generated_at_tool`, `platform`, `metadata`)
- `validation_status` (`unknown`, `valid`, `invalid`)

Cleanup policy is conservative:

- reject source-tree generated roots (e.g. `_out`, `_runs`, `_ci`, `out_hierarchical`, `_init_compression_*`, `logs`) for non-release-critical artifacts
- treat manifests as inventory and validation inputs first
- keep cleanup exclusions explicit and machine-readable

## Source vs artifact exceptions (stay tracked)

- package entry points:
  - `src/meso_uq/**/__init__.py`
- lightweight source config/docs that are not runtime artifacts:
  - `configs/artifacts/artifact_manifest.example.json`
  - `configs/platforms/generic_slurm.example.yaml`
  - `docs/ARTIFACT_POLICY.md`

## Governance direction

- Do not broad-delete generated roots under this policy slice.
- Current governance is manifest-first: avoid deleting `run`/`surrogate`/`reference` artifacts unless a later migration slice defines explicit cleanup behavior.
- Keep cleanup exclusions explicit and machine-readable.
- `extern/korali` is vendored source and remains outside generated-artifact cleanup.
