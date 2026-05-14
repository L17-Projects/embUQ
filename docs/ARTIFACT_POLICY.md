# Artifact Policy

Artifact manifests in this slice are intentionally small and machine-readable. The example manifest lives at `configs/artifacts/artifact_manifest.example.json` and carries:

- `schema_version`
- `manifest_id`
- `generated_at`
- `cleanup_policy`
- `artifacts`

Supported `artifact_class` values in the initial policy slice:

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

`extern/korali` remains vendored source, not a generated artifact root. It is protected from generated-artifact cleanup and must stay outside any automatic cleanup target set.
