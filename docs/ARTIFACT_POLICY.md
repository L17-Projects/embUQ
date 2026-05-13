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
- `surrogate`
- `posterior`
- `config`
- `figure`
- `report`
- `log`
- `metadata`
- `training_manifest`
- `runtime_manifest`
- `run_manifest`

Cleanup policy is conservative:

- do not delete generated roots or artifacts as part of this migration slice
- treat manifests as inventory and validation inputs first
- keep cleanup exclusions explicit and machine-readable

`extern/korali` remains vendored source, not a generated artifact root. It is protected from generated-artifact cleanup and must stay outside any automatic cleanup target set.
