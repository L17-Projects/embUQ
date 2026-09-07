# Config Composition

`configs/` is the schema-versioned study/composition layer. It describes how agents, modalities, datasets, surrogates, inference settings, noise models, platforms, reports, and artifacts fit together.

The current examples are additive policy/configuration skeletons:

- `agents/`: agent-family controllers and compatible modality roots.
- `modalities/`: structure and experiment selections such as `emb` plus `compression`.
- `datasets/`: reference or generated dataset descriptors.
- `surrogates/`: surrogate backend and artifact manifest references.
- `inference/`: algorithm and runtime-config template selection.
- `noise/`: observation-noise model examples.
- `platforms/`: workstation, Vega, Karolina, and generic Slurm path/runtime policy examples.
- `reports/` and `artifacts/`: reporting and artifact-manifest examples.

Active EMB production and validation workflow configs still live under `inference/configs/` and `reduced/configs/`. Use `configs/` for structure-general composition examples and policy-facing schema work; update the active runtime config roots when changing the workflows that current operators actually run.

Config paths should use placeholders such as `${MESOUQ_REPO_ROOT}`, `${MESOUQ_RUNS_ROOT}`, `${MESOUQ_SCRATCH_ROOT}`, and `${MESOUQ_SITE_RUNTIME_ROOT}` instead of user-specific absolute paths. Generated config snapshots and run manifests belong in the selected output root, not in this directory.
