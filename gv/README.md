# GV Numerical Experiment Assets

`gv/` contains gas-vesicle numerical experiment assets. Shared GV contracts, sampling APIs, runtime planning, postprocessing helpers, and registries live under `src/meso_uq`, especially `src/meso_uq/structures/gv`.

Canonical shape:

```text
gv/<modality>/
├── src/
├── evalkit/
└── surrogate/
```

Current modalities are:

- `buckling`
- `eigenmodes`
- `shear_flow`
- `stretching`
- `torsion`

Use each modality `src/` directory for experiment-specific Mirheo or mirheoOBMD source templates and thin run helpers, `evalkit/` for small modality-specific evaluation fixtures and descriptors, and `surrogate/` for modality-specific surrogate fixtures, manifests, or small artifact descriptors. Put reusable implementation in `src/meso_uq` instead of duplicating it across modality folders.

Generated Mirheo outputs, meshes, logs, restart data, trajectories, force outputs, compiled objects, and scratch artifacts are ignored and must stay under `_runs/...` or configured external run roots. Local `dir.md` placement guides are ignored by `**/dir.md` and are not source.
