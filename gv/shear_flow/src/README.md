# GV Shear-Flow Source Assets

`gv/shear_flow/src/` contains the shear-flow-specific runtime source templates and helper scripts. This lane uses the experimental mirheoOBMD runtime surface; shared GV runtime planning, contracts, and sampling helpers belong under `src/meso_uq/structures/gv`.

- `copy_and_modify.py` copies and adjusts shear-flow setup folders for conservative-force sweeps.
- `run_all.py` and `run_all_HPC.sh` are experiment-local helpers for submitting or running the generated shear-flow cases.

Generated folders, logs, mesh outputs, trajectories, restart data, and scheduler files are runtime artifacts. Keep them under `_runs/...` or a configured external run root, not in committed source.
