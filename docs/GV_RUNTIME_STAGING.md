# GV Runtime Staging (MES-161)

This contract stages small GV runtime template files from a configured Mirheo source tree into a per-run output tree. It is staging-only and dry-run safe: it does not execute Mirheo.

## Scope

- Staging modality: `gv`
- Supported experiments: `stretching`, `buckling`, `torsion`, `eigenmodes`, `shear_flow`
- Output layout: `<output_root>/gv/<experiment>/<run_id>/...`

## Source root policy

The Mirheo source root must be explicit:

1. pass `mirheo_source_root` directly, or
2. set `MESOUQ_GV_MIRHEO_SOURCE_ROOT`.

No platform-specific default path is assumed. This keeps behavior portable across Karolina, Vega, and workstation runs.

## Safety guards

- Staging rejects output roots under repository source trees (`src/`, `tests/`, `gv/`, `gv_simulation_files/`, `scripts/`).
- Staging rejects existing non-empty `output_root` unless `allow_existing_nonempty_output=True`.
- Staging validates each template file exists.
- Staging validates staged destination paths stay under the run root.
- Run directory reservation uses unique IDs with collision-safe `mkdir(exist_ok=False)` semantics.

## Dry-run behavior

- `dry_run=True` returns a manifest/plan and file hashes/sizes without copying files.
- `dry_run=False` copies templates into the run directory.
- Neither mode runs Mirheo; execution is intentionally out of scope for this contract.

## Repository hygiene

Staged runtime trees are operational artifacts and should remain outside committed source trees. Keep staging outputs in external or ignored run/output roots. Do not commit staged runtime directories.
