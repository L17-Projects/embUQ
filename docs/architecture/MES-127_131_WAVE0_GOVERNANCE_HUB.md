# Wave 0 governance hub

This page ties together the Wave 0 governance/inventory work for Linear issues
MES-127 through MES-131 and the GPU closeout gate MES-160.

The scope is documentation and evidence mapping only. It does not change source code or
launcher behavior.

## Current policy snapshot

The migration decisions being recorded here are:

- keep the import API as `meso_uq`
- allow project and CLI branding to read `mesouq`
- keep only curated release-critical surrogate artifacts in git, with manifests
- keep raw/reference data only when it is small, license-clear, documented, and
  reproducibility-critical
- move generated training data out of git
- treat `_out`, `_runs`, `_init_compression_*`, `out_hierarchical`, `_ci`, and root Slurm logs
  as non-reproduction state
- keep `extern/korali` vendored
- stage GV through manifests rather than permanent `gv/<experiment>/src` ownership
- support Karolina, Vega, and workstation execution
- refresh or archive `examples/configs`
- keep `papers/` for now, while heavy generated artifacts move out once manifests exist
- keep legacy imports and paths working for one release cycle with shims and warnings

## Linked evidence

- [Migration inventory](../MESO_UQ_MIGRATION_INVENTORY.md)
- [Compatibility surface](../MESO_UQ_COMPATIBILITY_SURFACE.md)
- [Migration risk register](../MESO_UQ_MIGRATION_RISK_REGISTER.md)
- [GPU validation gate](../MESO_UQ_GPU_VALIDATION_GATE.md)
- [Package contracts](../PACKAGE_CONTRACTS.md)

## Repo evidence anchors

- `docs/RELEASE_SCOPE.md`
- `docs/WORKFLOWS.md`
- `docs/VALIDATION_MATRIX.md`
- `docs/SUPPORT_MATRIX.md`
- `docs/KAROLINA_FULL_PLATFORM.md`
- `docs/VEGA_VALIDATION_MATRIX.md`
- `tests/test_gv_pipeline_governance.py`
- `tests/test_gv_runtime_governance.py`
- `tests/test_validation_matrix.py`
- `tests/test_karolina_validation_matrix.py`
