# Platform Policy

The first platform policy slice adds representative checked-in config stubs for:

- Karolina
- Vega
- workstation
- generic Slurm

These stubs are examples, not operator secrets. They must:

- stay free of private absolute paths
- prefer env var placeholders for repo, runs, scratch, account, partition, and QoS settings
- remain additive beside existing scripts and runtime helpers

Karolina policy explicitly forbids `/ceph/hpc/home/eubrieucb` inside checked-in platform config examples. Site operators must provide their own environment-backed values instead.

In addition to YAML examples, `src/meso_uq/platforms/policy.py` now carries immutable policy records for:

- scheduler and GPU syntax expectations,
- canonical runs-root policy,
- forbidden path-prefix expectations,
- expected environment scripts for GV/Mirheo/Korali runtime setup,
- and known-policy lookup helpers for `lookup_platform_policy`.

Validation helpers (currently contract-only, no runtime behavior changes) are available as:

- `meso_uq.platforms.validate_platform_path_policy`.

Compatibility notes:

- the existing platform scripts under `scripts/platforms/` remain the active operator entrypoints
- the new `configs/platforms/` files are validation-oriented policy stubs, not a full migration of historical platform configuration
- generic Slurm exists as a compatibility baseline for future site onboarding without encoding cluster-private filesystem roots
