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

Compatibility notes:

- the existing platform scripts under `scripts/platforms/` remain the active operator entrypoints
- the new `configs/platforms/` files are validation-oriented policy stubs, not a full migration of historical platform configuration
- generic Slurm exists as a compatibility baseline for future site onboarding without encoding cluster-private filesystem roots
