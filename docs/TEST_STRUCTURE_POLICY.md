# Test Structure Policy

MES-152 defines the test-suite layout for the scalable architecture migration. The policy is to keep fast package feedback as the default while making integration, optional-runtime, platform, and GPU/HPC checks explicit.

## Layout

- `tests/unit/`: pure package logic, config models, registries, policy helpers, and small deterministic fixtures.
- `tests/integration/`: cross-module behavior, workflow assembly, GV staging acceptance, and small end-to-end smoke paths.
- `tests/support/`: stdlib-only helpers shared by tests, including optional dependency and governance checks.
- `tests/test_*.py`: repository-level smoke, regression, compatibility, documentation, CI, governance, and workflow tests that have not yet become narrower unit or integration tests.

Do not place generated outputs, scheduler logs, checkpoints, posterior samples, or large data under `tests/`. Test fixtures must be small, documented, and covered by the artifact policy.

## Marker Contract

The repository declares these pytest markers in `pyproject.toml`:

- `integration`
- `operational`
- `slow`
- `gpu`
- `cuda`
- `hpc`
- `slurm`
- `mpi`
- `mirheo`
- `korali`
- `pyro`

Optional-runtime tests must use those markers and the helpers in `tests/support/optional_dependencies.py` instead of ad hoc imports. A missing optional dependency should produce a clear skip or failure message that names the missing runtime and how to enable it.

## Default Feedback Loop

The default local and CI test command remains:

```bash
pytest
```

Default tests must be deterministic and must not require GPU, Slurm, Mirheo, Korali, Pyro, MPI, large external data, or private filesystem paths. Tests that inspect HPC launchers or optional-runtime policy may run by default only when they are pure static checks.

`tests/conftest.py` clears ambient MesoUQ platform/runtime environment variables so a sourced Karolina or Vega shell does not change unit-test expectations. Tests that need those variables must set them explicitly.

## Explicit Test Slices

Integration-only:

```bash
pytest -m integration
```

GPU/CUDA-oriented:

```bash
pytest -m "gpu or cuda"
```

HPC/Slurm/operational:

```bash
pytest -m "hpc or slurm or operational"
```

Optional runtime families:

```bash
pytest -m "pyro or mpi or mirheo or korali"
```

Full local validation still runs the repository default suite plus docs, package build, installed-package smoke, and the Karolina GPU validation matrix when a migration issue is closed.

## Recurrence Rules

- New reusable package behavior belongs in `tests/unit/` unless it needs multiple subsystems.
- New cross-module or workflow assembly behavior belongs in `tests/integration/`.
- New platform launchers, sbatch templates, and operator commands need marker coverage and documentation.
- No default test may write to `_out/`, `_runs/`, `_ci/`, `out_hierarchical/`, `runtime/`, or root scheduler logs.
- Installed-package and import-boundary checks must avoid repository-root import leakage.
- Large or generated scientific artifacts must be represented by manifests or tiny fixtures, not copied into `tests/`.
