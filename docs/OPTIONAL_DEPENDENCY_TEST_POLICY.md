# Optional dependency test policy

The test suite treats optional dependencies as explicit contracts. Import-time collection must stay light, and guards must fail or skip with messages that tell the operator what is missing and what to do next.

The helper functions for these checks live in `tests/support/optional_dependencies.py`. They are intentionally stdlib-only so collection still works when `pyro`, `mpi4py`, `mirheo`, `korali`, CUDA, or Slurm are absent.

## Marker contract

The repository declares these pytest markers:

- `pyro`
- `mpi`
- `mirheo`
- `korali`
- `cuda`
- `slurm`
- `gpu`
- `hpc`
- `operational`
- `slow`
- `integration`

The guard tests verify that the `pyproject.toml` marker list keeps that set intact.

## Run commands

Default local collection:

```bash
pytest
```

Integration-only slice:

```bash
pytest -m integration
```

GPU-oriented slice:

```bash
pytest -m "gpu or cuda"
```

Karolina-oriented slice:

```bash
pytest -m "hpc or slurm or operational"
```

Vega-oriented slice:

```bash
pytest -m "gpu or operational"
```

Full operational slice:

```bash
pytest -m "integration or gpu or hpc or slurm or operational"
```

## Guard behavior

- `pyro`, `mpi`, and `mirheo` use importlib-based availability checks and runtime-import checks.
- `korali` distinguishes vendored source presence from runtime importability, so tests can explain whether the repo checkout is missing or only the runtime bootstrap is incomplete.
- `cuda` and `gpu` avoid collecting `torch` unless the guard is executed.
- `slurm` detects scheduler tools from `PATH` and can be simulated with a mocked `shutil.which`.
- `hpc` and `operational` compose the lower-level guards so end-to-end lanes can emit a single actionable failure message.

When writing new tests, prefer these helpers over ad hoc `pytest.importorskip()` calls for the optional runtime surfaces.
