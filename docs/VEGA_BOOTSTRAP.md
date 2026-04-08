# Vega bootstrap

This page documents the supported fresh-clone bootstrap path for Vega.

## Goal

A fresh clone should not depend on a user-global Korali install or an ad hoc `PYTHONPATH`.

The supported path is:

- use the repo checkout as the anchor
- place all Vega-specific state under `_vega/`
- build vendored `extern/korali/` into `_vega/korali/install`
- source the generated `_vega/korali/env.sh` before running workflows

## Recommended module stack

```bash
module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  openmpi/4.1.2.1 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0
```

## Python environment

Create or activate your preferred Python environment, then install the Python stack needed for local testing plus the Meson build helpers used by the vendored Korali bootstrap:

```bash
python -m venv _vega/venv
source _vega/venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test,mpi]"
pip install pybind11 meson ninja
```

The helper script below can also install `pybind11`, `meson`, and `ninja` into the active environment automatically.

## Preflight check

Run the Vega doctor before building Korali:

```bash
python scripts/vega/doctor_vega.py
```

The doctor reports:

- loaded module stack
- required commands (`mpicxx`, `nvcc`, `meson`, `ninja`, `pkg-config`)
- required pkg-config packages (`gsl`, `eigen3`)
- Python modules needed for the bootstrap path
- whether `korali` is resolving from an external user-global path instead of repo-local `_vega/`

If you already have a user-global Korali on `PYTHONPATH`, the doctor will report it as contamination that should be replaced by the repo-local install.

## Build vendored Korali

```bash
bash scripts/vega/bootstrap_korali.sh --jobs 8
```

Default behavior:

- builds `extern/korali/`
- installs into `_vega/korali/install`
- writes `_vega/korali/env.sh`
- records the bootstrap log at `_vega/logs/bootstrap_korali.log`

Optional flags:

- `--python-bin /abs/path/python`
- `--jobs N`
- `--reconfigure`
- `--native-cuda-batch`
- `--skip-python-build-deps`

If `--jobs` is omitted, the helper uses `SLURM_CPUS_PER_TASK` on an allocated node and otherwise caps itself conservatively on a login node.

## Activate the repo-local runtime

```bash
source _vega/korali/env.sh
python scripts/vega/doctor_vega.py --strict
```

The generated env script intentionally replaces inherited `PYTHONPATH` entries so the repo-local Korali install wins over any preexisting user-global Korali.

## Next steps

With the repo-local runtime active, continue with:

- `pytest`
- `python scripts/run_vega_acceptance.py ...`
- the public Phase 1 / Phase 2 / Phase 3b / propagation / MAP wrappers as they are added in later PRs
