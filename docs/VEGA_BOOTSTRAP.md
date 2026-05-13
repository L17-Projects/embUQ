# Vega bootstrap

This page documents the supported fresh-clone bootstrap path for Vega.

## Goal

A fresh clone should not depend on a user-global Korali install or an ad hoc `PYTHONPATH`.

The supported path is:

- use the repo checkout as the anchor
- place all Vega-specific state under `_vega/`
- build vendored `extern/korali/` into `_vega/korali/install`
- build Mirheo from the locked external source path into `_vega/mirheo/`
- install repo-local TinyTeX into `_vega/tinytex/`
- source the generated `_vega/korali/env.sh` before running workflows

## Recommended module stack

```bash
module purge
module load \
  Python/3.10.8-GCCcore-12.2.0 \
  OpenMPI/4.1.4-GCC-12.2.0 \
  CUDA/12.2.2 \
  GSL/2.7-GCC-12.2.0 \
  Eigen/3.4.0-GCCcore-12.2.0 \
  CMake/3.24.3-GCCcore-12.2.0 \
  HDF5/1.14.0-gompi-2022b \
  MPFR/4.2.0-GCCcore-12.2.0 \
  GMP/6.2.1-GCCcore-12.2.0
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
The Mirheo bootstrap helper creates `_vega/gv_venv` and installs the GV runtime Python dependencies there, including `h5py` and `MDAnalysis`.

That editable install now includes the mesh-preparation dependency `trimesh`, which is required by the public Phase 1 workflow bootstrap for compression and indentation.

## Preflight check

Run the Vega doctor before building Korali:

```bash
python scripts/platforms/vega/doctor_vega.py
```

The doctor reports:

- loaded module stack
- required commands (`mpicxx`, `nvcc`, `meson`, `ninja`, `pkg-config`)
- required pkg-config packages (`gsl`, `eigen3`)
- Python modules needed for the bootstrap path
- whether `korali` is resolving from an external user-global path instead of repo-local `_vega/`

If you already have a user-global Korali on `PYTHONPATH`, the doctor will report it as contamination that should be replaced by the repo-local install.

If you also need MAP Mirheo workflows, use the Mirheo-aware doctor mode:

```bash
python scripts/platforms/vega/doctor_vega.py --with-mirheo
```

## Build vendored Korali

```bash
bash scripts/platforms/vega/bootstrap_korali.sh --jobs 8
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

## Build repo-local Mirheo from the locked source path

The Mirheo source default is tracked in [`extern/mirheo.lock.json`](../extern/mirheo.lock.json). Today that lock points to:

- `/ceph/hpc/home/eubrieucb/software/Mirheo`

You can override it temporarily with `MESOUQ_MIRHEO_SRC=/abs/path/to/Mirheo` or `--source /abs/path/to/Mirheo`.

```bash
bash scripts/platforms/vega/bootstrap_mirheo.sh --jobs 8
```

Default behavior:

- resolves Mirheo from `extern/mirheo.lock.json`
- builds it into `_vega/mirheo/build`
- installs CMake outputs into `_vega/mirheo/install`
- creates `_vega/gv_venv`
- installs `h5py`, `MDAnalysis`, and the Mirheo Python package into `_vega/gv_venv`
- writes `_vega/mirheo/env.sh`
- writes `_vega/gv_venv/env.sh`
- records `_vega/mirheo/source_snapshot.json`
- records `_vega/logs/bootstrap_mirheo.log`

Supported MAP Mirheo micro-canary floor on Vega:

- `--n-displacements 1`
- `--numsteps 200`
- `--numsteps-eq 200`

Lower values are outside the supported sanity/canary contract and can trigger payload-level instability.
The MAP Mirheo smoke path prepares init directories automatically per dataset under
`<lane output>/map_mirheo/_scratch/<dataset_name>`; preexisting repo-root `_init_*_map`
directories are not required. Missing Phase 3b MAP manifests, init templates, or Mirheo
bootstrap inputs are hard failures, not skip conditions.

Optional flags:

- `--python-bin /abs/path/python`
- `--source /abs/path/to/Mirheo`
- `--jobs N`
- `--reconfigure`
- `--skip-python-deps`

## Activate the repo-local runtime

```bash
source _vega/korali/env.sh
source _vega/gv_venv/env.sh
source _vega/mirheo/env.sh
source _vega/tinytex/env.sh
python scripts/platforms/vega/doctor_vega.py --strict --with-mirheo --with-tex
```

The generated env script for Korali intentionally replaces inherited `PYTHONPATH` entries so the repo-local install wins over any preexisting user-global Korali.
The Mirheo env script records the resolved source path, repo-local build/install locations, and the source snapshot manifest used for reproducibility.
The `gv_venv` env script activates the dedicated GV runtime Python path, records Mirheo import paths, and exports the OpenMPI library path explicitly for rank launches.

For GV runtime hardening checks (mirheo import, `libmirheo`, `scale_space` resolution, MDAnalysis), run:

```bash
python scripts/platforms/vega/doctor_vega.py --with-gv-runtime
```

## Build repo-local TinyTeX for paper-facing figures

Paper-facing figure generation uses the original UQ_DPD TeX rendering path. On Vega this is now bootstrapped repo-locally:

```bash
bash scripts/platforms/vega/bootstrap_tex.sh
source _vega/tinytex/env.sh
python scripts/platforms/vega/doctor_vega.py --with-tex
```

The TinyTeX bootstrap installs the exact packages needed by the figure scripts, including:

- `psnfss` / `helvet.sty`
- `sansmath.sty`
- `revtex4-2.cls`
- `preview.sty`
- `dvipng`

## Next steps

With the repo-local runtime active, continue with:

- `pytest`
- `python scripts/platforms/vega/run_validation_matrix.py ...`
- `python scripts/run_vega_acceptance.py ...`
- the public Phase 1 / Phase 2 / Phase 3b / propagation / MAP wrappers as they are added in later PRs
