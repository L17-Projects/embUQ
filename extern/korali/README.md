# Korali — gpu-batch-eval fork

This is a modified fork of [Korali](https://github.com/cselab/korali) used by the
`Hierarchical_UQ_compression_dev` inference pipeline.

**Repository**: `BrieucB/korali`
**Branch**: `gpu-batch-eval`

The fork adds a batched-evaluation path so the full TMCMC particle population is
forwarded to the PyTorch surrogate in a single GPU tensor call instead of N serial
calls. Korali itself remains a CPU/MPI library; the GPU acceleration lives in the
Python surrogate layer. See
[`LOCAL_GPU_BATCHING_NOTES.md`](LOCAL_GPU_BATCHING_NOTES.md) for the exact modified
source files.

---

## Installing on a standalone Linux workstation (o369-style)

These instructions assume:

- a standalone workstation with no job scheduler,
- system Python ≥ 3.8,
- system OpenMPI ≥ 4.0,
- no root privileges required,
- GPU optional (the build does not need CUDA; GPU is used at runtime via PyTorch).

### 1. Clone this repo

```bash
# Recommended layout — keep korali/ next to the project repo
cd /path/to/workspace/UQ_DPD
git clone -b gpu-batch-eval git@github.com:BrieucB/korali.git korali
```

If you already have the clone, make sure you are on the right branch:

```bash
cd /path/to/workspace/UQ_DPD/korali
git checkout gpu-batch-eval
git pull
```

### 2. Create a Python virtual environment

Korali's Python bindings are tested against Python 3.8–3.11.

```bash
python3 -m venv /path/to/workspace/myenv
source /path/to/workspace/myenv/bin/activate

pip install --upgrade pip setuptools wheel
```

### 3. Install Python dependencies

```bash
pip install numpy scipy matplotlib pyyaml pydantic h5py
pip install meson ninja pybind11
MPICC=mpicc pip install --no-binary=mpi4py mpi4py
```

PyTorch is used at runtime for the surrogate forward pass:

```bash
# CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU (CUDA 12.x), if compatible:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install the HUQ project in editable mode

```bash
pip install -e /path/to/workspace/UQ_DPD/Hierarchical_UQ_compression_dev --no-deps
```

### 5. Install build dependencies for Korali

Korali needs GSL ≥ 2.5 and Eigen ≥ 3.3. Use system packages if available, otherwise provide user-local installs and export `PKG_CONFIG_PATH`, `CMAKE_PREFIX_PATH`, and `LD_LIBRARY_PATH` accordingly.

### 6. Build and install Korali

```bash
source /path/to/workspace/myenv/bin/activate
cd /path/to/workspace/UQ_DPD/korali
export KORALI_PREFIX=$HOME/Programs/korali/.local

meson setup build \
  --wipe \
  --buildtype=release \
  --prefix="$KORALI_PREFIX" \
  -Dmpi=true \
  -Dmpi4py=true \
  -Dopenmp=false

meson install -C build
```

### 7. Set KORALI_PYTHONPATH

```bash
export KORALI_PYTHONPATH=$(find "$KORALI_PREFIX" -type d -path '*/site-packages' | head -1)
```

### 8. Verify the installation

```bash
python -c "import korali, mpi4py; print('korali:', korali.__file__)"
mpirun -np 2 python -c "from mpi4py import MPI; import korali; print(MPI.COMM_WORLD.Get_rank(), korali.__file__)"
```

## Related documentation

- [`LOCAL_GPU_BATCHING_NOTES.md`](LOCAL_GPU_BATCHING_NOTES.md)
- `../docs/HPC_GPU_BATCHED_REDUCED_INDENTATION.md`
