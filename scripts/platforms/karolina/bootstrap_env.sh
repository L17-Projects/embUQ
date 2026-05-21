#!/usr/bin/env bash
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  if [[ -f /apps/all/Lmod/8.7.37/lmod/lmod/init/bash ]]; then
    source /apps/all/Lmod/8.7.37/lmod/lmod/init/bash
  fi
fi

module purge || true
modules=(
  CUDA/12.4.0
  HDF5/1.14.0-gompi-2022b
  GSL/2.7-GCC-12.3.0
  Eigen/3.4.0-GCCcore-12.2.0
  Python/3.10.8-GCCcore-12.2.0
  CMake/3.24.3-GCCcore-12.2.0
  MPFR/4.2.0-GCCcore-12.2.0
  GMP/6.2.1-GCCcore-12.2.0
)
module load "${modules[@]}"
unset modules

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/../hpc/bootstrap_env.sh" --site karolina "$@"
