#!/usr/bin/env bash
set -euo pipefail

if ! type module >/dev/null 2>&1; then
  if [[ -f /apps/all/Lmod/8.7.37/lmod/lmod/init/bash ]]; then
    source /apps/all/Lmod/8.7.37/lmod/lmod/init/bash
  fi
fi

module purge || true
modules=(
  Python/3.10.8-GCCcore-12.2.0
  Eigen/3.4.0-GCCcore-12.2.0
  MPFR/4.2.0-GCCcore-12.2.0
  GMP/6.2.1-GCCcore-12.2.0
  CMake/3.24.3-GCCcore-12.2.0
  CGAL/5.6-GCCcore-12.3.0
  Boost/1.82.0-GCC-12.3.0
)
module load "${modules[@]}"
unset modules

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/../hpc/bootstrap_gv_cgal_tools.sh" --site karolina "$@"
