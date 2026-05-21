#!/usr/bin/env bash
set -euo pipefail


usage() {
  cat <<'EOF'
Usage: bootstrap_korali.sh [--python-bin PYTHON] [--jobs N] [--reconfigure] [--native-cuda-batch] [--skip-python-build-deps]

Build and install the vendored extern/korali tree into the site runtime area.

Expected environment:
  - Vega module stack loaded
  - canonical site env Python with pip available
EOF
}

python_bin="${PYTHON_BIN:-python}"
site_arg=""
build_jobs="${KORALI_BUILD_JOBS:-}"
reconfigure=0
native_cuda_batch=0
install_python_build_deps=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)
      site_arg="$2"
      shift 2
      ;;
    --site=*)
      site_arg="${1#--site=}"
      shift
      ;;
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --jobs)
      build_jobs="$2"
      shift 2
      ;;
    --reconfigure)
      reconfigure=1
      shift
      ;;
    --native-cuda-batch)
      native_cuda_batch=1
      shift
      ;;
    --skip-python-build-deps)
      install_python_build_deps=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done


legacy_name="HPC""_SITE"
if [[ ${!legacy_name+x} ]]; then
  echo "${legacy_name} is no longer supported. Use --site or MESOUQ_SITE." >&2
  exit 2
fi
env_site="${MESOUQ_SITE:-}"
if [[ -n "$site_arg" && -n "$env_site" && "$site_arg" != "$env_site" ]]; then
  echo "Conflicting site selectors: --site=${site_arg} and MESOUQ_SITE=${env_site}." >&2
  exit 2
fi
site="${site_arg:-${env_site:-vega}}"
case "$site" in
  vega|karolina) ;;
  *)
    echo "Unsupported site=${site}. Expected one of: vega, karolina." >&2
    exit 2
    ;;
esac
export MESOUQ_SITE="$site"

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "$script_path")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

eval "$("$python_bin" - <<'PY' "$repo_root"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths  # noqa: E402

paths = get_runtime_paths(repo_root)
print(f"SITE={paths.site}")
print(f"SITE_ROOT={paths.site_root}")
print(f"LOGS_DIR={paths.logs_dir}")
print(f"KORALI_SOURCE={paths.korali_source}")
print(f"KORALI_BUILD_DIR={paths.korali_build_dir}")
print(f"KORALI_PREFIX={paths.korali_prefix}")
print(f"KORALI_SITE_PACKAGES={paths.korali_site_packages}")
print(f"KORALI_ENV_SCRIPT={paths.korali_env_script}")
PY
)"

mkdir -p "$LOGS_DIR" "$(dirname "$KORALI_ENV_SCRIPT")"
log_file="$LOGS_DIR/bootstrap_korali.log"
exec > >(tee "$log_file") 2>&1

echo "Repo root: $repo_root"
echo "Site:      $SITE"
echo "Log file:  $log_file"
echo "Install to: $KORALI_PREFIX"

if [[ -z "$build_jobs" ]]; then
  if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    build_jobs="${SLURM_CPUS_PER_TASK}"
  else
    build_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
    if [[ -z "${SLURM_JOB_ID:-}" && "$build_jobs" -gt 8 ]]; then
      build_jobs=8
    fi
  fi
fi

if ! [[ "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid --jobs value: $build_jobs" >&2
  exit 2
fi

echo "Compile jobs: $build_jobs"

if [[ "$python_bin" == */* ]]; then
  python_bin_dir="$(cd "$(dirname "$python_bin")" && pwd)"
else
  python_bin_dir="$(dirname "$(command -v "$python_bin")")"
fi
python_bin="$(readlink -f "$python_bin")"
export PATH="$python_bin_dir${PATH:+:$PATH}"
export PYTHONNOUSERSITE=1

if [[ "$install_python_build_deps" -eq 1 ]]; then
  "$python_bin" -m pip install pybind11 meson ninja
fi

for command in "$python_bin" mpicxx pkg-config meson ninja; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if ! pkg-config --exists gsl; then
  echo "GSL not found via pkg-config. Load GSL/2.7-GCC-12.2.0 first." >&2
  exit 1
fi

if ! pkg-config --exists eigen3; then
  echo "Eigen3 not found via pkg-config. Load Eigen/3.4.0-GCCcore-12.2.0 first." >&2
  exit 1
fi

sanitized_pythonpath="$("$python_bin" - <<'PY' "$repo_root"
from pathlib import Path
import os
import sys

repo_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import build_runtime_pythonpath, get_runtime_paths  # noqa: E402

paths = get_runtime_paths(repo_root)
print(build_runtime_pythonpath(repo_root, paths.korali_site_packages, os.environ.get("PYTHONPATH", ""), include_existing=True))
PY
)"
export PYTHONPATH="$sanitized_pythonpath"

meson_args=(
  setup
  "$KORALI_BUILD_DIR"
  "$KORALI_SOURCE"
  --buildtype=release
  --prefix="$KORALI_PREFIX"
  -Dmpi=true
  -Dmpi4py=true
  -Dopenmp=false
)

if [[ -d "$KORALI_BUILD_DIR" ]]; then
  if [[ "$reconfigure" -eq 1 ]]; then
    meson_args+=(--wipe)
  else
    meson_args+=(--reconfigure)
  fi
else
  meson_args+=(--wipe)
fi

if [[ "$native_cuda_batch" -eq 1 ]]; then
  meson_args+=(-Dnative_cuda_batch=true)
fi

echo "Running: meson ${meson_args[*]}"
meson "${meson_args[@]}"
echo "Running: meson compile -C $KORALI_BUILD_DIR -j $build_jobs"
meson compile -C "$KORALI_BUILD_DIR" -j "$build_jobs"
echo "Running: meson install -C $KORALI_BUILD_DIR --no-rebuild"
meson install -C "$KORALI_BUILD_DIR" --no-rebuild

if [[ ! -f "$KORALI_SITE_PACKAGES/korali/__init__.py" ]]; then
  echo "Korali install completed but no Python package was found at $KORALI_SITE_PACKAGES" >&2
  exit 1
fi

"$python_bin" - <<'PY' "$repo_root" "$KORALI_ENV_SCRIPT"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
env_script = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths, render_korali_env_script  # noqa: E402

paths = get_runtime_paths(repo_root)
env_script.write_text(render_korali_env_script(paths), encoding="utf-8")
PY

chmod +x "$KORALI_ENV_SCRIPT"

echo ""
echo "Korali bootstrap completed."
echo "Source the repo-local runtime before running workflows:"
echo "  source $KORALI_ENV_SCRIPT"
echo "Then re-run the doctor in strict mode:"
echo "  $python_bin $repo_root/scripts/platforms/hpc/doctor_hpc.py --strict"
