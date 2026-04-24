#!/usr/bin/env bash
set -euo pipefail

echo "WARNING: scripts/platforms/vega/bootstrap_mirheo.sh is a deprecated Vega-specific entrypoint." >&2
echo "WARNING: use scripts/platforms/hpc/bootstrap_mirheo.sh with HPC_SITE=vega|karolina." >&2

usage() {
  cat <<'EOF'
Usage: bootstrap_mirheo.sh [--python-bin PYTHON] [--source PATH] [--jobs N] [--reconfigure] [--skip-python-deps]

Build Mirheo from an external pinned source path into the repo-local _vega/ area
and install the Python package into the active repo-local venv.

Expected environment:
  - HPC module stack loaded
  - active Python environment with pip available
  - a Mirheo source tree available at the locked path or via --source / MESOUQ_MIRHEO_SRC
EOF
}

python_bin="${PYTHON_BIN:-python}"
source_override=""
build_jobs="${MIRHEO_BUILD_JOBS:-}"
reconfigure=0
install_python_deps=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      python_bin="$2"
      shift 2
      ;;
    --source)
      source_override="$2"
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
    --skip-python-deps)
      install_python_deps=0
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

eval "$("$python_bin" - <<'PY' "$repo_root" "$source_override"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
override = sys.argv[2] or None
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_vega_paths, resolve_mirheo_source  # noqa: E402

paths = get_vega_paths(repo_root)
source_root = resolve_mirheo_source(repo_root, override=override)
print(f"VEGA_ROOT={paths.vega_root}")
print(f"LOGS_DIR={paths.logs_dir}")
print(f"MIRHEO_SOURCE={source_root}")
print(f"MIRHEO_BUILD_DIR={paths.mirheo_build_dir}")
print(f"MIRHEO_PREFIX={paths.mirheo_prefix}")
print(f"MIRHEO_PACKAGE_DIR={paths.mirheo_package_dir}")
print(f"MIRHEO_ENV_SCRIPT={paths.mirheo_env_script}")
print(f"MIRHEO_SNAPSHOT_PATH={paths.mirheo_snapshot_path}")
PY
)"

mkdir -p "$LOGS_DIR" "$(dirname "$MIRHEO_ENV_SCRIPT")"
log_file="$LOGS_DIR/bootstrap_mirheo.log"
exec > >(tee "$log_file") 2>&1

echo "Repo root:      $repo_root"
echo "Log file:       $log_file"
echo "Mirheo source:  $MIRHEO_SOURCE"
echo "Build dir:      $MIRHEO_BUILD_DIR"
echo "Install prefix: $MIRHEO_PREFIX"

if [[ ! -f "$MIRHEO_SOURCE/CMakeLists.txt" || ! -f "$MIRHEO_SOURCE/setup.py" ]]; then
  echo "Mirheo source path is missing required files: $MIRHEO_SOURCE" >&2
  exit 1
fi

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

if [[ "$install_python_deps" -eq 1 ]]; then
  "$python_bin" -m pip install h5py
fi

for command in "$python_bin" mpicxx nvcc cmake make; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if [[ "$reconfigure" -eq 1 ]]; then
  rm -rf "$MIRHEO_BUILD_DIR" "$MIRHEO_PREFIX" "$MIRHEO_PACKAGE_DIR"
fi

mkdir -p "$(dirname "$MIRHEO_BUILD_DIR")" "$MIRHEO_PREFIX"

cmake_args=(
  -S "$MIRHEO_SOURCE"
  -B "$MIRHEO_BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$MIRHEO_PREFIX"
  -DPYBIND11_FINDPYTHON=ON
  -DMIR_DOUBLE_PRECISION=OFF
  -DMIR_MEMBRANE_DOUBLE=OFF
  -DMIR_ENABLE_STACKTRACE=OFF
  -DMIR_BUILD_TESTS=OFF
)

echo "Running: cmake ${cmake_args[*]}"
cmake "${cmake_args[@]}"
echo "Running: cmake --build $MIRHEO_BUILD_DIR -j $build_jobs"
cmake --build "$MIRHEO_BUILD_DIR" -j "$build_jobs"
echo "Running: cmake --install $MIRHEO_BUILD_DIR"
cmake --install "$MIRHEO_BUILD_DIR"

libmirheo_candidates=("$MIRHEO_BUILD_DIR"/src/mirheo/bindings/libmirheo.cpython*.so)
if [[ ! -e "${libmirheo_candidates[0]}" ]]; then
  echo "Mirheo build completed but no Python extension was found under $MIRHEO_BUILD_DIR/src/mirheo/bindings" >&2
  exit 1
fi

rm -rf "$MIRHEO_PACKAGE_DIR"
mkdir -p "$MIRHEO_PACKAGE_DIR"
rsync -a \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$MIRHEO_SOURCE/mirheo/" "$MIRHEO_PACKAGE_DIR/mirheo/"
cp "$MIRHEO_SOURCE/setup.py" "$MIRHEO_PACKAGE_DIR/setup.py"
if [[ -f "$MIRHEO_SOURCE/README.md" ]]; then
  cp "$MIRHEO_SOURCE/README.md" "$MIRHEO_PACKAGE_DIR/README.md"
fi
ln -sfn "$MIRHEO_BUILD_DIR" "$MIRHEO_PACKAGE_DIR/build"

echo "Installing Mirheo Python package into the active environment"
"$python_bin" -m pip install --no-deps --force-reinstall "$MIRHEO_PACKAGE_DIR"

"$python_bin" - <<'PY' "$repo_root" "$MIRHEO_SOURCE" "$MIRHEO_SNAPSHOT_PATH" "$MIRHEO_ENV_SCRIPT"
from pathlib import Path
import json
import sys

repo_root = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
snapshot_path = Path(sys.argv[3]).resolve()
env_script = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import gather_mirheo_source_snapshot, get_vega_paths, render_mirheo_env_script  # noqa: E402

paths = get_vega_paths(repo_root)
snapshot = gather_mirheo_source_snapshot(source_root)
snapshot["build_dir"] = str(paths.mirheo_build_dir)
snapshot["install_prefix"] = str(paths.mirheo_prefix)
snapshot["package_dir"] = str(paths.mirheo_package_dir)
snapshot_path.parent.mkdir(parents=True, exist_ok=True)
snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
env_script.write_text(
    render_mirheo_env_script(paths, source_root=source_root, snapshot_path=snapshot_path),
    encoding="utf-8",
)
PY

chmod +x "$MIRHEO_ENV_SCRIPT"

echo "Verifying Mirheo and h5py imports"
"$python_bin" - <<'PY'
import h5py
import inspect
import mirheo

print(f"h5py={h5py.__version__}")
print(f"mirheo={inspect.getfile(mirheo)}")
PY

echo ""
echo "Mirheo bootstrap completed."
echo "Source the repo-local runtime before running MAP Mirheo workflows:"
echo "  source $MIRHEO_ENV_SCRIPT"
echo "Then re-run the doctor in strict mode:"
echo "  $python_bin $repo_root/scripts/platforms/vega/doctor_vega.py --strict --with-mirheo"
