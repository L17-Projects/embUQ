#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bootstrap_gv_cgal_tools.sh [--site SITE] [--python-bin PYTHON] [--source PATH]
                                  [--jobs N] [--reconfigure]

Build the GV CGAL scale_space helper into the canonical site runtime:
  ${MESOUQ_SITE_RUNTIME_ROOT}/gv_cgal_tools/bin/scale_space

Expected environment:
  - site compiler/CMake/CGAL/Boost/Eigen/GMP/MPFR modules loaded
  - canonical site env created by bootstrap_env.sh, or Python with venv available
USAGE
}

python_bin="${PYTHON_BIN:-python}"
site_arg=""
source_override="${MESOUQ_GV_CGAL_SOURCE:-}"
build_jobs="${MESOUQ_BUILD_JOBS:-}"
reconfigure=0

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

eval "$("$python_bin" - <<'PY' "$repo_root" "$source_override"
from pathlib import Path
import shlex
import sys

repo_root = Path(sys.argv[1]).resolve()
override = sys.argv[2] or None
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths  # noqa: E402

paths = get_runtime_paths(repo_root)
candidate_sources = []
if override:
    candidate_sources.append(Path(override).expanduser())
candidate_sources.extend(
    [
        repo_root / "extern" / "gv_cgal_tools" / "src",
        repo_root / f"_{paths.site}" / "gv_cgal_tools" / "src",
    ]
)
source_root = None
for candidate in candidate_sources:
    resolved = candidate.resolve(strict=False)
    if (resolved / "CMakeLists.txt").is_file() and (resolved / "scale_space.cpp").is_file():
        source_root = resolved
        break
if source_root is None:
    rendered = ", ".join(str(path) for path in candidate_sources)
    raise SystemExit(f"GV CGAL source tree was not found. Checked: {rendered}")

values = {
    "SITE": paths.site,
    "SITE_ROOT": str(paths.site_root),
    "LOGS_DIR": str(paths.logs_dir),
    "GV_CGAL_SOURCE": str(source_root),
    "GV_CGAL_ROOT": str(paths.gv_cgal_tools_root),
    "GV_CGAL_BUILD_DIR": str(paths.gv_cgal_tools_root / "build"),
    "GV_CGAL_BIN_DIR": str(paths.gv_cgal_tools_bin_dir),
    "GV_SCALE_SPACE_BINARY": str(paths.scale_space_binary),
    "GV_CGAL_ENV_SCRIPT": str(paths.gv_cgal_tools_env_script),
    "ENV_ROOT": str(paths.env_root),
    "ENV_ENV_SCRIPT": str(paths.env_script),
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

mkdir -p "$LOGS_DIR" "$GV_CGAL_ROOT" "$ENV_ROOT"
log_file="$LOGS_DIR/bootstrap_gv_cgal_tools.log"
exec > >(tee "$log_file") 2>&1

echo "Repo root:       $repo_root"
echo "Site:            $SITE"
echo "Log file:        $log_file"
echo "CGAL source:     $GV_CGAL_SOURCE"
echo "Build dir:       $GV_CGAL_BUILD_DIR"
echo "scale_space:     $GV_SCALE_SPACE_BINARY"
echo "Unified env:     $ENV_ROOT"

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

if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "Creating canonical MesoUQ env at $ENV_ROOT"
  "$python_bin" -m venv "$ENV_ROOT"
fi
runtime_python="$ENV_ROOT/bin/python"

for command in cmake make c++; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if [[ "$reconfigure" -eq 1 ]]; then
  rm -rf "$GV_CGAL_BUILD_DIR" "$GV_CGAL_BIN_DIR"
fi

mkdir -p "$GV_CGAL_BUILD_DIR" "$GV_CGAL_BIN_DIR"

cmake_args=(
  -S "$GV_CGAL_SOURCE"
  -B "$GV_CGAL_BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
)
echo "Running: cmake ${cmake_args[*]}"
cmake "${cmake_args[@]}"
echo "Running: cmake --build $GV_CGAL_BUILD_DIR --target scale_space --parallel $build_jobs"
cmake --build "$GV_CGAL_BUILD_DIR" --target scale_space --parallel "$build_jobs"
install -m 0755 "$GV_CGAL_BUILD_DIR/scale_space" "$GV_SCALE_SPACE_BINARY"

"$runtime_python" - <<'PY' "$repo_root" "$GV_CGAL_SOURCE" "$GV_CGAL_ENV_SCRIPT"
from pathlib import Path
import hashlib
import json
import os
import sys

repo_root = Path(sys.argv[1]).resolve()
source_root = Path(sys.argv[2]).resolve()
env_script = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths, render_gv_cgal_tools_env_script  # noqa: E402

paths = get_runtime_paths(repo_root)
library_paths = tuple(path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path)
env_script.parent.mkdir(parents=True, exist_ok=True)
env_script.write_text(render_gv_cgal_tools_env_script(paths, library_paths=library_paths), encoding="utf-8")
env_script.chmod(0o755)

source_manifest = {
    "source_root": str(source_root),
    "files": [],
}
for path in sorted(source_root.glob("*")):
    if not path.is_file():
        continue
    source_manifest["files"].append(
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
(paths.gv_cgal_tools_root / "source_snapshot.json").write_text(
    json.dumps(source_manifest, indent=2),
    encoding="utf-8",
)
print(env_script)
PY

echo "Verifying scale_space dynamic libraries"
ldd "$GV_SCALE_SPACE_BINARY"

echo ""
echo "GV CGAL tooling completed."
echo "Source the canonical runtime before GV geometry workflows:"
echo "  source $ENV_ENV_SCRIPT"
