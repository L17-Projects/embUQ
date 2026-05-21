#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bootstrap_env.sh [--site SITE] [--python-bin PYTHON] [--recreate] [--system-site-packages]
                        [--skip-python-deps] [--with-korali] [--with-mirheo] [--with-gv-cgal]
                        [--mirheo-source PATH] [--gv-cgal-source PATH] [--jobs N] [--reconfigure]

Create or refresh the canonical MesoUQ site environment:
  ${MESOUQ_SITE_RUNTIME_ROOT}/env

The script installs the full MesoUQ development/runtime Python stack into that env
and writes ${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh. Site-specific native stacks
remain outside the Python venv and are captured by generated env scripts.
USAGE
}

python_bin="${PYTHON_BIN:-python}"
site_arg=""
recreate=0
system_site_packages=0
install_python_deps=1
with_korali=0
with_mirheo=0
with_gv_cgal=0
mirheo_source=""
gv_cgal_source=""
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
    --recreate)
      recreate=1
      shift
      ;;
    --system-site-packages)
      system_site_packages=1
      shift
      ;;
    --skip-python-deps)
      install_python_deps=0
      shift
      ;;
    --with-korali)
      with_korali=1
      shift
      ;;
    --with-mirheo)
      with_mirheo=1
      shift
      ;;
    --with-gv-cgal)
      with_gv_cgal=1
      shift
      ;;
    --mirheo-source)
      mirheo_source="$2"
      shift 2
      ;;
    --gv-cgal-source)
      gv_cgal_source="$2"
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

# Resolve site paths with the requested bootstrap Python; the env Python may not exist yet.
eval "$("$python_bin" - <<'PY' "$repo_root"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import (  # noqa: E402
    UNIFIED_ENV_EXTRAS,
    discover_python_runtime_library_dirs,
    get_runtime_paths,
)

paths = get_runtime_paths(repo_root)
print(f"SITE={paths.site}")
print(f"SITE_ROOT={paths.site_root}")
print(f"LOGS_DIR={paths.logs_dir}")
print(f"ENV_ROOT={paths.env_root}")
print(f"ENV_SCRIPT={paths.env_script}")
print(f"ENV_SITE_PACKAGES={paths.env_site_packages}")
print(f"PIP_EXTRAS={','.join(UNIFIED_ENV_EXTRAS)}")
print("PYTHON_RUNTIME_LIB_DIRS=" + ":".join(str(path) for path in discover_python_runtime_library_dirs()))
PY
)"

if [[ -n "${PYTHON_RUNTIME_LIB_DIRS:-}" ]]; then
  IFS=':' read -r -a _mesouq_python_runtime_lib_dirs <<< "$PYTHON_RUNTIME_LIB_DIRS"
  for _mesouq_python_runtime_lib_dir in "${_mesouq_python_runtime_lib_dirs[@]}"; do
    if [[ -d "$_mesouq_python_runtime_lib_dir" ]]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${_mesouq_python_runtime_lib_dir}:"*) ;;
        *) export LD_LIBRARY_PATH="${_mesouq_python_runtime_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
      esac
    fi
  done
  unset _mesouq_python_runtime_lib_dir _mesouq_python_runtime_lib_dirs
fi

mkdir -p "$LOGS_DIR" "$ENV_ROOT"
log_file="$LOGS_DIR/bootstrap_env.log"
exec > >(tee "$log_file") 2>&1

echo "Repo root:  $repo_root"
echo "Site:       $SITE"
echo "Env root:   $ENV_ROOT"
echo "Env script: $ENV_SCRIPT"
echo "Log file:   $log_file"

venv_args=()
if [[ "$system_site_packages" -eq 1 ]]; then
  venv_args+=(--system-site-packages)
fi
if [[ "$recreate" -eq 1 ]]; then
  venv_args+=(--clear)
fi

if [[ "$recreate" -eq 1 || ! -x "$ENV_ROOT/bin/python" ]]; then
  echo "Creating canonical MesoUQ env at $ENV_ROOT"
  "$python_bin" -m venv "${venv_args[@]}" "$ENV_ROOT"
fi

env_python="$ENV_ROOT/bin/python"
if [[ ! -x "$env_python" ]]; then
  echo "Canonical env Python was not created: $env_python" >&2
  exit 1
fi

if [[ "$install_python_deps" -eq 1 ]]; then
  "$env_python" -m pip install --upgrade pip setuptools wheel
  "$env_python" -m pip install -e "$repo_root[$PIP_EXTRAS]"
fi

if [[ "$with_korali" -eq 1 ]]; then
  korali_args=(--site "$SITE" --python-bin "$env_python")
  if [[ "$install_python_deps" -eq 1 ]]; then
    korali_args+=(--skip-python-build-deps)
  fi
  if [[ -n "$build_jobs" ]]; then
    korali_args+=(--jobs "$build_jobs")
  fi
  if [[ "$reconfigure" -eq 1 ]]; then
    korali_args+=(--reconfigure)
  fi
  bash "$repo_root/scripts/platforms/hpc/bootstrap_korali.sh" "${korali_args[@]}"
fi

if [[ "$with_mirheo" -eq 1 ]]; then
  mirheo_args=(--site "$SITE" --python-bin "$env_python" --skip-python-deps)
  if [[ -n "$mirheo_source" ]]; then
    mirheo_args+=(--source "$mirheo_source")
  fi
  if [[ -n "$build_jobs" ]]; then
    mirheo_args+=(--jobs "$build_jobs")
  fi
  if [[ "$reconfigure" -eq 1 ]]; then
    mirheo_args+=(--reconfigure)
  fi
  bash "$repo_root/scripts/platforms/hpc/bootstrap_mirheo.sh" "${mirheo_args[@]}"
fi

if [[ "$with_gv_cgal" -eq 1 ]]; then
  gv_cgal_args=(--python-bin "$env_python")
  if [[ -n "$gv_cgal_source" ]]; then
    gv_cgal_args+=(--source "$gv_cgal_source")
  fi
  if [[ -n "$build_jobs" ]]; then
    gv_cgal_args+=(--jobs "$build_jobs")
  fi
  if [[ "$reconfigure" -eq 1 ]]; then
    gv_cgal_args+=(--reconfigure)
  fi
  site_gv_cgal_script="$repo_root/scripts/platforms/$SITE/bootstrap_gv_cgal_tools.sh"
  if [[ -x "$site_gv_cgal_script" ]]; then
    bash "$site_gv_cgal_script" "${gv_cgal_args[@]}"
  else
    bash "$repo_root/scripts/platforms/hpc/bootstrap_gv_cgal_tools.sh" --site "$SITE" "${gv_cgal_args[@]}"
  fi
fi

"$env_python" - <<'PY' "$repo_root" "$ENV_SCRIPT"
from pathlib import Path
import json
import subprocess
import sys

repo_root = Path(sys.argv[1]).resolve()
env_script = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths, render_unified_env_script, resolve_mirheo_source  # noqa: E402

paths = get_runtime_paths(repo_root)
source_root = None
try:
    candidate = resolve_mirheo_source(repo_root, site=paths.site)
    if candidate.is_dir():
        source_root = candidate
except Exception:
    source_root = None
snapshot_path = paths.mirheo_snapshot_path if paths.mirheo_snapshot_path.is_file() else None
env_script.parent.mkdir(parents=True, exist_ok=True)
env_script.write_text(
    render_unified_env_script(paths, source_root=source_root, snapshot_path=snapshot_path),
    encoding="utf-8",
)
manifest = {
    "site": paths.site,
    "repo_root": str(paths.repo_root),
    "env_root": str(paths.env_root),
    "env_script": str(paths.env_script),
    "python": str(paths.env_root / "bin" / "python"),
}
freeze = subprocess.run([str(paths.env_root / "bin" / "python"), "-m", "pip", "freeze"], text=True, capture_output=True, check=False)
manifest["pip_freeze"] = (freeze.stdout or "").splitlines()
(paths.env_root / "bootstrap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
chmod +x "$ENV_SCRIPT"

echo "Validating meso_uq import through canonical env activation."
(
  source "$ENV_SCRIPT"
  "$env_python" - <<'PY'
import meso_uq
print(f"meso_uq import ok: {meso_uq.__file__}")
PY
)

echo ""
echo "Unified MesoUQ environment completed."
echo "Source it with:"
echo "  source $ENV_SCRIPT"
echo "Then run:"
echo "  $env_python $repo_root/scripts/platforms/hpc/doctor_hpc.py --site $SITE --strict"
