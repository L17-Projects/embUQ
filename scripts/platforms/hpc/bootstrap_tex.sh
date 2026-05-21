#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bootstrap_tex.sh [--site SITE] [--python-bin PYTHON]

Install or refresh TinyTeX support under the canonical site runtime root:
  ${MESOUQ_SITE_RUNTIME_ROOT}/tinytex
USAGE
}

python_bin="${PYTHON_BIN:-python3}"
site_arg=""

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

eval "$("$python_bin" - <<'PYCODE' "$repo_root"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths  # noqa: E402

paths = get_runtime_paths(repo_root)
print(f"SITE={paths.site}")
print(f"SITE_ROOT={paths.site_root}")
print(f"LOGS_DIR={paths.logs_dir}")
print(f"TINYTEX_ROOT={paths.tinytex_root}")
print(f"TINYTEX_BIN_DIR={paths.tinytex_bin_dir}")
print(f"TINYTEX_ENV_SCRIPT={paths.tinytex_env_script}")
PYCODE
)"

mkdir -p "${LOGS_DIR}" "${SITE_ROOT}"
log_path="${LOGS_DIR}/bootstrap_tex.log"

export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -x "${TINYTEX_BIN_DIR}/tlmgr" ]]; then
  rm -rf "${TINYTEX_ROOT}"
  installer="$(mktemp)"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${installer}" https://yihui.org/tinytex/install-unx.sh
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "${installer}" https://yihui.org/tinytex/install-unx.sh
  else
    echo "Missing downloader: expected wget or curl to install TinyTeX." >&2
    exit 1
  fi
  TINYTEX_DIR="${TINYTEX_ROOT}" bash "${installer}" >>"${log_path}" 2>&1
  rm -f "${installer}"
fi

export PATH="${TINYTEX_BIN_DIR}:${PATH}"

packages=(
  dvipng
  preview
  cm-super
  type1cm
  psnfss
  sansmath
  amsmath
  amsfonts
  mathtools
  booktabs
  siunitx
  multirow
  caption
  ulem
  pgf
  xcolor
  float
  hyperref
  natbib
  revtex
)

tlmgr install "${packages[@]}" >>"${log_path}" 2>&1

"$python_bin" - <<'PYCODE' "$repo_root"
from pathlib import Path
import sys

repo_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_runtime_paths, render_tinytex_env_script  # noqa: E402

paths = get_runtime_paths(repo_root)
paths.tinytex_root.mkdir(parents=True, exist_ok=True)
paths.tinytex_env_script.write_text(render_tinytex_env_script(paths), encoding="utf-8")
PYCODE

printf 'TinyTeX ready at %s
' "${TINYTEX_ROOT}"
printf 'Source with: source %s
' "${TINYTEX_ENV_SCRIPT}"
