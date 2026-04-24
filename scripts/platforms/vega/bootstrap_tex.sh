#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
vega_root="${repo_root}/_vega"
tinytex_link="${vega_root}/tinytex"
home_tinytex="${HOME}/.TinyTeX"
env_script="${tinytex_link}/env.sh"
log_dir="${vega_root}/logs"
log_path="${log_dir}/bootstrap_tex.log"

mkdir -p "${log_dir}" "${vega_root}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -d "${home_tinytex}" ]]; then
  installer="$(mktemp)"
  wget -O "${installer}" https://yihui.org/tinytex/install-unx.sh
  bash "${installer}" >>"${log_path}" 2>&1
  rm -f "${installer}"
fi

ln -sfn "${home_tinytex}" "${tinytex_link}"
export PATH="${tinytex_link}/bin/x86_64-linux:${PATH}"

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

MESOUQ_REPO_ROOT="${repo_root}" python3 - <<'PY'
import os
import sys
from pathlib import Path
repo_root = Path(os.environ["MESOUQ_REPO_ROOT"])
sys.path.insert(0, str(repo_root / "src"))
from meso_uq.vega import get_vega_paths, render_tinytex_env_script
paths = get_vega_paths(repo_root)
paths.tinytex_root.mkdir(parents=True, exist_ok=True)
paths.tinytex_env_script.write_text(render_tinytex_env_script(paths), encoding='utf-8')
PY

printf 'TinyTeX ready at %s\n' "${tinytex_link}"
printf 'Source with: source %s\n' "${env_script}"
