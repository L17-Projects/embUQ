#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HPC_SITE=karolina
export MESOUQ_SITE=karolina
exec "${script_dir}/../vega/bootstrap_mirheo.sh" "$@"
