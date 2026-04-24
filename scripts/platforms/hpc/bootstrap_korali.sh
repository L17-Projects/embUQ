#!/usr/bin/env bash
set -euo pipefail

site="${HPC_SITE:-vega}"
case "$site" in
  vega|karolina) ;;
  *)
    echo "Unsupported HPC_SITE=${site}. Expected one of: vega, karolina." >&2
    exit 2
    ;;
esac

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/${site}/bootstrap_korali.sh" "$@"
