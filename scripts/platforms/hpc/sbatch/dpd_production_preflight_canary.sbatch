#!/usr/bin/env bash
set -euo pipefail

legacy_name="HPC""_SITE"
if [[ ${!legacy_name+x} ]]; then
  echo "${legacy_name} is no longer supported. Use --site or MESOUQ_SITE." >&2
  exit 2
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SITE_ARG=""
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --site)
      SITE_ARG="$2"
      shift 2
      ;;
    --site=*)
      SITE_ARG="${1#--site=}"
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

ENV_SITE="${MESOUQ_SITE:-}"
if [[ -n "$SITE_ARG" && -n "$ENV_SITE" && "$SITE_ARG" != "$ENV_SITE" ]]; then
  echo "Conflicting site selectors: --site=${SITE_ARG} and MESOUQ_SITE=${ENV_SITE}." >&2
  exit 2
fi
SITE="${SITE_ARG:-${ENV_SITE:-vega}}"
case "$SITE" in
  vega|karolina) ;;
  *)
    echo "Unsupported site=${SITE}. Expected one of: vega, karolina." >&2
    exit 2
    ;;
esac

export MESOUQ_SITE="$SITE"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "This dispatcher submits a site-specific Slurm template; run it with bash, not sbatch." >&2
  exit 2
fi

exec sbatch \
  --export=ALL,MESOUQ_SITE="${SITE}",REPO_ROOT="${REPO_ROOT}" \
  "${REPO_ROOT}/scripts/platforms/${SITE}/sbatch/dpd_production_preflight_canary.sbatch" \
  "--site" "$SITE" "${args[@]}"
