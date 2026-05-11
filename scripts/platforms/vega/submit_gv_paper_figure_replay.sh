#!/bin/bash
# Submit GV paper replay lanes with lane-aware walltime defaults.
#
# Required:
#   CAMPAIGN_ID=<unique replay campaign id>
#
# Optional:
#   LANES="stretching buckling torsion eigenmodes"
#   PAPER_EXACT=1
#   OUTPUT_ROOT=_runs/gv/figure_replay/<campaign>
#   GV_PAPER_REPLAY_TIME_LIMIT=HH:MM:SS   # explicit override
#   STRETCHING_POINT_START=<int>
#   STRETCHING_POINT_STOP=<int>
#   BUCKLING_BUCK_MAX=<float>
#   BUCKLING_POINT_COUNT=<int>
#   BUCKLING_TIMEOUT_SECONDS=<int>

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SBATCH_SCRIPT="${REPO_ROOT}/scripts/platforms/vega/sbatch/gv_paper_figure_replay.sbatch"

CAMPAIGN_ID="${CAMPAIGN_ID:-}"
if [[ -z "${CAMPAIGN_ID}" ]]; then
  echo "CAMPAIGN_ID is required." >&2
  exit 2
fi

LANES="${LANES:-}"
PAPER_EXACT="${PAPER_EXACT:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-_runs/gv/figure_replay/${CAMPAIGN_ID}}"

_paper_exact_enabled() {
  [[ "${PAPER_EXACT}" == "1" || "${PAPER_EXACT}" == "true" || "${PAPER_EXACT}" == "TRUE" ]]
}

_single_lane() {
  local normalized
  normalized="$(xargs <<<"${LANES}")"
  [[ "${normalized}" != *" "* ]] && [[ -n "${normalized}" ]]
}

_stretching_point_count() {
  if [[ -z "${STRETCHING_POINT_STOP:-}" ]]; then
    echo ""
    return 0
  fi
  local start="${STRETCHING_POINT_START:-0}"
  echo "$((STRETCHING_POINT_STOP - start))"
}

_default_time_limit() {
  if ! _single_lane; then
    echo "24:00:00"
    return 0
  fi

  case "$(xargs <<<"${LANES}")" in
    torsion)
      echo "01:00:00"
      ;;
    buckling)
      echo "04:00:00"
      ;;
    eigenmodes)
      echo "08:00:00"
      ;;
    stretching)
      local count
      count="$(_stretching_point_count)"
      if _paper_exact_enabled && [[ -n "${count}" ]] && (( count <= 30 )); then
        echo "03:00:00"
      else
        echo "24:00:00"
      fi
      ;;
    *)
      echo "24:00:00"
      ;;
  esac
}

TIME_LIMIT="${GV_PAPER_REPLAY_TIME_LIMIT:-$(_default_time_limit)}"

exports=(
  "ALL"
  "REPO_ROOT=${REPO_ROOT}"
  "CAMPAIGN_ID=${CAMPAIGN_ID}"
  "LANES=${LANES}"
  "PAPER_EXACT=${PAPER_EXACT}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
)

if [[ -n "${STRETCHING_POINT_START:-}" ]]; then
  exports+=("STRETCHING_POINT_START=${STRETCHING_POINT_START}")
fi
if [[ -n "${STRETCHING_POINT_STOP:-}" ]]; then
  exports+=("STRETCHING_POINT_STOP=${STRETCHING_POINT_STOP}")
fi
if [[ -n "${BUCKLING_BUCK_MAX:-}" ]]; then
  exports+=("BUCKLING_BUCK_MAX=${BUCKLING_BUCK_MAX}")
fi
if [[ -n "${BUCKLING_POINT_COUNT:-}" ]]; then
  exports+=("BUCKLING_POINT_COUNT=${BUCKLING_POINT_COUNT}")
fi
if [[ -n "${BUCKLING_TIMEOUT_SECONDS:-}" ]]; then
  exports+=("BUCKLING_TIMEOUT_SECONDS=${BUCKLING_TIMEOUT_SECONDS}")
fi

IFS=,
export_arg="${exports[*]}"
unset IFS

exec sbatch --time="${TIME_LIMIT}" --export="${export_arg}" "${SBATCH_SCRIPT}"
