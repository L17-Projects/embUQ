#!/bin/bash
set -euo pipefail

mode=$1

simnum0="00001"
simnum=${2:-${simnum0}}

nranks=${3:-${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}}
domain_ranks=${4:-${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}}
export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS="${domain_ranks}"

mkdir -p logs restart mesh parameter force trj_eq stats

echo "Simulation number: $simnum"
echo "Mirheo domain ranks: ${domain_ranks}"

python3 parameters.py --simnum "${simnum}"
if [[ -n "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON:-}" ]]; then
    python3 -m meso_uq.structures.gv.material_parameters \
        "parameter/parameters.prms${simnum}.yaml" \
        --overrides-json "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON}"
fi
mpirun --bind-to none -np "${nranks}" python3 equil.py "${mode}" --simnum "${simnum}" --domain-ranks "${domain_ranks}"
