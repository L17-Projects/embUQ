#!/bin/bash
set -euo pipefail

mode=$1

simnum0="00001"
simnum=${2:-${simnum0}}

nranks=${3:-${MESOUQ_GV_EIGENMODES_MPI_RANKS:-1}}

mkdir -p logs restart mesh parameter force trj_eq stats

echo "Simulation number: $simnum"

python3 parameters.py --simnum ${simnum}
if [[ -n "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON:-}" ]]; then
    python3 -m meso_uq.structures.gv.material_parameters \
        "parameter/parameters.prms${simnum}.yaml" \
        --overrides-json "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON}"
fi
mpirun --bind-to none -np ${nranks} python3 equil.py $mode --simnum ${simnum}
