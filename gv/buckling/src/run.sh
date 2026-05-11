#!/bin/bash
set -euo pipefail

mode=$1

simnum0="00001"
simnum=${2:-${simnum0}}

nranks=${3:-${MESOUQ_GV_MPI_RANKS:-2}}

mkdir -p logs restart mesh parameter force trj_eq stats anchor ply_eq pressure

echo "Simulation number: $simnum"

python3 parameters.py --simnum ${simnum}
if [[ -n "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON:-}" ]]; then
    python3 -m meso_uq.structures.gv.material_parameters \
        "parameter/parameters.prms${simnum}.yaml" \
        --overrides-json "${MESOUQ_GV_MATERIAL_OVERRIDES_JSON}"
fi

if [[ -n "${MESOUQ_OPENMPI_LIB_DIR:-}" && -d "${MESOUQ_OPENMPI_LIB_DIR}" ]]; then
    export LD_LIBRARY_PATH="${MESOUQ_OPENMPI_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
elif [[ -d /cvmfs/sling.si/modules/el7/software/OpenMPI/4.1.4-GCC-12.2.0/lib ]]; then
    export LD_LIBRARY_PATH="/cvmfs/sling.si/modules/el7/software/OpenMPI/4.1.4-GCC-12.2.0/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

mpirun --bind-to none \
    -x PATH \
    -x PYTHONPATH \
    -x LD_LIBRARY_PATH \
    -x MESOUQ_GV_MIRHEO_MODULE \
    -x MESOUQ_GV_PAPER_EXACT \
    -x MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE \
    -x MESOUQ_GV_BUCKLING_FLUID_MODE \
    -x MESOUQ_GV_BUCKLING_FLUID_STABILIZATION \
    -x MESOUQ_GV_BUCKLING_PIN_OBJECT \
    -x MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE \
    -np ${nranks} python3 equil.py $mode --simnum ${simnum}
