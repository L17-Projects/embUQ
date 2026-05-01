#!/bin/bash
set -euo pipefail

mode=$1

simnum0="00001"
simnum=${2:-${simnum0}}

nranks=${3:-2}

mkdir -p logs restart mesh parameter force trj_eq stats anchor anchor_min anchor_max

echo "Simulation number: $simnum"
 
python3 parameters.py --simnum ${simnum}
mpirun -np ${nranks} python3 equil.py $mode --simnum ${simnum} --vacuum
