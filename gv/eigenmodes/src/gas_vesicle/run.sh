#!/usr/bin/env bash
set -euo pipefail

simnum=${1:-"00001eq"}

python3 create_gv.py --simnum ${simnum} #$height

python3 statistics.py --simnum ${simnum}

#python3 plot_gv.py

mv out.off gv.off
