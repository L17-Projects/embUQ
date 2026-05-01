#!/usr/bin/env bash
set -euo pipefail

simnum=${1:-"00001eq"}

python3 create_gv.py --simnum ${simnum} #$height

python3 statistics.py --simnum ${simnum}

# plot_gv.py is not part of the canonical runtime source bundle.

mv out.off gv.off
