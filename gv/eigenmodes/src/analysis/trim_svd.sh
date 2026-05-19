#!/bin/bash
set -euo pipefail

n=${1:-30}

python3 trim_eigenmodes.py --mode-count "$n"

#tac output/eigvectors_tmp.txt > output/eigvectors_new.txt
#tac output/eigvalues_tmp.txt > output/eigvalues_new.txt
