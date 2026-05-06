#!/bin/bash

mkdir -p output

rm -r output/*

python3 combine.py

python3 initial.py

python3 all_analysis.py

bash trim_svd.sh
