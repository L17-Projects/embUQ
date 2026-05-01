#!/bin/bash

main () {
  mkdir -p logs mesh
  
  #bash clean_all.sh
  
  rm -r trj_eq/sim00001
  
  python3 generate.py -p bpress -91.0 -91.0 1 --object "gv" --forward --first
  
  bash commands.txt
}

time main
