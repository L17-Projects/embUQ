#!/bin/bash

main () {
  mkdir -p logs
  
  #bash clean_all.sh
  
  python3 generate.py -p ptan 0.4 0.4 1 --object "gv" --parallel --first -g 1 -N 1 # --first --forward
    
  sbatch run_HPC.sbatch
}

time main
