#!/bin/bash

main () {
  mkdir -p logs mesh anchor

  bash clean_all.sh

  #python3 generate.py -p buck 25.0 0.0 1 --object "emb" --parallel
  python3 generate.py -p tot_force 500.0 50000.0 80 -p bpress -91.0 -100.0 1 --object "gv" --forward --first #--first 30000 for gvs  50000 for emb

  bash commands.txt
}

time main
