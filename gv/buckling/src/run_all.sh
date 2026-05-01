#!/bin/bash

main () {
  mkdir -p logs mesh anchor
  
  bash clean_all.sh
  
  python3 generate.py -p bpress -91.0 -91.0 1 -p buck 0.0 0.75 50 --object "gv" --forward #--first
  
  bash commands.txt
}

time main
