#!/bin/bash

main () {
  mkdir -p logs mesh anchor

  bash clean_all.sh

  python3 generate.py -p theta 0.01 0.1 10 --object "gv" --forward

  bash commands.txt
}

time main
