#!/bin/bash

n=${1:-30}

tail -$n output/eigvectors.txt > output/eigvectors_tmp.txt
tail -$n output/eigvalues.txt > output/eigvalues_tmp.txt

tac output/eigvectors_tmp.txt > output/eigvectors_new.txt
tac output/eigvalues_tmp.txt > output/eigvalues_new.txt
