cp ../trj_eq/sim0001/emb_0000000.xyz .
python3 combine.py
python3 initial.py
python3 all_analysis.py
bash trim_svd.sh


