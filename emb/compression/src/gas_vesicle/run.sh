#!\bin/bash

simnum=${1:-"00001eq"}

python3 create_gv.py --simnum ${simnum} #$height # input folder : src/gas_vesicle, output  : src/gas_vesicle/out.off

#python3 statistics.py --simnum ${simnum} 

#python3 plot_gv.py

mv out.off gv.off 
