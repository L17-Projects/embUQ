#!/usr/bin/env python3

import os 
import yaml
from argparse import ArgumentParser

parser = ArgumentParser()       

# which parameters to iterate through: --parameter par_name start stop steps
parser.add_argument('-p', '--parameter', dest = 'par', action = 'append', nargs = 4, default = None)

# takes parameters-default.obj.yaml file
parser.add_argument('-o', '--object', dest = 'obj', default = None)

# if you want a series of simulations, which use previous state as the new initial state type --forward
# if you want forward and backward sweep type --hysteresis
# if you want just individual equlibration of each script type: --parallel
group_sweep = parser.add_mutually_exclusive_group(required=True)
group_sweep.add_argument('--forward', action = 'store_true', default = None)
group_sweep.add_argument('--hysteresis', action = 'store_true', default = None)
group_sweep.add_argument('--parallel', action = 'store_true', default = None)

parser.add_argument('-g', type = int, default = 1)
parser.add_argument('-N', type = int, default = 1)

# if you want the first simulation to be a pair of equilibration and then restart
# in combination with --forward or --hysteresis this yields: --equil 00001 --restart 00001 --restart 00002 --restart 00003 ...
# in combination with --parallel this yields: --equil 00001 --restart 00001 --equil 00002 --restart 00002 ...
# without --first the --forward or --hysteresis yields: --equil 00001 --restart 00002 --restart 00003 ...
# without --first the --parallel yields: --equil 00001 --equil 00002 --equil 00003 ...
parser.add_argument('--first', action = 'store_true', default = None)

# how many simulations at the same time
# not yet implemented
parser.add_argument('-j', '--jobs', dest = 'numJobs', type = int, default = 1)

# collect all arguments
args = parser.parse_args()

num_gpus = args.g
num_nodes = args.N

ntasks_per_node = num_gpus * 2  # Adjust as needed
mem_per_gpu = 20  # Memory in GB per GPU, adjust as needed
total_mem = mem_per_gpu * num_gpus


os.system(f'cp parameters-default.{args.obj}.yaml parameters-default.yaml')

if(args.par == None):
    os.system('mkdir -p parameter')
    os.system('rm -r parameter/* 2>/dev/null')
    os.system(f'cp parameters-default.{args.obj}.yaml parameter/parameters-default00001.yaml')
    cnt = 1
    
else:
    tipi = [str, float, float, int]
    parsed = [[tipi[i](args.par[j][i]) for i in range(len(args.par[j]))] for j in range(len(args.par))]

    names = [parsed[i][0] for i in range(len(parsed))]
    starts = [parsed[i][1] for i in range(len(parsed))]
    stops = [parsed[i][2] for i in range(len(parsed))]
    widths = [parsed[i][3] for i in range(len(parsed))]
    depth = len(parsed) - 1

    def parameter_loop(depth, widths, starts, stops, names, parameters_default, cnt):
        width = widths[depth]
        start = starts[depth]
        stop = stops[depth]
        name = names[depth]
        step = (0 if width == 1 else (stop - start) / (width - 1))
        if(depth > 0):
            for i in range(width):
                new_val = start + i * step
                if(type(parameters_default[name]) == int):
                    parameters_default[name] = int(new_val)
                else:
                    parameters_default[name] = new_val
                cnt = parameter_loop(depth-1, widths, starts, stops, names, parameters_default, cnt)
        else:
            for i in range(width):
                new_val = start + i * step
                num = f'{cnt + 1 :05d}'
                cnt += 1
                if(type(parameters_default[name]) == int):
                    parameters_default[name] = int(new_val)
                else:
                    parameters_default[name] = new_val
                filename = f'parameter/parameters-default{num}.yaml'
                with open(filename, 'w') as f:
                    yaml.dump(parameters_default, f)
        return cnt

    filename_default = 'parameters-default.yaml'
    with open(filename_default, 'rb') as f:
        parameters_default = yaml.load(f, Loader = yaml.CLoader)

    os.system('mkdir -p parameter')
    os.system('rm -r parameter/* 2>/dev/null')

    cnt = 0
    cnt = parameter_loop(depth, widths, starts, stops, names, parameters_default, cnt)

def write_commands(filename, runscript, extra = ""):
	file_commands = open(filename, 'w')
	cnt0 = cnt
	cnt_sim = cnt
	cnt_par = cnt
	for i in range(cnt):
		num = f'{i + 1 :05d}'
		if(args.parallel):
			file_commands.write(f'bash {runscript} --equil {num}eq {extra}\n')
			os.system(f'cp parameter/parameters-default{num}.yaml parameter/parameters-default{num}eq.yaml')
			if(args.first):
				cnt_par += 1
				cnt_sim += 1
				file_commands.write(f'bash {runscript} --restart {num} {extra}\n')
		elif(i == 0 and (args.forward or args.hysteresis)):
			#file_commands.write(f'bash {runscript} --equil {num}eq {extra}\n')
			#os.system(f'cp parameter/parameters-default{num}.yaml parameter/parameters-default{num}eq.yaml')
			#print('here')
			if(args.first):
				cnt_par += 1
				cnt_sim += 1
				os.system(f'cp parameter/parameters-default{num}.yaml parameter/parameters-default{num}eq.yaml')
				file_commands.write(f'bash {runscript} --equil {num}eq {extra}\n')
				file_commands.write(f'bash {runscript} --restart {num} {extra}\n')
			else:
				print('here')
				file_commands.write(f'bash {runscript} --restart {num} {extra}\n')
		elif(i > 0 and (args.forward or args.hysteresis)):
			file_commands.write(f'bash {runscript} --restart {num} {extra}\n')
	if(args.hysteresis):
		for i in reversed(range(cnt - 1)):
			num = f'{i + 1 :05d}'
			sim = f'{cnt0 + cnt - 1 - i :05d}'
			os.system(f'cp parameter/parameters-default{num}.yaml parameter/parameters-default{sim}.yaml')
			cnt_par += 1
			cnt_sim += 1
			file_commands.write(f'bash {runscript} --restart {sim} {extra}\n')
	file_commands.close()
	return cnt_sim, cnt_par

cnt_sim, cnt_par = write_commands('commands.txt', 'run.sh', f'{2 * num_gpus}')

os.system(f'rm parameters-default.yaml')

print(f'Total number of generated parameter files = {cnt_par}')

print(f'Total number of simulations = {cnt_sim}')

file_vega = open('run_HPC.sbatch', 'w')
file_vega.write(f'''#!/bin/bash

#SBATCH --job-name="KeyserSoze"
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --partition=gpu
#SBATCH --mem={total_mem}GB
#SBATCH --output=output.out
#SBATCH --signal=INT@60

bash commands.txt
''')
