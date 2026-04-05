#!/usr/bin/env python3

import os
import sys

import yaml


def _resolve_config_file() -> str:
    config_paths = [
        "../../inference/configs/production/inference_config_indentation.yaml",
        "inference/configs/production/inference_config_indentation.yaml",
        "../inference/configs/production/inference_config_indentation.yaml",
    ]
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_based_root = os.path.dirname(os.path.dirname(file_dir))
    config_paths.append(os.path.join(file_based_root, "inference/configs/production/inference_config_indentation.yaml"))
    for path in config_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find production indentation config in any of: {config_paths}")


def generate_sim(source_path, simu_path, par, obj, forward, hysteresis, parallel, g, N, first, numJobs):
    with open(_resolve_config_file(), "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    num_gpus = g
    num_nodes = N
    ntasks_per_node = num_gpus * 2
    mem_per_gpu = 20
    total_mem = mem_per_gpu * num_gpus
    os.system(f"cp {source_path}parameters-default.{obj}.yaml {simu_path}parameters-default.yaml")
    if par is None:
        os.system(f"mkdir -p {simu_path}parameter")
        os.system(f"rm -r {simu_path}parameter/* 2>/dev/null")
        os.system(f"cp {source_path}parameters-default.{obj}.yaml {simu_path}parameter/parameters-default00001.yaml")
        cnt = 1
    else:
        tipi = [str, float, float, int]
        parsed = [[tipi[i](par[j][i]) for i in range(len(par[j]))] for j in range(len(par))]
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
            step = 0 if width == 1 else (stop - start) / (width - 1)
            if depth > 0:
                for i in range(width):
                    new_val = start + i * step
                    parameters_default[name] = int(new_val) if isinstance(parameters_default[name], int) else new_val
                    cnt = parameter_loop(depth - 1, widths, starts, stops, names, parameters_default, cnt)
            else:
                for i in range(width):
                    new_val = start + i * step
                    num = f"{cnt + 1 :05d}"
                    cnt += 1
                    parameters_default[name] = int(new_val) if isinstance(parameters_default[name], int) else new_val
                    filename = simu_path + f"parameter/parameters-default{num}.yaml"
                    with open(filename, "w") as f:
                        yaml.dump(parameters_default, f)
            return cnt

        filename_default = simu_path + "parameters-default.yaml"
        with open(filename_default, "rb") as f:
            parameters_default = yaml.load(f, Loader=yaml.CLoader)
        os.system(f"mkdir -p {simu_path}parameter")
        os.system(f"rm -r {simu_path}parameter/* 2>/dev/null")
        cnt = parameter_loop(depth, widths, starts, stops, names, parameters_default, 0)

    def write_commands(filename, runscript, extra=""):
        file_commands = open(filename, "w")
        cnt0 = cnt
        cnt_sim = cnt
        cnt_par = cnt
        for i in range(cnt):
            num = f"{i + 1 :05d}"
            if parallel:
                file_commands.write(f"bash {runscript} --equil {num}eq {extra}\n")
                os.system(f"cp {simu_path}parameter/parameters-default{num}.yaml {simu_path}parameter/parameters-default{num}eq.yaml")
                if first:
                    cnt_par += 1
                    cnt_sim += 1
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
            elif i == 0 and (forward or hysteresis):
                if first:
                    cnt_par += 1
                    cnt_sim += 1
                    os.system(f"cp {source_path}parameter/parameters-default{num}.yaml {simu_path}parameter/parameters-default{num}eq.yaml")
                    file_commands.write(f"bash {runscript} --equil {num}eq {extra}\n")
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
                else:
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
            elif i > 0 and (forward or hysteresis):
                file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
        if hysteresis:
            for i in reversed(range(cnt - 1)):
                num = f"{i + 1 :05d}"
                sim = f"{cnt0 + cnt - 1 - i :05d}"
                os.system(f"cp {source_path}parameter/parameters-default{num}.yaml {simu_path}parameter/parameters-default{sim}.yaml")
                cnt_par += 1
                cnt_sim += 1
                file_commands.write(f"bash {runscript} --restart {sim} {extra}\n")
        file_commands.close()
        return cnt_sim, cnt_par

    cnt_sim, cnt_par = write_commands(f"{simu_path}commands.txt", "run.sh", f"{2 * num_gpus}")
    os.system(f"rm {simu_path}parameters-default.yaml")
    file_vega = open(simu_path + "run_HPC.sbatch", "w")
    file_vega.write(f'''#!/bin/bash

#SBATCH --job-name="KeyserSoze"
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --partition=gpu
#SBATCH --exclude=compute-2-1
#SBATCH --mem={total_mem}GB
#SBATCH --output=output.out
#SBATCH --signal=INT@60

bash commands.txt
''')


def main(argv):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-p", "--parameter", dest="par", action="append", nargs=4, default=None)
    parser.add_argument("-o", "--object", dest="obj", default=None)
    group_sweep = parser.add_mutually_exclusive_group(required=True)
    group_sweep.add_argument("--forward", action="store_true", default=None)
    group_sweep.add_argument("--hysteresis", action="store_true", default=None)
    group_sweep.add_argument("--parallel", action="store_true", default=None)
    parser.add_argument("-g", type=int, default=1)
    parser.add_argument("-N", type=int, default=1)
    parser.add_argument("--first", action="store_true", default=None)
    parser.add_argument("-j", "--jobs", dest="numJobs", type=int, default=1)
    args = parser.parse_args()
    generate_sim(source_path="", simu_path="", par=args.par, obj=args.obj, forward=args.forward, hysteresis=args.hysteresis, parallel=args.parallel, g=args.g, N=args.N, first=args.first, numJobs=args.numJobs)


if __name__ == "__main__":
    main(sys.argv[1:])
