#!/usr/bin/env python3

import os
import shlex
import yaml
from argparse import ArgumentParser
from pathlib import Path

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

mem_per_gpu = 20  # Memory in GB per GPU, adjust as needed
total_mem = mem_per_gpu * num_gpus


os.system(f'cp parameters-default.{args.obj}.yaml parameters-default.yaml')

_PAPER_EXACT_DEFAULT_OVERRIDES = {
    'dt': 0.0001,
    'dt_eq': 0.0001,
    'gamma_dpd': 16.3,
    'gamma_dpd_gas': 3.0,
    'numsteps': 40000000,
    'numsteps_eq': 500000,
    's_g': 0.125,
    'stslik': 200000,
    'stslik_eq': 100,
}
_CANARY_DEFAULT_OVERRIDES = {
    **_PAPER_EXACT_DEFAULT_OVERRIDES,
    'numsteps': 4000000,
    'numsteps_eq': 50000,
    'stslik': 20000,
}
_PAPER_EXACT_ENV_OVERRIDES = {
    'numsteps': 'MESOUQ_GV_EIGENMODES_NUMSTEPS',
    'numsteps_eq': 'MESOUQ_GV_EIGENMODES_NUMSTEPS_EQ',
    'stslik': 'MESOUQ_GV_EIGENMODES_STSLIK',
    'stslik_eq': 'MESOUQ_GV_EIGENMODES_STSLIK_EQ',
}
_PROFILE_DEFAULTS = {
    'paper': {
        'label': 'paper',
        'parameter_overrides': _PAPER_EXACT_DEFAULT_OVERRIDES,
        'mode_window_policy': 'frequency-min',
        'mode_min_frequency': '22.5',
    },
    'canary': {
        'label': 'canary',
        'parameter_overrides': _CANARY_DEFAULT_OVERRIDES,
        'mode_window_policy': 'frequency-min',
        'mode_min_frequency': '22.5',
    },
}
_DEFAULT_MODE_COUNT = 30


def _parse_domain_ranks(raw_value):
    text = str(raw_value).strip()
    if not text:
        raise ValueError('MESOUQ_GV_EIGENMODES_DOMAIN_RANKS must not be empty.')
    tokens = text.replace('x', ',').replace('X', ',').split(',')
    if len(tokens) == 1:
        tokens = text.split()
    if len(tokens) != 3:
        raise ValueError(
            'MESOUQ_GV_EIGENMODES_DOMAIN_RANKS must have three positive integers, '
            "for example '1,1,1' or '2x1x1'."
        )
    try:
        ranks = tuple(int(token.strip()) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            f'Invalid MESOUQ_GV_EIGENMODES_DOMAIN_RANKS={raw_value!r}; '
            'expected three positive integers.'
        ) from exc
    if any(rank <= 0 for rank in ranks):
        raise ValueError(
            f'Invalid MESOUQ_GV_EIGENMODES_DOMAIN_RANKS={raw_value!r}; each rank must be positive.'
        )
    return ranks


def _format_domain_ranks(raw_value):
    return ','.join(str(rank) for rank in _parse_domain_ranks(raw_value))


def _domain_rank_product(raw_value):
    product = 1
    for rank in _parse_domain_ranks(raw_value):
        product *= rank
    return product


def _default_mpi_ranks(raw_domain_ranks):
    product = _domain_rank_product(raw_domain_ranks)
    return 2 * product


domain_ranks = _format_domain_ranks(
    os.environ.get('MESOUQ_GV_EIGENMODES_DOMAIN_RANKS', '1,1,1')
)
num_mpi_ranks = int(
    os.environ.get('MESOUQ_GV_EIGENMODES_MPI_RANKS', str(_default_mpi_ranks(domain_ranks)))
)
if num_mpi_ranks <= 0:
    raise ValueError('MESOUQ_GV_EIGENMODES_MPI_RANKS must be a positive integer.')


def _paper_exact_enabled():
    return os.environ.get('MESOUQ_GV_PAPER_EXACT', '').lower() in {'1', 'true', 'yes'}


def _runtime_profile_name():
    raw_value = os.environ.get('MESOUQ_GV_EIGENMODES_PROFILE', '').strip().lower()
    if not raw_value:
        return 'paper' if _paper_exact_enabled() else ''
    if raw_value not in _PROFILE_DEFAULTS:
        raise ValueError(
            f"Unsupported MESOUQ_GV_EIGENMODES_PROFILE={raw_value!r}; "
            f"expected one of {sorted(_PROFILE_DEFAULTS)}."
        )
    return raw_value


runtime_profile_name = _runtime_profile_name()


def _apply_paper_exact_defaults(parameters_default):
    if not runtime_profile_name:
        return parameters_default
    resolved = dict(parameters_default)
    resolved.update(_PROFILE_DEFAULTS[runtime_profile_name]['parameter_overrides'])
    for key, env_name in _PAPER_EXACT_ENV_OVERRIDES.items():
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f'{env_name} must be a positive integer.')
        resolved[key] = value
    return resolved


def _write_parameters_default(filename, parameters_default):
    with open(filename, 'w') as f:
        yaml.dump(parameters_default, f)


def _mode_window_policy():
    explicit = os.environ.get('MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY', '').strip()
    if explicit:
        return explicit
    if runtime_profile_name:
        return str(_PROFILE_DEFAULTS[runtime_profile_name]['mode_window_policy'])
    return 'frequency-min'


def _mode_min_frequency():
    explicit = os.environ.get('MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY', '').strip()
    if explicit:
        return explicit
    if runtime_profile_name:
        return str(_PROFILE_DEFAULTS[runtime_profile_name]['mode_min_frequency'])
    return '22.5'


def _mode_count():
    raw_value = os.environ.get('MESOUQ_GV_EIGENMODES_MODE_COUNT', str(_DEFAULT_MODE_COUNT))
    value = int(raw_value)
    if value <= 0:
        raise ValueError('MESOUQ_GV_EIGENMODES_MODE_COUNT must be a positive integer.')
    return value


def _write_runtime_profile_manifest():
    mode_count = _mode_count()
    payload = {
        'schema': 'mesouq.gv.eigenmodes.runtime_profile.v1',
        'profile': runtime_profile_name or 'unprofiled',
        'paper_exact_requested': _paper_exact_enabled(),
        'parameter_profile': runtime_profile_name or 'none',
        'numsteps': None,
        'numsteps_eq': None,
        'stslik': None,
        'stslik_eq': None,
        'mode_window_policy': _mode_window_policy(),
        'mode_min_frequency_tau_inv': float(_mode_min_frequency()),
        'configured_final_mode_count': mode_count,
        'selected_paper_mode_indices': list(range(mode_count)),
        'domain_ranks': domain_ranks,
    }
    filename = 'parameter/parameters-default00001.yaml'
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            parameters_default = yaml.load(f, Loader=yaml.CLoader)
        for key in ('numsteps', 'numsteps_eq', 'stslik', 'stslik_eq'):
            payload[key] = int(parameters_default[key])
    with open('parameter/eigenmodes_runtime_profile.json', 'w') as f:
        import json

        json.dump(payload, f, indent=2, sort_keys=True)


def _find_repo_root():
    env_root = os.environ.get('MESOUQ_REPO_ROOT')
    if env_root:
        return Path(env_root).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file():
            return parent
    return None


def _gv_env_script():
    env_script = os.environ.get('MESOUQ_GV_ENV_SCRIPT', '').strip()
    if env_script:
        return str(Path(env_script).expanduser().resolve())
    runtime_root = os.environ.get('MESOUQ_SITE_RUNTIME_ROOT', '').strip()
    if runtime_root:
        return str((Path(runtime_root).expanduser() / 'env' / 'env.sh').resolve())
    repo_root = _find_repo_root()
    if repo_root is None:
        return ''

    legacy_site_env = 'HPC' + '_SITE'
    if os.environ.get(legacy_site_env):
        raise RuntimeError(f'{legacy_site_env} is no longer supported; use MESOUQ_SITE.')
    site = (os.environ.get('MESOUQ_SITE', '') or 'vega').strip().lower()
    if site not in {'vega', 'karolina'}:
        site = 'vega'
    return str((repo_root / f'_{site}' / 'env' / 'env.sh').resolve())
def _write_runtime_preamble(file_commands):
    env_script = _gv_env_script()
    if env_script:
        file_commands.write(f'if [[ -f {shlex.quote(env_script)} ]]; then\n')
        file_commands.write(f'  source {shlex.quote(env_script)}\n')
        file_commands.write('fi\n')
    material_overrides = os.environ.get('MESOUQ_GV_MATERIAL_OVERRIDES_JSON', '')
    if material_overrides:
        file_commands.write(
            'export MESOUQ_GV_MATERIAL_OVERRIDES_JSON='
            f'{shlex.quote(material_overrides)}\n'
        )
    mirheo_module = os.environ.get('MESOUQ_GV_MIRHEO_MODULE', '')
    if mirheo_module:
        file_commands.write(
            'export MESOUQ_GV_MIRHEO_MODULE='
            f'{shlex.quote(mirheo_module)}\n'
        )
    paper_exact = os.environ.get('MESOUQ_GV_PAPER_EXACT', '')
    if paper_exact:
        file_commands.write(
            'export MESOUQ_GV_PAPER_EXACT='
            f'{shlex.quote(paper_exact)}\n'
        )
    for env_name in _PAPER_EXACT_ENV_OVERRIDES.values():
        raw_value = os.environ.get(env_name, '')
        if raw_value:
            file_commands.write(f'export {env_name}={shlex.quote(raw_value)}\n')
    if runtime_profile_name:
        file_commands.write(
            'export MESOUQ_GV_EIGENMODES_PROFILE='
            f'{shlex.quote(runtime_profile_name)}\n'
        )
    file_commands.write(
        'export MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY='
        f'{shlex.quote(_mode_window_policy())}\n'
    )
    file_commands.write(
        'export MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY='
        f'{shlex.quote(_mode_min_frequency())}\n'
    )
    file_commands.write(
        'export MESOUQ_GV_EIGENMODES_DOMAIN_RANKS='
        f'{shlex.quote(domain_ranks)}\n'
    )
    file_commands.write('\n')

if(args.par == None):
    os.system('mkdir -p parameter')
    os.system('rm -r parameter/* 2>/dev/null')
    os.system(f'cp parameters-default.{args.obj}.yaml parameter/parameters-default00001.yaml')
    if runtime_profile_name:
        filename = 'parameter/parameters-default00001.yaml'
        with open(filename, 'rb') as f:
            parameters_default = yaml.load(f, Loader=yaml.CLoader)
        _write_parameters_default(filename, _apply_paper_exact_defaults(parameters_default))
    _write_runtime_profile_manifest()
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
    parameters_default = _apply_paper_exact_defaults(parameters_default)

    os.system('mkdir -p parameter')
    os.system('rm -r parameter/* 2>/dev/null')

    cnt = 0
    cnt = parameter_loop(depth, widths, starts, stops, names, parameters_default, cnt)
    _write_runtime_profile_manifest()

def write_commands(filename, runscript, extra = ""):
	file_commands = open(filename, 'w')
	_write_runtime_preamble(file_commands)
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

cnt_sim, cnt_par = write_commands('commands.txt', 'run.sh', f'{num_mpi_ranks} {domain_ranks}')

os.system(f'rm parameters-default.yaml')

print(f'Total number of generated parameter files = {cnt_par}')

print(f'Total number of simulations = {cnt_sim}')

file_vega = open('run_HPC.sbatch', 'w')
file_vega.write(f'''#!/bin/bash

#SBATCH --job-name="KeyserSoze"
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_mpi_ranks}
#SBATCH --partition=gpu
#SBATCH --mem={total_mem}GB
#SBATCH --output=output.out
#SBATCH --signal=INT@60

module --force purge >/dev/null 2>&1 || true
module load Python/3.10.8-GCCcore-12.2.0
module load OpenMPI/4.1.4-GCC-12.2.0
module load CUDA/12.2.2
module load GSL/2.7-GCC-12.2.0
module load Eigen/3.4.0-GCCcore-12.2.0
module load HDF5/1.14.0-gompi-2022b
module load MPFR/4.2.0-GCCcore-12.2.0
module load GMP/6.2.1-GCCcore-12.2.0

if [[ -f {shlex.quote(_gv_env_script())} ]]; then
  source {shlex.quote(_gv_env_script())}
fi

bash commands.txt
''')
