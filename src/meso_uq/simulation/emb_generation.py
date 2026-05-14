from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from meso_uq.agents.emb.workflows import resolve_emb_config_file
from meso_uq.core import Modality


ShellCommand = Callable[[str], int]


@dataclass(frozen=True)
class ParameterSweep:
    name: str
    start: float
    stop: float
    width: int

    @property
    def step(self) -> float:
        return 0.0 if self.width == 1 else (self.stop - self.start) / (self.width - 1)

    def values(self) -> tuple[float, ...]:
        return tuple(self.start + i * self.step for i in range(self.width))


@dataclass(frozen=True)
class EmbGenerationResult:
    parameter_count: int
    simulation_count: int
    command_file: str
    sbatch_file: str
    config_file: str


def _yaml_loader() -> type[yaml.Loader]:
    return getattr(yaml, "CLoader", yaml.SafeLoader)


def _join_legacy_path(root: str, suffix: str) -> str:
    if not root:
        return suffix
    if root.endswith("/"):
        return f"{root}{suffix}"
    return f"{root}/{suffix}"


def parse_parameter_sweeps(par: Sequence[Sequence[Any]] | None) -> tuple[ParameterSweep, ...]:
    if par is None:
        return ()
    sweeps = []
    for item in par:
        if len(item) != 4:
            raise ValueError(f"Parameter sweep entries must have 4 fields: name, start, stop, width. Got: {item!r}")
        name, start, stop, width = item
        sweep = ParameterSweep(str(name), float(start), float(stop), int(width))
        if sweep.width < 1:
            raise ValueError(f"Parameter sweep width for '{sweep.name}' must be >= 1.")
        sweeps.append(sweep)
    return tuple(sweeps)


def _coerce_payload_value(defaults: Mapping[str, Any], name: str, value: float) -> int | float:
    return int(value) if isinstance(defaults[name], int) else value


def generate_parameter_payloads(
    parameters_default: Mapping[str, Any],
    sweeps: Sequence[ParameterSweep],
) -> Iterator[dict[str, Any]]:
    if not sweeps:
        yield dict(parameters_default)
        return

    current = dict(parameters_default)

    def loop(depth: int) -> Iterator[dict[str, Any]]:
        sweep = sweeps[depth]
        for value in sweep.values():
            current[sweep.name] = _coerce_payload_value(parameters_default, sweep.name, value)
            if depth > 0:
                yield from loop(depth - 1)
            else:
                yield dict(current)

    yield from loop(len(sweeps) - 1)


def write_emb_generation_commands(
    filename: str,
    *,
    runscript: str,
    count: int,
    source_path: str,
    simu_path: str,
    forward: bool | None,
    hysteresis: bool | None,
    parallel: bool | None,
    first: bool | None,
    extra: str = "",
    shell: ShellCommand | None = None,
) -> tuple[int, int]:
    run_shell = os.system if shell is None else shell
    cnt0 = count
    cnt_sim = count
    cnt_par = count
    with open(filename, "w", encoding="utf-8") as file_commands:
        for i in range(count):
            num = f"{i + 1 :05d}"
            if parallel:
                file_commands.write(f"bash {runscript} --equil {num}eq {extra}\n")
                run_shell(
                    f"cp {_join_legacy_path(simu_path, f'parameter/parameters-default{num}.yaml')} "
                    f"{_join_legacy_path(simu_path, f'parameter/parameters-default{num}eq.yaml')}"
                )
                if first:
                    cnt_par += 1
                    cnt_sim += 1
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
            elif i == 0 and (forward or hysteresis):
                if first:
                    cnt_par += 1
                    cnt_sim += 1
                    run_shell(
                        f"cp {_join_legacy_path(source_path, f'parameter/parameters-default{num}.yaml')} "
                        f"{_join_legacy_path(simu_path, f'parameter/parameters-default{num}eq.yaml')}"
                    )
                    file_commands.write(f"bash {runscript} --equil {num}eq {extra}\n")
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
                else:
                    file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
            elif i > 0 and (forward or hysteresis):
                file_commands.write(f"bash {runscript} --restart {num} {extra}\n")
        if hysteresis:
            for i in reversed(range(count - 1)):
                num = f"{i + 1 :05d}"
                sim = f"{cnt0 + count - 1 - i :05d}"
                run_shell(
                    f"cp {_join_legacy_path(source_path, f'parameter/parameters-default{num}.yaml')} "
                    f"{_join_legacy_path(simu_path, f'parameter/parameters-default{sim}.yaml')}"
                )
                cnt_par += 1
                cnt_sim += 1
                file_commands.write(f"bash {runscript} --restart {sim} {extra}\n")
    return cnt_sim, cnt_par


def build_emb_hpc_sbatch_text(*, num_gpus: int, num_nodes: int, ntasks_per_node: int, total_mem: int) -> str:
    return f'''#!/bin/bash

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
'''


def generate_emb_simulation(
    *,
    modality: Modality | str,
    source_path: str,
    simu_path: str,
    par: Sequence[Sequence[Any]] | None,
    obj: str | None,
    forward: bool | None,
    hysteresis: bool | None,
    parallel: bool | None,
    g: int,
    N: int,
    first: bool | None,
    numJobs: int,
    config_file: str | Path | None = None,
    anchor_file: str | Path | None = None,
    shell: ShellCommand | None = None,
    debug_script_file: str | Path | None = None,
    debug_function_name: str = "generate_sim",
) -> EmbGenerationResult:
    del numJobs
    selected_modality = Modality(modality)
    resolved_config = (
        str(config_file)
        if config_file is not None
        else resolve_emb_config_file(selected_modality, purpose="generation", anchor_file=anchor_file)
    )
    with open(resolved_config, "rb") as f:
        config = yaml.load(f, Loader=_yaml_loader())
    if debug_script_file is not None and config.get("debug", 0) >= 1:
        print(f"Running from function {debug_function_name} in script {debug_script_file}")

    run_shell = os.system if shell is None else shell
    num_gpus = g
    num_nodes = N
    ntasks_per_node = num_gpus * 2
    mem_per_gpu = 20
    total_mem = mem_per_gpu * num_gpus
    object_name = obj or "emb"

    run_shell(
        f"cp {_join_legacy_path(source_path, f'parameters-default.{object_name}.yaml')} "
        f"{_join_legacy_path(simu_path, 'parameters-default.yaml')}"
    )

    if par is None:
        run_shell(f"mkdir -p {_join_legacy_path(simu_path, 'parameter')}")
        run_shell(f"rm -r {_join_legacy_path(simu_path, 'parameter/*')} 2>/dev/null")
        run_shell(
            f"cp {_join_legacy_path(source_path, f'parameters-default.{object_name}.yaml')} "
            f"{_join_legacy_path(simu_path, 'parameter/parameters-default00001.yaml')}"
        )
        count = 1
    else:
        sweeps = parse_parameter_sweeps(par)
        filename_default = _join_legacy_path(simu_path, "parameters-default.yaml")
        with open(filename_default, "rb") as f:
            parameters_default = yaml.load(f, Loader=_yaml_loader())
        run_shell(f"mkdir -p {_join_legacy_path(simu_path, 'parameter')}")
        run_shell(f"rm -r {_join_legacy_path(simu_path, 'parameter/*')} 2>/dev/null")
        count = 0
        for count, payload in enumerate(generate_parameter_payloads(parameters_default, sweeps), start=1):
            num = f"{count :05d}"
            filename = _join_legacy_path(simu_path, f"parameter/parameters-default{num}.yaml")
            with open(filename, "w", encoding="utf-8") as f:
                yaml.dump(payload, f)

    command_file = _join_legacy_path(simu_path, "commands.txt")
    cnt_sim, cnt_par = write_emb_generation_commands(
        command_file,
        runscript="run.sh",
        count=count,
        source_path=source_path,
        simu_path=simu_path,
        forward=forward,
        hysteresis=hysteresis,
        parallel=parallel,
        first=first,
        extra=f"{2 * num_gpus}",
        shell=run_shell,
    )
    run_shell(f"rm {_join_legacy_path(simu_path, 'parameters-default.yaml')}")
    sbatch_file = _join_legacy_path(simu_path, "run_HPC.sbatch")
    with open(sbatch_file, "w", encoding="utf-8") as file_hpc:
        file_hpc.write(
            build_emb_hpc_sbatch_text(
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                ntasks_per_node=ntasks_per_node,
                total_mem=total_mem,
            )
        )
    return EmbGenerationResult(
        parameter_count=cnt_par,
        simulation_count=cnt_sim,
        command_file=command_file,
        sbatch_file=sbatch_file,
        config_file=resolved_config,
    )


__all__ = [
    "EmbGenerationResult",
    "ParameterSweep",
    "build_emb_hpc_sbatch_text",
    "generate_emb_simulation",
    "generate_parameter_payloads",
    "parse_parameter_sweeps",
    "write_emb_generation_commands",
]
