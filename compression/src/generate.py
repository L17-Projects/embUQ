#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.agents.emb import resolve_emb_config_file
from meso_uq.core import Modality
from meso_uq.simulation import generate_emb_simulation


def _resolve_config_file() -> str:
    return resolve_emb_config_file(Modality.COMPRESSION, purpose="generation", anchor_file=__file__)


def generate_sim(source_path, simu_path, par, obj, forward, hysteresis, parallel, g, N, first, numJobs):
    return generate_emb_simulation(
        modality=Modality.COMPRESSION,
        source_path=source_path,
        simu_path=simu_path,
        par=par,
        obj=obj,
        forward=forward,
        hysteresis=hysteresis,
        parallel=parallel,
        g=g,
        N=N,
        first=first,
        numJobs=numJobs,
        config_file=_resolve_config_file(),
        shell=os.system,
        debug_script_file=__file__,
        debug_function_name="generate_sim",
    )


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
    generate_sim(
        source_path="",
        simu_path="",
        par=args.par,
        obj=args.obj,
        forward=args.forward,
        hysteresis=args.hysteresis,
        parallel=args.parallel,
        g=args.g,
        N=args.N,
        first=args.first,
        numJobs=args.numJobs,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
