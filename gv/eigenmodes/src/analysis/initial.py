#! /usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
import yaml

##########################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest='simnum', default='00001')
parser.add_argument('--latex', action='store_true', default=None)
args = parser.parse_args()

filename_default = Path('../parameter/parameters-default00001.yaml')
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader=yaml.CLoader)


def _first_existing_reference(simnum):
    roots = (
        Path('../trj_eq') / f'sim{simnum}eq',
        Path('../trj_eq') / f'sim{simnum}',
    )
    for root in roots:
        exact = root / 'emb_0000000.xyz'
        if exact.is_file():
            return exact
        candidates = sorted(root.glob('emb_*.xyz'))
        if candidates:
            return candidates[0]
    return None


def _box_centered_vertices():
    obj_file = Path('../mesh') / ((parameters_default['objFile'])[0:-4] + args.simnum + '.off')
    mesh = trimesh.load(obj_file)
    vertices = np.asarray(mesh.vertices, dtype=float)
    domain_center = np.asarray(
        [
            0.5 * float(parameters_default['Lx']),
            0.5 * float(parameters_default['Ly']),
            0.5 * float(parameters_default['Lz']),
        ],
        dtype=float,
    )
    if np.linalg.norm(np.mean(vertices, axis=0)) < 0.25 * float(np.max(domain_center)):
        vertices = vertices + domain_center
    return vertices


def _write_xyz(path, vertices):
    nicle = np.zeros(len(vertices))
    with open(path, 'w') as file0:
        file0.write(f'{len(vertices)}\n')
        file0.write('# generated using eigenmodes analysis reference materialization\n')
        np.savetxt(file0, np.column_stack((nicle, vertices[:, 0], vertices[:, 1], vertices[:, 2])))


reference = _first_existing_reference(args.simnum)
target = Path('emb_0000000.xyz')
if reference is not None:
    shutil.copy2(reference, target)
    print(f'Using equilibrated eigenmode reference frame: {reference}')
else:
    _write_xyz(target, _box_centered_vertices())
    print('Using box-centered mesh fallback for eigenmode reference frame.')
