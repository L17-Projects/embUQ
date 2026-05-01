#! /usr/bin/env python3

import numpy as np
import yaml 
import argparse
import trimesh 

##########################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
parser.add_argument('--latex', action = 'store_true', default = None)
args = parser.parse_args()

filename_default = '../parameter/parameters-default00001.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

objFile = '../mesh/' + (parameters_default["objFile"])[0:-4] + args.simnum + '.off'

mesh = trimesh.load(objFile)
vertices = mesh.vertices
nicle = np.zeros(len(vertices))

file0 = open('xyz0.xyz', 'w')

file0.write(f'{len(vertices)}\n')
file0.write('# generated using trimesh\n')

nicle = np.zeros(len(vertices))

cols = np.column_stack((nicle, vertices[:,0], vertices[:,1], vertices[:,2]))

np.savetxt(file0, cols)

##########################
