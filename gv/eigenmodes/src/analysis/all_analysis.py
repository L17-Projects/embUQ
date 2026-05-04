#! /usr/bin/env python3

import MDAnalysis as mda
import numpy as np
import os, fnmatch
import warnings
from MDAnalysis.analysis import pca
import yaml
from timeit import default_timer as timer

import argparse

##########################

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
parser.add_argument('--latex', action = 'store_true', default = None)
args = parser.parse_args()

start = timer()

warnings.filterwarnings('ignore')

filename_default = '../parameter/parameters-default00001.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)
filename = '../parameter/parameters00001.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)


file0 = 'emb_0000000.xyz'

print('Step:::::Loading trajectory output/positions.xyz')

trj = mda.Universe(file0, 'output/positions.xyz', format="XYZ", dt=1)

###########################################################

print('Step:::::Performing trajectory alignment using RMSD')

from MDAnalysis.analysis import align

ref = mda.Universe(file0, format="XYZ")

alignment = align.AlignTraj(trj, ref, in_memory=True, filename='output/rmsfit.xyz')

alignment.run()

print('Step:::::Calculating average from the aligned trajectory')

av = 0
trajs = len(trj.trajectory)
vertices = trj.trajectory[0].positions
num_vertices = len(vertices)
nicle = np.zeros(num_vertices)
for i in range(trajs):
    av += trj.trajectory[i].positions / trajs

f = open('output/ref1.xyz', 'w')
f.write(f'{num_vertices}\n')
f.write(f'# generated in MDAnalysis\n')
cols = np.column_stack((nicle, av))
np.savetxt(f, cols)
f.close()
###########################################################

print('Step:::::Subtracting average/reference structure from trajectory')
trj_np = np.zeros((trajs, 3 * num_vertices))

av = np.loadtxt('emb_0000000.xyz', skiprows=2)[:,1:]

for i in range(trajs):
    trj_np[i] = trj.trajectory[i].positions.reshape(1,-1) - av.reshape(1,-1)

###########################################################

print(f'Step:::::Preparing mass-weighted trajectory matrix of size {trj_np.shape}.')
trj_np *= np.sqrt(parameters["mvert"])
###########################################################

method = 'svd' # 'eig', 'svd', 'mdanalysis'
print(f'Step:::::Calculating eigenvalues and eigenvectors using np.{method} method.')
if(method == 'eigh'):
    a = np.cov(trj_np.T)
    eigvalues, eigvectors = np.linalg.eigh(a)
elif(method == 'eig'):
    a = np.cov(trj_np.T)
    eigvalues, eigvectors = np.linalg.eig(a)
elif(method == 'svd'):
    _, singular_values, Vh = np.linalg.svd(trj_np, full_matrices=False)
    denominator = max(trajs - 1, 1)
    eigvalues = singular_values**2 / denominator
    U = Vh.T
elif(method == 'mdanalysis'):
    pc = pca.PCA(trj, select='all', mean=None, n_components=None).run()

###########################################################

nlim = -1
nlimt = ('all' if nlim == -1 else nlim)
print(f'Step:::::Storing {nlimt} eigenvalues and eigenvectors.')

if(method == 'eigh'):
    np.savetxt('output/eigvalues.txt', eigvalues[0:nlim].real)
    np.savetxt('output/eigvectors.txt', eigvectors.T.real[0:nlim])
if(method == 'svd'):
    np.savetxt('output/eigvalues.txt', eigvalues[0:nlim].real)
    np.savetxt('output/eigvectors.txt', U.T.real[0:nlim])
if(method == 'eig'):
    idx = eigvalues.argsort()[::-1]
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]
    np.savetxt('output/eigvalues.txt', eigvalues[0:nlim].real)
    np.savetxt('output/eigvectors.txt', eigvectors.T.real[0:nlim])
end = timer()
print(f'\nDone in {end - start} seconds')


'''
###########################################################
print('Step:::::Calculating average/reference structure')
trajs = len(trj.trajectory)
vertices = trj.trajectory[0].positions
num_vertices = len(vertices)
nicle = np.zeros(num_vertices)
av = 0
for i in range(trajs):
    av += trj.trajectory[i].positions / trajs

f = open('output/ref.xyz', 'w')
f.write(f'{num_vertices}\n')
f.write(f'# generated in MDAnalysis\n')
cols = np.column_stack((nicle, av))
np.savetxt(f, cols)
f.close()'''

#trj.select_atoms('all').masses = 1.0

# for large number of files increase the number of files that can be opened
# ulimit -S -n 4096  #4096 is the HARD limit
#print(xyz_list)

#o364.vrana15
#160 s for 2562 and eigh
#274 s for 2562 and eig
#262 s for 2562 and svd

