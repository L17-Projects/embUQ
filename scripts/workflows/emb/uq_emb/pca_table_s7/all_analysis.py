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

filename_default = 'parameter/parameters-default00001.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)
filename = 'parameter/parameters00001.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)


file0 = 'xyz0.xyz' #'emb_0000014.xyz' #'emb_0000000.xyz'

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

#av = np.loadtxt('emb_0000014.xyz', skiprows=2)[:,1:]
av = np.loadtxt('output/ref1.xyz', skiprows=2)[:,1:]

for i in range(trajs):
    trj_np[i] = trj.trajectory[i].positions.reshape(1,-1) - av.reshape(1,-1)

###########################################################

print(f'Step:::::Calculating covariance matrix of size {trj_np.shape}.')
a = np.cov(trj_np.T)

# All membrane vertices have the same mass, so M^(1/2) C M^(1/2)
# is exactly mvert * C. Avoid materializing two dense diagonal matrices.
mvert = float(parameters["mvert"])
if not np.isfinite(mvert) or mvert <= 0.0:
    raise ValueError(f"mvert must be finite and positive, got {mvert!r}")
a *= mvert
###########################################################

method = 'svd' # 'eig', 'svd', 'mdanalysis'
print(f'Step:::::Calculating eigenvalues and eigenvectors using np.{method} method.')
if(method == 'eigh'):
    eigvalues, eigvectors = np.linalg.eigh(a)
elif(method == 'eig'):
    eigvalues, eigvectors = np.linalg.eig(a)
elif(method == 'svd'):
    U, eigvalues, Vh = np.linalg.svd(a)
elif(method == 'mdanalysis'):
    pc = pca.PCA(trj, select='all', mean=None, n_components=None).run()

###########################################################

nlim = -1
nlimt = ('all' if nlim == -1 else nlim)
print(f'Step:::::Storing {nlimt} eigenvalues and eigenvectors.')
selection = slice(None) if nlim == -1 else slice(0, nlim)

if(method == 'eigh'):
    np.savetxt('output/eigvalues.txt', eigvalues[selection].real)
    np.savetxt('output/eigvectors.txt', eigvectors.T.real[selection])
if(method == 'svd'):
    np.savetxt('output/eigvalues.txt', eigvalues[selection].real)
    np.savetxt('output/eigvectors.txt', U.T.real[selection])
if(method == 'eig'):
    idx = eigvalues.argsort()[::-1]   
    eigvalues = eigvalues[idx]
    eigvectors = eigvectors[:,idx]
    np.savetxt('output/eigvalues.txt', eigvalues[selection].real)
    np.savetxt('output/eigvectors.txt', eigvectors.T.real[selection])
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
