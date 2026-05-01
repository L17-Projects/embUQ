#! /usr/bin/env python3

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from argparse import ArgumentParser
import MDAnalysis as mda
import warnings
import os
import yaml

warnings.warn('ignore')
warnings.filterwarnings('ignore')

from matplotlib import animation
## Mode number to plot: python3 plot_modes.py -m 1

parser = ArgumentParser()       
parser.add_argument('-m', '--mode', dest = 'mode', type = int, default = 0)
parser.add_argument('-s', '--scale', dest = 'scale', type = float, default = 10)

parser.add_argument('-p', '--parameter', dest = 'par', action = 'append', nargs = 4, default = None)
parser.add_argument('-r', '--range', dest = 'range', action = 'append', nargs = 2, default = None)

parser.add_argument('--animation', action = 'store_true', default = None)
parser.add_argument('--surface', action = 'store_true', default = None)

parser.add_argument('--simnum', dest = 'simnum', default = '00001')

args = parser.parse_args()

print(args.range)

filename_default = 'parameters-default' + args.simnum + '.yaml'

with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

objFile = parameters_default["objFile"]
   
mmin = 0
mmax = 1
if(args.range):
    tipi = [int, int]
    parsed = [tipi[i](args.range[0][i]) for i in range(len(args.range[0]))]
    mmin = int(parsed[0])
    mmax = int(parsed[1]) + 1
elif(args.mode):
    mmin = args.mode
    mmax = args.mode + 1
else:
    print('error. specify mode or range')


ref0 = trimesh.load(objFile).vertices

## Load eigenvectors

eigvectors = np.loadtxt('eigvectors_new.txt')

print(eigvectors.shape)
m = mmin

while(m < mmax):
    vectors = np.reshape(eigvectors[m], (-1,3))
    
    #print(f'vsota = {np.sum(vectors)}')
    
    dvol = 0
    
    print(f'volume change = {np.sum(vectors * ref0)}')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    scale = args.scale
    
    x = ref0[:,0]
    y = ref0[:,1]
    z = ref0[:,2]
    
    u = scale * vectors[:,0]
    v = scale * vectors[:,1]
    w = scale * vectors[:,2]
    
    color_list = []
    
    # stupid matplotlib stuff
    for i in range(len(x)):
        if( (x[i] - np.mean(x))*u[i] + (y[i] - np.mean(y))*v[i] + (z[i] - np.mean(z))*w[i] > 0):
            color_list.append('blue')
        else:
            color_list.append('red')
    color_tmp = np.repeat(color_list, 2)
    color_list.extend(color_tmp)
    #
    
    ax.view_init(5, 35)
    quiver = ax.quiver(x, y, z, u, v, w,  colors = color_list)
    om0 = 4.0
    
    ln = 7.5
    
    ax.set_xlim(-ln, ln)
    ax.set_ylim(-ln, ln)
    ax.set_zlim(-ln, ln)
    def update(t):
        global quiver
        quiver.remove()
        
        xx = x +  np.sin(om0 * t) * u
        yy = y +  np.sin(om0 * t) * v
        zz = z +  np.sin(om0 * t) * w
        
        
        uu = u * np.sin(om0 * t)
        vv = v * np.sin(om0 * t)
        ww = w * np.sin(om0 * t)
        quiver = ax.quiver(x, y, z, uu, vv, ww,  colors = color_list)
    
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
    if(args.range):
        plt.savefig(f'mode_{m}.png')
        plt.clf()
    elif(args.animation):
        anim = animation.FuncAnimation(fig, update, frames = np.linspace(0,10,100), interval = 100)
        fig.tight_layout()
        plt.show()
    elif(args.surface):
        ax.plot_trisurf(x, y, z, cmap='Accent')
        plt.show()
    m += 1

#plt.show()
