#!/usr/bin/env python3

import numpy as np
import argparse
import yaml
import os

######################################################
# set-up input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
args = parser.parse_args()

######################################################
# read parameters

filename = 'parameter/parameters' + args.simnum + '.yaml'
filename_prms = 'parameter/parameters.prms' + args.simnum + '.yaml'
filename_obmd = 'parameter/parameters.obmd' + args.simnum + '.yaml'
filename_default = 'parameter/parameters-default' + args.simnum + '.yaml'

with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

######################################################
# define variables

rho_water = parameters_default["rho_water"]
rho_gas = parameters_default["rho_gas"]
rhow = parameters_default["rhow"]
rhog = parameters_default["rhog"]
energyFactor = parameters_default["energyFactor"]
kbol = parameters_default["kbol"]
t0 = parameters_default["t0"]
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]

######################################################
# fundamental scales

ul = parameters_default["ul"]
ue = energyFactor * kbol * t0
um = rho_water * ul**3 / rhow
ut = np.sqrt(um * ul**2 / ue)
kbt = 1 / energyFactor
uvis = um / ul / ut

visw = parameters_default["visw"] #8.9e-4
visg = parameters_default["visg"]
visw_dpd = visw / uvis
visg_dpd = visg / uvis

######################################################
# masses

mw = rho_water * ul**3 / (rhow * um)
mg = rho_gas * ul**3 / (rhog * um)

######################################################
# print out

print(f'unit of time = {ut}')
print(f'unit of length = {ul}')
print(f'unit of energy = {ue}')
print(f'unit of mass = {um}')

######################################################
# object details

import trimesh

objType = (parameters_default["objFile"])[0:-4]
os.makedirs("mesh", exist_ok=True)

if(objType == 'gv'):
    os.system(f'''
    cd gas_vesicle
    bash run.sh {args.simnum}
    cd ..
    cp gas_vesicle/gv.off mesh/gv{args.simnum}.off
    ''')

else:
    os.system(f'''
    cd microbubble
    python3 sphere_icosphere.py --simnum {args.simnum} -o ../mesh/emb{args.simnum}.off -s {int(parameters_default["subDiv"])} -r {parameters_default["radp"]}
    cd ..
    ''')

objFile = 'mesh/' + objType + args.simnum + '.off'
mesh = trimesh.load(objFile)
nverts = len(mesh.vertices)
rho_shell = parameters_default["rho_shell"]

shell_th = parameters_default["th_fac"] * parameters_default["shell_th"]

tot_area = mesh.area
tot_volume = np.abs(mesh.volume)
mvert = rho_shell * shell_th * tot_area * ul**2 / um / nverts

######################################################
# elasticity

ka_tot = parameters_default["ka_tot"]
kv_tot = parameters_default["kv_tot"]
gammaC = parameters_default["gammaC"]
kBT = kbt

# engineering constants (Young's moduli, Poisson's ratio, ...)
fscale = parameters_default["fscale"]
Yt = parameters_default["yt_fac"] * parameters_default["Yt"]
Yl = parameters_default["Yl"]
nu = parameters_default["nu"]
s = parameters_default["s"]
s_g = parameters_default["s_g"]
rc = parameters_default["rc"]

k_fsi = parameters_default["k_fsi"]

rho_surf = nverts / tot_area
gamma_fsi = fscale * 2.0 * visw_dpd * (2 * k_fsi + 1) * (2 * k_fsi + 2) * (2 * k_fsi + 3) * (2 * k_fsi + 4) / (3 * np.pi * rc **4 * rhow * rho_surf)
gamma_fsi_gas = 1.23 * fscale * 2.0 * visg_dpd * (2 * k_fsi + 1) * (2 * k_fsi + 2) * (2 * k_fsi + 3) * (2 * k_fsi + 4) / (3 * np.pi * rc **4 * rhog * rho_surf)

# Lim Isotropic
ka = fscale * Yt * shell_th / (2 * (1 - nu)) / (ue / ul**2)
mu = fscale * Yt * shell_th / (2 * (1 + nu)) / (ue / ul**2)
a3 = parameters_default["a3"]
a4 = parameters_default["a4"]
b1 = parameters_default["b1"]
b2 = parameters_default["b2"]
radp = parameters_default["radp"]
radGV = parameters_default["radGV"]
height = parameters_default["height"]

# Kantor
kb_fac = parameters_default["kb_fac"]
kb = kb_fac * fscale * 2.0 / np.sqrt(3) * Yt * shell_th**3 / (12 * (1 - nu**2)) / ue
theta = parameters_default["theta"]

######################################################
# write prms_list
prms_gv = dict()

gammaC = 2 * parameters_default["gamma_dpd"]

if(objType == 'emb'):
    print('You are using: Microbubbles')
    prms_list = ["ka_tot", "kv_tot", "gammaC", "kBT", "tot_area", "tot_volume", "kb", "ka", "mu", "a3", "a4", "b1", "b2"]
    for var in prms_list:
        prms_gv[var] = float(eval(var))

# Lim Orthotropic
ka = fscale * Yt * Yl * shell_th * (1 + nu) / (2 * (Yl - nu**2 * Yt)) / (ue / ul**2)
mu = fscale * Yt * Yl * shell_th * (1 - nu) / (2 * (Yl - nu**2 * Yt)) / (ue / ul**2)
muL = np.abs(Yt * Yl * (1 - nu) / (2 * (Yl - nu**2 * Yt)))
c = fscale * (Yl**2 + Yt * Yl + 4 * Yt * muL * nu**2 - 4 * Yl * muL  - 2 * Yt * Yl * nu) * shell_th / (Yl - nu**2 * Yt) / (ue / ul**2)
muL = mu

def find_tips(mesh):
    r = mesh.vertices
    zmax = np.argmax(mesh.vertices[:,2])
    zmin = np.argmin(mesh.vertices[:,2])
    return zmin, zmax

tip1, tip2 = find_tips(mesh)

print('tips1 = ', tip1, tip2)

if(objType == 'gv'):
    print('You are using: Gas vesicles')
    prms_list = ["ka_tot", "kv_tot", "gammaC", "kBT", "tot_area", "tot_volume", "kb", "ka", "mu", "a3", "a4", "b1", "b2", "c", "muL", "tip1", "tip2"]
    for var in prms_list:
        prms_gv[var] = float(eval(var))
    prms_gv["tip1"] = int(tip1)
    prms_gv["tip2"] = int(tip2)

# dump parameters back to parameters.prms.yaml
with open(filename_prms, 'w') as f:
    yaml.dump(prms_gv, f)

######################################################
# random positions and orientations of objects

from scipy.stats import qmc

numObjects = parameters_default["numObjects"]
m = int(np.log2(numObjects))
sampler = qmc.Sobol(d = 3, scramble = True)
sample = sampler.random_base2(m = m)

fac = 0.6
sample[:,0] = 0.5 * (1 - fac) * Lx + sample[:,0] * fac * Lx
sample[:,1] = 0.5 * (1 - fac) * Ly + sample[:,1] * fac * Ly
sample[:,2] = 0.5 * (1 - fac) * Lz + sample[:,2] * fac * Lz

pos_q = []
rand_rot = True

rtol = 2.0 * parameters_default["radp"]

def testDistance(sample, rtol):
    flag = False
    for i in range(numObjects):
        for j in range(numObjects):
            rel = sample[j] - sample[i]
            r = np.linalg.norm(rel)
            if(i is not j and r < rtol):
                sample[i] -= rel * 0.5 * (1.03 * rtol - r)
                sample[j] += rel * 0.5 * (1.03 * rtol - r)
                flag = True
    if(flag):
        return sample, False
    else:
        return sample, True

ok = False
while(not ok):
    sample, ok = testDistance(sample, rtol)

x = sample[:,0]
y = sample[:,1]
z = sample[:,2]

custom_quat = False
if(len(x) == 1):
    pos_q = [[0.5 * Lx, 0.5 * Ly, 0.5 * Lz, 1, 0, 0, 0]]
    if(custom_quat):
        # this is for z to x rotation
        axis = [0, 1, 0]
        angle = np.pi / 2
        pos_q = [[0.5 * Lx, 0.5 * Ly, 0.5 * Lz, np.cos(angle / 2), np.sin(angle / 2) * axis[0], np.sin(angle / 2) * axis[1], np.sin(angle / 2) * axis[2]]]
        print(pos_q)
else:
    for i in range(len(x)):
        u = np.random.random()
        v = np.random.random()
        w = np.random.random()
        quatr = [1.0, 0.0, 0.0, 0.0]
        if(rand_rot):
            quatr = [np.sqrt(1 - u) * np.sin(2 * np.pi * v), np.sqrt(1 - u) * np.cos(2 * np.pi * v), np.sqrt(u) * np.sin(2 * np.pi * w), np.sqrt(u) * np.cos(2 * np.pi * w)]
        pos_q.append([x[i], y[i], z[i], quatr[0], quatr[1], quatr[2], quatr[3]])

######################################################
# write computed parameters to parameters.yaml

np.savetxt('posq.txt', pos_q)

nevery = int(parameters_default["numsteps"] / parameters_default["stslik"])
nevery_eq = int(parameters_default["numsteps_eq"] / parameters_default["stslik_eq"])

parameters = dict()

parameters.update({
"ut": float(ut),
"ue": float(ue),
"um": float(um),
"uvis": float(uvis),
"visw_dpd": float(visw_dpd),
"visg_dpd": float(visg_dpd),
"kbt": float(kbt),
"nevery": int(nevery),
"nevery_eq": int(nevery_eq),
"nverts": int(nverts),
"mw": float(mw),
"mg": float(mg),
"mvert": float(mvert),
"tot_area": float(tot_area),
"tot_volume": float(tot_volume),
"gamma_fsi": float(gamma_fsi),
"gamma_fsi_gas": float(gamma_fsi_gas),
"ka": float(ka),
"mu": float(mu),
"muL": float(muL),
"kb": float(kb),
"c": float(c)
})

# dump parameters back to parameters.yaml
with open(filename, 'w') as f:
    yaml.dump(parameters, f)
