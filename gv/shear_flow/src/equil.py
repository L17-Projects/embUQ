#!/usr/bin/env python3

import mirheoOBMD as mir
import numpy as np
import trimesh
import yaml
import argparse
import os

######################################################
# set-up simulation type: equilibration or restart

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--equil', action = 'store_true', default = None)
group.add_argument('--restart', action = 'store_true', default = None)
parser.add_argument('--simnum', dest = 'simnum', default = '00001')

args = parser.parse_args()

dir_name = 'restart'
if(args.restart):
	if os.path.isdir(dir_name):
		if not os.listdir(dir_name):
			print("Directory restart is empty. Changing simulation mode from --restart to --equil.")
			args.restart = False
			args.equil = True
		print("Restarting")
	else:
		args.restart = False
		args.equil = True
		print("Given directory doesn't exist. Changing simulation mode from --restart to --equil.")

######################################################
# set-up parameters

filename_default = 'parameter/parameters-default' + args.simnum + '.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)
    
filename = 'parameter/parameters' + args.simnum + '.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)    

filename_prms = 'parameter/parameters.prms' + args.simnum + '.yaml'
with open(filename_prms, 'rb') as f:
    prms_emb = yaml.load(f, Loader = yaml.CLoader)

objType = (parameters_default["objFile"])[0:-4]
objFile = 'mesh/' + objType + args.simnum + '.off'
numsteps = (parameters_default["numsteps"] if args.restart else parameters_default["numsteps_eq"])
dt = (parameters_default["dt"] if args.restart else parameters_default["dt_eq"])
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]
nevery = (parameters["nevery"] if args.restart else parameters["nevery_eq"])
rhow = parameters_default["rhow"]
rhog = parameters_default["rhog"]
alpha = parameters_default["alpha"]
aii = parameters_default["aii"]
strong = parameters_default["strong"]
gamma_dpd = parameters_default["gamma_dpd"]
gamma_dpd_gas = parameters_default["gamma_dpd_gas"]
gamma_fsi = parameters["gamma_fsi"]
gamma_fsi_gas = parameters["gamma_fsi_gas"]
rc = parameters_default["rc"]
s = parameters_default["s"] 
s_g = parameters_default["s_g"] 
k_fsi = parameters_default["k_fsi"] 
kbt = parameters["kbt"]
obmd_flag = parameters_default["obmd_flag"]
mvert = parameters["mvert"]
mw = parameters["mw"]
mg = parameters["mg"]

pos_q = np.reshape(np.loadtxt('posq.txt'), (-1, 7))

pext = rhow * kbt + alpha * aii * rhow**2

if not args.restart:
    obmd = {
    "bufferSize" : parameters_default["bufferSize"] * Lx,
    "bufferAlpha" : parameters_default["bufferAlpha"],
    "bufferTau" : parameters_default["bufferTau"] * dt,
    "pext" : pext, 
    "ptan": 0.0 
    }

else:
    obmd = {
    "bufferSize" : parameters_default["bufferSize"] * Lx,
    "bufferAlpha" : parameters_default["bufferAlpha"],
    "bufferTau" : parameters_default["bufferTau"] * dt,
    "pext" : pext, 
    "ptan": parameters_default["ptan"] 
    }

ranks = (1, 1, 1)                       
domain = (Lx, Ly, Lz)   

######################################################

#mirheo coordinator
checkpoint_step = numsteps - 1
checkpoint_folder = ("restart/" if args.equil else "restart_prod/")
u = mir.Mirheo(
    ranks,
    domain,
    obmd,
    "open",
    debug_level = 3,
    log_filename = 'logs/log',
    checkpoint_folder = checkpoint_folder,
    checkpoint_every = checkpoint_step,
)

#loads the off script
mesh = trimesh.load_mesh(objFile)

#reads vertices, faces
mesh_emb = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())

emb = mir.ParticleVectors.MembraneVector("emb", mass = mvert, mesh = mesh_emb)

# Use the configured self-repulsion cutoff when available. The generated GV mesh
# has very short local edges; deriving the cutoff from the minimum edge can make
# Mirheo's cell-list setup unstable.
triangle = mesh.vertices[mesh.faces]
edge1 = triangle[:,1] - triangle[:,0]
edges = np.linalg.norm(edge1, axis=1)
mesh_lj_fac = 0.8 * np.min(edges)
lj_fac = parameters_default.get("lj_fac", mesh_lj_fac)

#initial condition for EMB
ic_emb   = mir.InitialConditions.Membrane(pos_q)

#register
u.registerParticleVector(emb, ic_emb)

#water
water = mir.ParticleVectors.ParticleVector('water', mass = mw, obmd = obmd_flag)
ic_water = mir.InitialConditions.Uniform(number_density = rhow)
u.registerParticleVector(water, ic_water)

#solvent 
sol2 = mir.ParticleVectors.ParticleVector('sol2', mass = mg, obmd = 0)
ic_outer2 = mir.InitialConditions.Uniform(number_density = rhog)
u.registerParticleVector(sol2, ic_outer2)

#splits the particle vector into inner part and outer part defined by the membrane
#only one can be not null, either inside or outside
inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
u.registerObjectBelongingChecker(inner_checker_1, emb)
gas = u.applyObjectBelongingChecker(inner_checker_1, sol2, correct_every = 0, inside = "gas", outside = "") # correct_every = 100000
#https://mirheo.readthedocs.io/en/latest/user/tutorials.html

inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
u.registerObjectBelongingChecker(inner_checker_2, emb)
u.applyObjectBelongingChecker(inner_checker_2, water, correct_every = 0, inside = "none", outside = "")

#interactions
strong = parameters_default["strong"]
afsi = parameters_default.get("afsi", 0.0) #strong * aii ##0.4 * aii
# mirheoOBMD 1.6.1 rejects the standard-Mirheo `bpress` membrane parameter.
# Keep bpress as a runtime control/provenance axis until OBMD pressure support is available.
int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = 0.0 * aii, gamma = 0.0 * gamma_dpd, kBT = kbt, power = s)
dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_gas = mir.Interactions.Pairwise('dpd_gas', rc, kind = "DPD", a = 0.0, gamma = gamma_dpd_gas, kBT = kbt, power = s_g)
dpd_fsi = mir.Interactions.Pairwise('dpd_fsi', rc, kind = "DPD", a = afsi, gamma = gamma_fsi, kBT = kbt, power = k_fsi)
dpd_fsi_gas = mir.Interactions.Pairwise('dpd_fsi_gas', rc, kind = "DPD", a = 0.0, gamma = gamma_fsi_gas, kBT = kbt, power = k_fsi)
lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
lj_int = mir.Interactions.Pairwise('lj_int', lj_fac, kind = "RepulsiveLJ", epsilon = 10000.0, sigma = lj_fac / (2**(1/6)), max_force = 10000.0)

######################################## INTEGRATOR ########################################
#initialize and register substep integrator
#substeps = 2 
#ss = mir.Integrators.SubStep('substep_membrane', substeps, [int_emb])
#u.registerIntegrator(ss)

#initialize and register Velocity Verlet integrator
vv = mir.Integrators.VelocityVerlet('vv')
u.registerIntegrator(vv)

#set integrator for various parts
if(args.restart):
	u.setIntegrator(vv, emb)
u.setIntegrator(vv, water)
u.setIntegrator(vv, gas)

######################################## INTERACTIONS ########################################
#register interactions
u.registerInteraction(int_emb)
u.registerInteraction(dpd)
u.registerInteraction(dpd_wat)
u.registerInteraction(dpd_gas)
u.registerInteraction(dpd_fsi)
u.registerInteraction(dpd_fsi_gas)
u.registerInteraction(lj)
u.registerInteraction(lj_int)

#set interaction
u.setInteraction(int_emb, emb, emb)
u.setInteraction(dpd_wat, water, water)
u.setInteraction(dpd_gas, gas, gas)
u.setInteraction(dpd, water, gas)
u.setInteraction(dpd_fsi, emb, water)
u.setInteraction(dpd_fsi_gas, emb, gas)
u.setInteraction(lj, emb, emb)
u.setInteraction(lj_int, emb, emb)
######################################## REFLECTION BOUNDARIES ########################################
#reflection boundaries of gas vesicle shells
#bouncer = mir.Bouncers.Mesh("membrane_bounce", "bounce_back")
bouncer = mir.Bouncers.Mesh("membrane_bounce", "bounce_maxwell", kBT = 1.0 * kbt)
u.registerBouncer(bouncer)
u.setBouncer(bouncer, emb, water)
u.setBouncer(bouncer, emb, gas)

######################################## PLUGINS ########################################

gridSampleEvery = 2
gridDumpEvery   = 1000
gridBinSize     = (1., Ly, Lz)

#u.registerPlugins(mir.Plugins.createDumpAverage('field', [water], gridSampleEvery, gridDumpEvery, gridBinSize, ["velocities"], f'h5/solvent{args.simnum}-'))
ptan = parameters_default["ptan"]
folder = 'trj/shear' + '/shear' + str(ptan) + '-'

if args.equil:
    print('equilibration')
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", emb, nevery, f"ply/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', water, nevery, [], folder))
    u.run(numsteps, dt = dt)

if args.restart:
    print('productionn')
    u.restart("restart/")
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.registerPlugins(mir.Plugins.createDumpMesh("mesh_dump", emb, nevery, f"ply/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', water, nevery, [], folder))
    u.run(numsteps, dt = dt)
