#!/usr/bin/env python3

import mirheo as mir
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
parser.add_argument('--vacuum', action = 'store_true', default = None)

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


def computeForces1(vertices, force, z0, tolerance):
    vertices = np.array(vertices)
    
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])
    
    # Select points close to z_max - z0 and z_min + z0 within the tolerance
    top_mask = np.abs(vertices[:, 2] - (z_max - z0)) <= tolerance
    bottom_mask = np.abs(vertices[:, 2] - (z_min + z0)) <= tolerance
    
    top_indices = np.where(top_mask)[0]
    bottom_indices = np.where(bottom_mask)[0]
    
    forces = np.zeros((len(vertices), 3))
    
    forces[top_indices, 2] = +force
    forces[bottom_indices, 2] = -force
    
    return forces


def computeForces(vertices, force, distance):
    vertices = np.array(vertices)
    
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])
    
    # Select points above (z_max - distance) or below (z_min + distance)
    top_mask = vertices[:, 2] >= (z_max - distance)
    bottom_mask = vertices[:, 2] <= (z_min + distance)
    
    top_indices = np.where(top_mask)[0]
    bottom_indices = np.where(bottom_mask)[0]
    
    forces = np.zeros((len(vertices), 3))
    
    forces[top_indices, 2] = +force
    forces[bottom_indices, 2] = -force
    
    return forces

def computeIndices1(vertices, z0, tolerance):
    vertices = np.array(vertices)
    
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])
    
    # Select points close to z_max - z0 and z_min + z0 within the tolerance
    top_mask = np.abs(vertices[:, 2] - (z_max - z0)) <= tolerance
    bottom_mask = np.abs(vertices[:, 2] - (z_min + z0)) <= tolerance
    
    top_indices = np.where(top_mask)[0]
    bottom_indices = np.where(bottom_mask)[0]

    return bottom_indices, top_indices

def computeIndices(vertices, distance):
    vertices = np.array(vertices)
    
    z_max = np.max(vertices[:, 2])
    z_min = np.min(vertices[:, 2])
    
    # Select points above (z_max - distance) or below (z_min + distance)
    top_mask = vertices[:, 2] >= (z_max - distance)
    bottom_mask = vertices[:, 2] <= (z_min + distance)
    
    top_indices = np.where(top_mask)[0]
    bottom_indices = np.where(bottom_mask)[0]

    return bottom_indices, top_indices


objType = (parameters_default["objFile"])[0:-4]
objFile = 'mesh/' + objType + args.simnum + '.off'
#objFile = parameters_default["objFile"]
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
lj_fac = parameters_default["lj_fac"]

pos_q = np.reshape(np.loadtxt('posq.txt'), (-1, 7))

ranks = (1, 1, 1)                       
domain = (Lx, Ly, Lz)  

force = parameters_default["force"]         

######################################################
checkpoint_step = numsteps - 1

#mirheo coordinator
u = mir.Mirheo(ranks, domain, debug_level = 3, log_filename = 'logs/log', checkpoint_folder = "restart/", checkpoint_every = checkpoint_step)

#loads the off script
mesh = trimesh.load_mesh(objFile)

# Use the configured self-repulsion cutoff when available. The generated GV mesh
# has very short local edges; deriving the cutoff from the minimum edge can make
# Mirheo's cell-list setup unstable.
triangle = mesh.vertices[mesh.faces]
edge1 = triangle[:,1] - triangle[:,0]
edges = np.linalg.norm(edge1, axis=1)
mesh_lj_fac = 0.8 * np.min(edges)
lj_fac = parameters_default.get("lj_fac", mesh_lj_fac)

#reads vertices, faces
mesh_emb = mir.ParticleVectors.MembraneMesh(vertices = mesh.vertices.tolist(), stress_free_vertices = mesh.vertices.tolist(), faces = mesh.faces.tolist())

emb = mir.ParticleVectors.MembraneVector("emb", mass = 50*mvert, mesh = mesh_emb)

#initial condition for GV
ic_emb = mir.InitialConditions.Membrane(pos_q)

#register
u.registerParticleVector(emb, ic_emb)

if(not args.vacuum):
    #water
    water = mir.ParticleVectors.ParticleVector('water', mass = mw)
    ic_water = mir.InitialConditions.Uniform(number_density = rhow)
    u.registerParticleVector(water, ic_water)

    #solvent 
    sol2 = mir.ParticleVectors.ParticleVector('sol2', mass = mg)
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
#int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
#dpd_thermostat = mir.Interactions.Pairwise('dpd_thermostat', rc, kind = "DPD", a = 0.0, gamma = gamma_dpd, kBT = kbt, power = s)
#dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
#dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
#lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')

afsi = 0.4 * aii
bpress = parameters_default["bpress"]
#if(args.vacuum):
#    prms_emb["gammaC"] = gamma_dpd
if(objType == 'gv'):
    prms_emb["bpress"] = bpress #-29.0
    int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
else:
    int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True)

if(args.vacuum):
    dpd0 = mir.Interactions.Pairwise('dpd0', rc, kind = "DPD", a = 0.0, gamma = 3 * gamma_dpd, kBT = 0.015 * kbt, power = s)    
dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = 0*aii, gamma = 0*gamma_dpd, kBT = kbt, power = s)
dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_gas = mir.Interactions.Pairwise('dpd_gas', rc, kind = "DPD", a = 0.0 * aii, gamma = gamma_dpd_gas, kBT = kbt, power = s_g)
dpd_fsi = mir.Interactions.Pairwise('dpd_fsi', rc, kind = "DPD", a = afsi, gamma = 1*gamma_fsi, kBT = kbt, power = k_fsi)
dpd_fsi_gas = mir.Interactions.Pairwise('dpd_fsi_gas', rc, kind = "DPD", a = 0.0 * afsi, gamma = gamma_fsi_gas, kBT = kbt, power = k_fsi)
#lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
lj_int = mir.Interactions.Pairwise('lj_int', lj_fac, kind = "RepulsiveLJ", epsilon = 10000.0, sigma = lj_fac / (2**(1/6)), max_force = 100000.0)
#lj_int = mir.Interactions.Pairwise('lj_int', lj_fac * rc, kind = "DPD", a = 100.0, gamma = 0.0, kBT = 0.0, power = s)

######################################## INTEGRATOR ########################################
#initialize integrator
vv = mir.Integrators.VelocityVerlet('vv')

#register integrator
u.registerIntegrator(vv)

#set integrator for various parts
#sif(args.restart):
u.setIntegrator(vv, emb)

if(not args.vacuum):
    u.setIntegrator(vv, water)
    u.setIntegrator(vv, gas)

######################################## INTERACTIONS ########################################
#register interactions
u.registerInteraction(int_emb)
if(args.vacuum):
    u.registerInteraction(dpd0)

u.registerInteraction(dpd)
u.registerInteraction(dpd_wat)
u.registerInteraction(dpd_gas)
u.registerInteraction(dpd_fsi)
u.registerInteraction(dpd_fsi_gas)
#u.registerInteraction(lj)
u.registerInteraction(lj_int)

#set interaction
u.setInteraction(int_emb, emb, emb)
#u.setInteraction(lj, emb, emb)
u.setInteraction(lj_int, emb, emb)
if(not args.vacuum):
    u.setInteraction(dpd_wat, water, water)
    u.setInteraction(dpd_gas, gas, gas)
    u.setInteraction(dpd, water, gas)
    u.setInteraction(dpd_fsi, emb, water)
    u.setInteraction(dpd_fsi_gas, emb, gas)

if(args.vacuum):
    u.setInteraction(dpd0, emb, emb)
######################################## REFLECTION BOUNDARIES ########################################
#reflection boundaries of gas vesicle shells

if(not args.vacuum):
    bouncer = mir.Bouncers.Mesh("membrane_bounce", 150, 150, "bounce_maxwell", kBT = kbt)
    u.registerBouncer(bouncer)
    u.setBouncer(bouncer, emb, water)
    u.setBouncer(bouncer, emb, gas)
#    u.setBouncer(bouncer, emb, emb)

######################################## RUN ########################################

fraction = parameters_default["fraction"]
tolerance = 0.2
z0 = 0.15 * parameters_default["height"]
#ind_min, ind_max = computeIndices(mesh.vertices, z0, tolerance)
ind_min, ind_max = computeIndices(mesh.vertices, z0)

force = parameters_default["tot_force"] / len(ind_min)

#forces = computeForces(mesh.vertices, force, z0, tolerance).tolist()
forces = computeForces(mesh.vertices, force, z0).tolist()


cen = np.array([0.5 * Lx, 0.5 * Ly, 0.5 * Lz])

def positions(t):
	return [cen + mesh.vertices[np.argmin(mesh.vertices[:,2])], cen + mesh.vertices[np.argmax(mesh.vertices[:,2])]]

def velocities(t):
	return [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

if args.equil:
    print('equilibration')
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    pids = [np.argmin(mesh.vertices[:,2]), np.argmax(mesh.vertices[:,2])]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    #forces = computeForces(mesh.vertices, force, z0, tolerance).tolist()
    forces = computeForces(mesh.vertices, force, z0).tolist()
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_gas', gas, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", emb, positions, velocities, pids, nevery, "anchor/"))
    #u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
    u.run(numsteps, dt = dt)
    #del u

if args.restart:
    print('productionn')
    u.restart("restart/")
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    #forces = computeForces(mesh.vertices, force, z0, tolerance).tolist()
    forces = computeForces(mesh.vertices, force, z0).tolist()
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_gas', gas, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
    u.run(numsteps, dt = dt)
