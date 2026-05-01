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

def computeForces(vertices, fraction, force):
    vertices= np.array(vertices)
    k = int(fraction * 0.5 * len(vertices))
    
    #k = 1

    ind_max = np.argpartition(+vertices[:,2], -k)[-k:]
    ind_min = np.argpartition(-vertices[:,2], -k)[-k:]

    #vert1 = np.max(vertices[ind_max][:,2])
    vert2 = np.argmin(vertices[ind_max][:,2])
    print('max vertex', vert2)
    
    forces = np.zeros((len(vertices), 3))

    forces[ind_max,2] = +force
    forces[ind_min,2] = -force
    return forces

def computeTorsionalForces(vertices, z0, force, tol=0.1):
    vertices = np.array(vertices)
    
    # Identify indices of particles near heights z0 and -z0 within tolerance
    ind_max = np.where((vertices[:, 2] > z0 - tol) & (vertices[:, 2] < z0 + tol))[0]
    ind_min = np.where((vertices[:, 2] > -z0 - tol) & (vertices[:, 2] < -z0 + tol))[0]

    # Initialize forces array
    forces = np.zeros((len(vertices), 3))

    # Get the XY coordinates of the vertices near z0 and -z0
    vertices_max_xy = vertices[ind_max, :2]
    vertices_min_xy = vertices[ind_min, :2]

    # Compute perpendicular vectors in the XY plane for the selected vertices
    # Perpendicular vector for (x, y) is (-y, x)
    perp_vec_max = np.column_stack((-vertices_max_xy[:, 1], vertices_max_xy[:, 0]))
    perp_vec_min = np.column_stack((-vertices_min_xy[:, 1], vertices_min_xy[:, 0]))

    # Normalize these perpendicular vectors
    perp_vec_max /= np.linalg.norm(perp_vec_max, axis=1)[:, np.newaxis]
    perp_vec_min /= np.linalg.norm(perp_vec_min, axis=1)[:, np.newaxis]

    # Scale by the force magnitude
    perp_vec_max *= force
    perp_vec_min *= -force

    # Assign the forces to the appropriate indices in the forces array
    forces[ind_max, :2] = perp_vec_max
    forces[ind_min, :2] = perp_vec_min

    return forces


def rotate_points_around_z(vertices, indices, theta):
    # Convert to radians if theta is provided in degrees (optional)
    # theta = np.radians(theta)
    
    # Extract the points to rotate
    points = vertices[indices, :2]  # Only take XY coordinates for rotation
    
    # Define the rotation matrix in the XY plane
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    # Apply the rotation to each point in the XY plane
    rotated_points = points @ rotation_matrix.T
    
    # Update the original vertices array with the rotated points
    vertices_rotated = np.copy(vertices)
    vertices_rotated[indices, :2] = rotated_points  # Update only XY coordinates

    return vertices_rotated[indices]

def findpids(vertices, z0, tol = 0.1):
    vertices = np.array(vertices)
    
    # Identify indices of particles near heights z0 and -z0 within tolerance
    #ind_max = np.where((vertices[:, 2] > z0 - tol) & (vertices[:, 2] < z0 + tol))[0]
    #ind_min = np.where((vertices[:, 2] > -z0 - tol) & (vertices[:, 2] < -z0 + tol))[0]

    ind_max = np.where(vertices[:, 2] > z0)[0]
    ind_min = np.where(vertices[:, 2] < -z0)[0]
    
    return ind_max, ind_min


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

#reads vertices, faces
mesh_emb = mir.ParticleVectors.MembraneMesh(vertices = mesh.vertices.tolist(), stress_free_vertices = mesh.vertices.tolist(), faces = mesh.faces.tolist())

emb = mir.ParticleVectors.MembraneVector("emb", mass = mvert, mesh = mesh_emb)

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

#if(args.vacuum):
#    prms_emb["gammaC"] = gamma_dpd
if(objType == 'gv'):
    prms_emb["bpress"] = -91.0
    int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
else:
    int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True)

if(args.vacuum):
    dpd0 = mir.Interactions.Pairwise('dpd0', rc, kind = "DPD", a = 0.0, gamma = 1 * gamma_dpd, kBT = 0.015 * kbt, power = s)  
dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = 0*aii, gamma = 0*gamma_dpd, kBT = kbt, power = s)
dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_gas = mir.Interactions.Pairwise('dpd_gas', rc, kind = "DPD", a = aii, gamma = gamma_dpd_gas, kBT = kbt, power = s_g)
dpd_fsi = mir.Interactions.Pairwise('dpd_fsi', rc, kind = "DPD", a = afsi, gamma = gamma_fsi, kBT = kbt, power = k_fsi)
dpd_fsi_gas = mir.Interactions.Pairwise('dpd_fsi_gas', rc, kind = "DPD", a = afsi, gamma = gamma_fsi_gas, kBT = kbt, power = k_fsi)
lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
#lj_int = mir.Interactions.Pairwise('lj_int', lj_fac * rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = lj_fac * rc / (2**(1/6)), max_force = 10.0)

######################################## INTEGRATOR ########################################
#initialize integrator
vv = mir.Integrators.VelocityVerlet('vv')

#register integrator
u.registerIntegrator(vv)

#set integrator for various parts
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
u.registerInteraction(lj)
#u.registerInteraction(lj_int)

#set interaction
u.setInteraction(int_emb, emb, emb)
u.setInteraction(lj, emb, emb)
#u.setInteraction(lj_int, emb, emb)
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

#if(not args.vacuum):
#    bouncer = mir.Bouncers.Mesh("membrane_bounce", "bounce_maxwell", kBT = kbt)
#    u.registerBouncer(bouncer)
#    u.setBouncer(bouncer, emb, water)
#    u.setBouncer(bouncer, emb, gas)
#    #u.setBouncer(bouncer, emb, emb)

######################################## RUN ########################################

height = parameters_default["height"]

cen = np.array([0.5 * Lx, 0.5 * Ly, 0.5 * Lz])

frac_center = 0.35
ids_max, ids_min = findpids(mesh.vertices, frac_center * height, tol = 0.1)


def positions(t):
	return [cen + mesh.vertices[np.argmin(mesh.vertices[:,2])], cen + mesh.vertices[np.argmax(mesh.vertices[:,2])]]

def velocities(t):
	return [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
	

theta = parameters_default["theta"]
rot1 = rotate_points_around_z(mesh.vertices, ids_max, theta)
rot2 = rotate_points_around_z(mesh.vertices, ids_min, -theta)

print(f'folzine {len(rot1)}, {len(ids_min)}')

def positions_min(t):
	return cen + rot2 #rot2 #mesh.vertices[ids_min] - np.array([0.0, 0.0, displ])

def velocities_min(t):
	return len(ids_min) * [(0.0, 0.0, 0.0)]


def positions_max(t):
	return cen + rot1 #mesh.vertices[ids_max] + np.array([0.0, 0.0, displ])

def velocities_max(t):
	return len(ids_max) * [(0.0, 0.0, 0.0)]

if args.equil:
    print('equilibration')
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    pids = [np.argmin(mesh.vertices[:,2]), np.argmax(mesh.vertices[:,2])]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    #forces = computeTorsionalForces(mesh.vertices, fraction, force).tolist()
    #forces = computeTorsionalForces(mesh.vertices, 0.4 * height, force, tol = 0.1).tolist()
    
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor_min", emb, positions_min, velocities_min, ids_min, nevery, f"anchor_min/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor_max", emb, positions_max, velocities_max, ids_max, nevery, f"anchor_max/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", emb, positions, velocities, pids, nevery, "anchor/"))
    u.run(numsteps, dt = dt)
    del u

if args.restart:
    print('productionn')
    u.restart("restart/")
    unr = mir.Plugins.PinObject.Unrestricted
    omega = [unr, unr, unr]
    velocity = [0.0, 0.0, 0.0]
    pids = [np.argmin(mesh.vertices[:,2]), np.argmax(mesh.vertices[:,2])]
    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    #forces = computeTorsionalForces(mesh.vertices, fraction, force).tolist()
    #forces = computeTorsionalForces(mesh.vertices, 0.4 * height, force, tol = 0.1).tolist()
    
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor_min", emb, positions_min, velocities_min, ids_min, nevery, f"anchor_min/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor_max", emb, positions_max, velocities_max, ids_max, nevery, f"anchor_max/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", emb, positions, velocities, pids, nevery, "anchor/"))
    u.run(numsteps, dt = dt)
    del u
