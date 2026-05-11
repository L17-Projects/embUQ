#!/usr/bin/env python3

import importlib
import os

_MIRHEO_MODULE = os.environ.get("MESOUQ_GV_MIRHEO_MODULE", "mirheo")
mir = importlib.import_module(_MIRHEO_MODULE)
_PAPER_EXACT = os.environ.get("MESOUQ_GV_PAPER_EXACT", "").lower() in {"1", "true", "yes"}
_DEFAULT_MEMBRANE_BPRESS_MODE = "runtime"
_MEMBRANE_BPRESS_MODE = os.environ.get(
    "MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE",
    _DEFAULT_MEMBRANE_BPRESS_MODE,
).lower()
_FLUID_MODE = os.environ.get("MESOUQ_GV_BUCKLING_FLUID_MODE", "dropped").lower()
_FLUID_STABILIZATION_OVERRIDE = os.environ.get("MESOUQ_GV_BUCKLING_FLUID_STABILIZATION")
_PIN_OBJECT_MODE = os.environ.get("MESOUQ_GV_BUCKLING_PIN_OBJECT", "1").lower()
_ODPD_AMP_SCALE = float(os.environ.get("MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE", "1.0"))


def _pin_object_enabled():
    return _PIN_OBJECT_MODE not in {"0", "false", "no", "off"}


def _legacy_stabilized_fluid_mode():
    return _FLUID_MODE in {"legacy", "legacy_stabilized", "paper_legacy"}


def _fluid_stabilization():
    if _FLUID_STABILIZATION_OVERRIDE is None:
        return 1.0 if _legacy_stabilized_fluid_mode() else 0.0
    value = float(_FLUID_STABILIZATION_OVERRIDE)
    if value < 0.0 or value > 1.0:
        raise ValueError("MESOUQ_GV_BUCKLING_FLUID_STABILIZATION must be in [0, 1].")
    return value


def _create_mirheo(ranks, domain, **kwargs):
    if _MIRHEO_MODULE.startswith("mirheoOBMD"):
        return mir.Mirheo(ranks, domain, {}, "open", **kwargs)
    return mir.Mirheo(ranks, domain, **kwargs)


def _create_particle_vector(name, *, mass, obmd):
    if _MIRHEO_MODULE.startswith("mirheoOBMD"):
        return mir.ParticleVectors.ParticleVector(name, mass=mass, obmd=obmd)
    return mir.ParticleVectors.ParticleVector(name, mass=mass)

import numpy as np
import trimesh
import yaml
import argparse

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
numsteps = (int(parameters_default["numsteps"]) if args.restart else int(parameters_default["numsteps_eq"]))
numsteps_eq = int(parameters_default["numsteps_eq"])
dt = (parameters_default["dt"] if args.restart else parameters_default["dt_eq"])
dt_eq = parameters_default["dt_eq"]
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]
nevery = (parameters["nevery"] if args.restart else parameters["nevery_eq"])
rhow = parameters_default["rhow"]
rhog = parameters_default["rhog"]
alpha = parameters_default["alpha"]
aii = parameters_default["aii"] * parameters["kbt"]
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
buck = parameters_default["buck"]



pos_q = np.reshape(np.loadtxt('posq.txt'), (-1, 7))

ranks = (1, 1, 1)
domain = (Lx, Ly, Lz)

######################################################
checkpoint_step = numsteps - 1

u = _create_mirheo(ranks, domain, debug_level = 3, log_filename = 'logs/log', checkpoint_folder = "restart/", checkpoint_every = checkpoint_step) #, MPI._addressof(comm))


#loads the off script
mesh = trimesh.load_mesh(objFile)

#sets the lj_fac to half the minimum edge length
triangle = mesh.vertices[mesh.faces]
edge1 = triangle[:,1] - triangle[:,0]
edges = np.linalg.norm(edge1, axis=1)
mesh_lj_fac = 0.8 * np.min(edges)
lj_fac = 0.7 * np.min(edges) if _legacy_stabilized_fluid_mode() else mesh_lj_fac

#reads vertices, faces
mesh_emb = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())

emb_mass = mvert if _legacy_stabilized_fluid_mode() else 100 * mvert
emb = mir.ParticleVectors.MembraneVector("emb", mass = emb_mass, mesh = mesh_emb)

#initial condition for EMB
ic_emb  = mir.InitialConditions.Membrane(pos_q)

#register
u.registerParticleVector(emb, ic_emb)

#water
water = _create_particle_vector('water', mass = mw, obmd = obmd_flag)
ic_water = mir.InitialConditions.Uniform(number_density = rhow)
u.registerParticleVector(water, ic_water)

#solvent
sol2 = _create_particle_vector('sol2', mass = mg, obmd = 0)
ic_outer2 = mir.InitialConditions.Uniform(number_density = rhog)
u.registerParticleVector(sol2, ic_outer2)

#splits the particle vector into inner part and outer part defined by the membrane
#only one can be not null, either inside or outside
inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
u.registerObjectBelongingChecker(inner_checker_1, emb)
gas = u.applyObjectBelongingChecker(inner_checker_1, sol2, correct_every = 0, inside = "gas", outside = "")
#https://mirheo.readthedocs.io/en/latest/user/tutorials.html

inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
u.registerObjectBelongingChecker(inner_checker_2, emb)
u.applyObjectBelongingChecker(inner_checker_2, water, correct_every = 0, inside = "none", outside = "")

tdump = nevery * dt

radGV = parameters_default["radGV"]
#buck = buck / radGV

#interactions
fluid_stabilization = _fluid_stabilization()
legacy_afsi_factor = 10.0 * (2.0 / radGV)**2
afsi = (2.0 + fluid_stabilization * (legacy_afsi_factor - 2.0)) * aii #buck * aii #prej 0.5 *
if fluid_stabilization <= 0.0:
    lj_fac = parameters_default["lj_fac"]
facg = parameters_default["facg"]
bpress = parameters_default["bpress"]

if(objType == 'gv'):
    if _legacy_stabilized_fluid_mode():
        buck = buck * (2.0 / radGV)**2
    if _MEMBRANE_BPRESS_MODE == "runtime":
        prms_emb["bpress"] = bpress #-29.0
    elif _MEMBRANE_BPRESS_MODE == "zero":
        prms_emb["bpress"] = 0.0
    else:
        raise ValueError("MESOUQ_GV_BUCKLING_MEMBRANE_BPRESS_MODE must be 'runtime' or 'zero'.")
    int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
else:
    int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True)


t0 = 0.0
print(u)
if(u.isComputeTask()):
    print('yess')
    t0 = u.getState().current_time
print(f'time = {t0}')
#timestart = t0
#timeend = (t0 + numsteps * dt if args.restart else t0 + numsteps_eq * dt_eq)

water_gas_a = fluid_stabilization * aii
water_gas_gamma = fluid_stabilization * gamma_dpd
gas_a = fluid_stabilization * facg * aii
gas_fsi_a = fluid_stabilization * afsi
dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = water_gas_a, gamma = water_gas_gamma, kBT = kbt, power = s)
dpd_gas = mir.Interactions.Pairwise('dpd_gas', rc, kind = "DPD", a = gas_a, gamma = gamma_dpd_gas, kBT = kbt, power = s_g)
dpd_fsi = mir.Interactions.Pairwise('dpd_fsi', rc, kind = "DPD", a = afsi, gamma = gamma_fsi, kBT = kbt, power = k_fsi)
dpd_fsi_gas = mir.Interactions.Pairwise('dpd_fsi_gas', rc, kind = "DPD", a = gas_fsi_a, gamma = gamma_fsi_gas, kBT = kbt, power = k_fsi)
lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
#lj_int = mir.Interactions.Pairwise('lj_int', lj_fac * rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = lj_fac * rc / (2**(1/6)), max_force = 10.0)
lj_epsilon = 1000.0 if _legacy_stabilized_fluid_mode() else 10000.0
lj_sigma = (0.99 * lj_fac if _legacy_stabilized_fluid_mode() else lj_fac) / (2**(1/6))
lj_max_force = 1000.0 if _legacy_stabilized_fluid_mode() else 10000.0
lj_int = mir.Interactions.Pairwise('lj_int', lj_fac, kind = "RepulsiveLJ", epsilon = lj_epsilon, sigma = lj_sigma, max_force = lj_max_force)
#dpd_int = mir.Interactions.Pairwise('dpd_int', lj_fac , kind = "DPD", a = 10*aii, gamma = 0*gamma_dpd, kBT = kbt, power = s)

#niter = 10
#epps = 1e-4
#fbuck = 0.75
#delta_buck = fbuck / (niter - 1)
a0 = aii
#if buck > delta_buck - epps:
#	a0 = aii + buck * aii - delta_buck * aii

#nbstep = buck / delta_buck

#timestart = (numsteps_eq * dt_eq + nbstep * numsteps * dt if args.restart else 0.0)
#timeend = (numsteps_eq * dt_eq + numsteps * dt + nbstep * numsteps * dt if args.restart else numsteps_eq * dt_eq)

timestart = (numsteps_eq * dt_eq if args.restart else 0.0)
timeend = (numsteps_eq * dt_eq + numsteps * dt if args.restart else numsteps_eq * dt_eq)

#odpd = mir.Interactions.Pairwise('odpd', rc, kind = "ODPD", a = a0, gamma = gamma_dpd, kBT = kbt, power = s, timestart = timestart, timeend = timeend, amp = buck * aii, mode = 2, stress=True, stress_period = tdump) # 1 = 'hysteresis', 0 - 'forward', 2 - 'forward + equil'

if _legacy_stabilized_fluid_mode():
    odpd = mir.Interactions.Pairwise('odpd', rc, kind = "ODPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s, timestart = timestart, timeend = timeend, amp = _ODPD_AMP_SCALE * buck * aii, mode = 2, stress=True, stress_period = tdump) # 1 = 'hysteresis', 0 - 'forward', 2 - 'forward + equil'
elif(args.restart):
    odpd = mir.Interactions.Pairwise('odpd', rc, kind = "ODPD", a = a0, gamma = gamma_dpd, kBT = kbt, power = s, timestart = timestart, timeend = timeend, amp = _ODPD_AMP_SCALE * buck * aii, mode = 2, stress=True, stress_period = tdump) # 1 = 'hysteresis', 0 - 'forward', 2 - 'forward + equil'
else:
    odpd = mir.Interactions.Pairwise('odpd', rc, kind = "ODPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s, timestart = timestart, timeend = timeend, amp = 0 * aii, mode = 2, stress=True, stress_period = tdump) # 1 = 'hysteresis', 0 - 'forward', 2 - 'forward + equil'

#print('*********************')
#print(f'a0 = {a0}, amp = {buck * aii}, delta_buck = {delta_buck}, buck = {buck}')
#print(f'timestart = {timestart}, timeend = {timeend}')
#print('*********************')
######################################## INTEGRATOR ########################################
#initialize integrator
# 1step of dt for solvent, substep steps of dt / substeps for membrane
#substeps = 20
#ss = mir.Integrators.SubStep('substep_membrane', substeps, [int_emb])
#u.registerIntegrator(ss)

vv = mir.Integrators.VelocityVerlet('vv')

#register integrator
u.registerIntegrator(vv)

#set integrator for various parts
if _legacy_stabilized_fluid_mode() or args.restart:
    u.setIntegrator(vv, emb)

#u.setIntegrator(vv, emb)
u.setIntegrator(vv, water)
u.setIntegrator(vv, gas)

######################################## INTERACTIONS ########################################
#register interactions
#u.registerInteraction(int_emb)
u.registerInteraction(int_emb)
u.registerInteraction(dpd_gas)
u.registerInteraction(dpd)
u.registerInteraction(odpd)
u.registerInteraction(dpd_fsi)
u.registerInteraction(dpd_fsi_gas)
u.registerInteraction(lj)
u.registerInteraction(lj_int)

#u.registerInteraction(dpd_int)
#set interaction
#u.setInteraction(int_emb, emb, emb)
u.setInteraction(int_emb, emb, emb)
#if(args.restart):
u.setInteraction(odpd, water, water)
print('restart odpd')
#else:
#	u.setInteraction(dpd, water, water)
#	print('without odpd')

u.setInteraction(dpd_gas, gas, gas)
u.setInteraction(dpd, water, gas)

u.setInteraction(dpd_fsi, emb, water)
u.setInteraction(dpd_fsi_gas, emb, gas)

#u.setInteraction(dpd, emb, water)
#u.setInteraction(dpd, emb, gas)

u.setInteraction(lj, emb, emb)
u.setInteraction(lj_int, emb, emb)

######################################## REFLECTION BOUNDARIES ########################################
#reflection boundaries of gas vesicle shells
bouncer = mir.Bouncers.Mesh("membrane_bounce", 100, 150, "bounce_maxwell", kBT = kbt)
#bouncer = mir.Bouncers.Mesh("membrane_bounce", 1000, 150, "bounce_back")#, kBT = 0*kbt)
u.registerBouncer(bouncer)
u.setBouncer(bouncer, emb, water)
u.setBouncer(bouncer, emb, gas)

######################################## RUN ########################################



def predicate_all_domain(r):
	return 1.0

h = (1.0, 1.0, 1.0)

height = parameters_default["height"]

cen = np.array([0.5 * Lx, 0.5 * Ly, 0.5 * Lz])

def positions(t):
	return [cen + mesh.vertices[np.argmin(mesh.vertices[:,2])], cen + mesh.vertices[np.argmax(mesh.vertices[:,2])]]

def velocities(t):
	return [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

if args.equil:
    print('equilibration')
    #pids = [0, len(mesh.vertices) - 1]
    pids = [np.argmin(mesh.vertices[:,2]), np.argmax(mesh.vertices[:,2])]
    #print(f'utime eq = {u.getState().current_time}')
    #set_interactions(u)
    if _pin_object_enabled():
        unr = mir.Plugins.PinObject.Unrestricted
        omega = [unr, unr, unr]
        velocity = [0.0, 0.0, 0.0]
        u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpMesh('ply_dump', emb, nevery, f"ply_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_gas', gas, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createVirialPressurePlugin('virial', water, predicate_all_domain, h, nevery, 'pressure/p' + args.simnum))
    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", emb, positions, velocities, pids, nevery, "anchor/"))
    u.run(numsteps, dt = dt)
    del u

if args.restart:
    print('productionn')
    pids = [np.argmin(mesh.vertices[:,2]), np.argmax(mesh.vertices[:,2])]
    u.restart("restart/")
    #set_interactions(u)
    #print(f'utime res = {u.getState().current_time}')
    if _pin_object_enabled():
        unr = mir.Plugins.PinObject.Unrestricted
        omega = [unr, unr, unr]
        velocity = [0.0, 0.0, 0.0]
        u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpMesh('ply_dump', emb, nevery, f"ply_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_gas', gas, nevery, f"trj_eq/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createVirialPressurePlugin('virial', water, predicate_all_domain, h, nevery, 'pressure/p' + args.simnum))

    u.registerPlugins(mir.Plugins.createDumpObjectStats('objStats', emb, nevery, filename = 'stats/object' + args.simnum))
    u.registerPlugins(mir.Plugins.createAnchorParticles("anchor", emb, positions, velocities, pids, nevery, "anchor/"))
    u.run(numsteps, dt = dt)
    del u
