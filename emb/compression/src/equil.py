"""
Mirheo DPD simulation driver for EMB compression experiments.

Requires: mirheo (Python 3.8, GPU), mpi4py, trimesh, numpy, pyyaml, torch.
Source: Hierarchical_UQ_compression_dev/emb/compression/src/equil.py
"""

import inspect
import os
import sys

import mirheo as mir
import numpy as np
import torch  # type: ignore
import trimesh
import yaml
from mpi4py import MPI


def findpids(vertices, z0, dz):
    vertices = np.array(vertices)

    ind = np.where((vertices[:, 2] >= z0 - dz) & (vertices[:, 2] <= z0 + dz))[0]
    # pins = np.int64(np.linspace(0, len(ind)-1, 3))
    # ind = ind[pins]
    pos = vertices[ind]

    return pos, list(ind)


def run_equil(
    source_path: str,
    simu_path: str,
    simnum: str,
    equil: bool,
    restart: bool,
    restart_path: str,
    comm: MPI.Comm,
    dump: bool = None,
):

    # Search for config file in multiple locations
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_based_root = os.path.dirname(os.path.dirname(os.path.dirname(file_dir)))
    config_paths = [
        os.path.join(file_based_root, "inference/configs/production/inference_config_compression.yaml"),
        os.path.join(file_based_root, "inference/configs/production/inference_config.yaml"),
        "../../../inference/configs/production/inference_config_compression.yaml",  # From emb/compression/src/
        "../../../inference/configs/production/inference_config.yaml",  # Legacy production filename
        "inference/configs/production/inference_config_compression.yaml",  # From project root
        "inference/configs/production/inference_config.yaml",  # Legacy project-root filename
        "../inference/configs/production/inference_config_compression.yaml",  # From emb/compression/
        "../inference/configs/production/inference_config.yaml",  # Legacy emb/compression/ filename
        "../../inference/configs/production/inference_config_compression.yaml",  # Historical compression/src/ candidate
        "../../inference/configs/production/inference_config.yaml",
    ]
    config_file = None
    for path in config_paths:
        if os.path.exists(path):
            config_file = path
            break
    if config_file is None:
        raise FileNotFoundError(
            f"Could not find production compression config in any of: {config_paths}"
        )

    with open(config_file, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    if config["debug"] >= 1:
        print(f"Running from function {inspect.currentframe().f_code.co_name} in script {__file__}")

    # Allow dump parameter to override config (for evaluation/debugging)
    if dump is None:
        dump = config["dump"]
    frac_diam = config["frac_diam"]

    ######################################################
    # set-up parameters

    filename_default = simu_path + "parameter/parameters-default" + simnum + ".yaml"
    with open(filename_default, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)

    filename = simu_path + "parameter/parameters" + simnum + ".yaml"
    with open(filename, "rb") as f:
        parameters = yaml.load(f, Loader=yaml.CLoader)

    filename_prms = simu_path + "parameter/parameters.prms" + simnum + ".yaml"
    with open(filename_prms, "rb") as f:
        prms_emb = yaml.load(f, Loader=yaml.CLoader)

    objType = (parameters_default["objFile"])[0:-4]
    objFile = "mesh/" + objType + simnum + ".off"
    numsteps = int(parameters_default["numsteps"])
    numsteps_eq = int(parameters_default["numsteps_eq"])
    dt = parameters_default["dt"]
    dt_eq = parameters_default["dt_eq"]
    Lx = parameters_default["Lx"]
    Ly = parameters_default["Ly"]
    Lz = parameters_default["Lz"]
    radp = parameters_default["radp"]
    nevery = parameters["nevery"]
    nevery_eq = parameters["nevery_eq"]
    rhow = parameters_default["rhow"]
    rhog = parameters_default["rhog"]
    aii = parameters_default["aii"] * parameters["kbt"]
    gamma_dpd = parameters_default["gamma_dpd"]
    gamma_dpd_gas = parameters_default["gamma_dpd_gas"]
    gamma_fsi = parameters["gamma_fsi"]
    gamma_fsi_gas = parameters["gamma_fsi_gas"]
    rc = parameters_default["rc"]
    s = parameters_default["s"]
    s_g = parameters_default["s_g"]
    k_fsi = parameters_default["k_fsi"]
    kbt = parameters["kbt"]
    mvert = parameters["mvert"]
    mw = parameters["mw"]
    mg = parameters["mg"]
    disp = parameters_default["disp"]
    fm_rigid = parameters_default["fm_rigid"]

    pos_q = np.reshape(np.loadtxt(simu_path + "posq.txt"), (-1, 7))
    # this is for z to x rotation
    # axis = [0, 1, 0]
    # angle = 0.0*(np.pi / 2)
    # pos_q = [[0.5 * Lx, 0.5 * Ly, 0.5 * Lz, np.cos(angle / 2), np.sin(angle / 2) * axis[0], np.sin(angle / 2) * axis[1], np.sin(angle / 2) * axis[2]]]
    # pos_q = np.array(pos_q)

    ranks = (1, 1, 1)
    domain = (Lx, Ly, Lz)

    ######################################################
    checkpoint_step = numsteps - 1

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
    else:
        current_device = None
        device_name = "0"
    hostname = os.uname()[1]
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
    if CUDA_VISIBLE_DEVICES is None:
        CUDA_VISIBLE_DEVICES = str(0)

    # Create logs directory for Mirheo (allow override via env)
    log_root = os.environ.get("MIRHEO_LOG_ROOT", "logs")
    log_dir = os.path.join(log_root, hostname, f"GPU{CUDA_VISIBLE_DEVICES}")
    os.makedirs(log_dir, exist_ok=True)

    # mirheo coordinator
    u = mir.Mirheo(
        nranks=ranks,
        domain=domain,
        debug_level=3,
        log_filename=os.path.join(log_dir, "mirheo"),
        checkpoint_folder=simu_path + "restart/",
        checkpoint_every=checkpoint_step,
        no_splash=True,
        comm_ptr=MPI._addressof(comm),
    )

    # loads the off script
    mesh = trimesh.load_mesh(simu_path + objFile)

    # sets the lj_fac to half the minimum edge length
    triangle = mesh.vertices[mesh.faces]
    edge1 = triangle[:, 1] - triangle[:, 0]
    edges = np.linalg.norm(edge1, axis=1)
    lj_fac = 0.8 * np.min(edges)

    # Print all masses
    # if u.isMasterTask():
    #     print(f"Mass of EMB particles: {mvert}")
    #     print(f"Mass of water particles: {mw}")
    #     print(f"Mass of gas particles: {mg}")
    #     print(f"Mass of cantilever and plate particles: {fm_rigid*mvert}")

    # reads vertices, faces
    mesh_emb = mir.ParticleVectors.MembraneMesh(mesh.vertices.tolist(), mesh.faces.tolist())
    emb = mir.ParticleVectors.MembraneVector("emb", mass=mvert, mesh=mesh_emb)

    # initial condition for EMB
    ic_emb = mir.InitialConditions.Membrane(pos_q)
    # if u.isMasterTask():
    #     print('EMB from z =', 0.5*Lz - 1.*radp, 'to z =', 0.5*Lz + 1.*radp)

    # register
    u.registerParticleVector(emb, ic_emb)

    # water
    water = mir.ParticleVectors.ParticleVector(
        "water", mass=mw
    )  # , obmd = obmd_flag) # ne smes imeti istega imena "water" za več "pv"-jev, "water" je ime "pv"-ja znotraj mirhea
    ic_water = mir.InitialConditions.Uniform(number_density=rhow)
    u.registerParticleVector(water, ic_water)

    # solvent
    sol2 = mir.ParticleVectors.ParticleVector("sol2", mass=mg)  # , obmd = obmd_flag)
    ic_outer2 = mir.InitialConditions.Uniform(number_density=rhog)
    u.registerParticleVector(sol2, ic_outer2)

    # splits the particle vector into inner part and outer part defined by the membrane
    # only one can be not null, either inside or outside
    inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
    u.registerObjectBelongingChecker(inner_checker_1, emb)
    gas = u.applyObjectBelongingChecker(
        inner_checker_1, sol2, correct_every=0, inside="gas", outside=""
    )
    # https://mirheo.readthedocs.io/en/latest/user/tutorials.html

    inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
    u.registerObjectBelongingChecker(inner_checker_2, emb)
    u.applyObjectBelongingChecker(
        inner_checker_2, water, correct_every=0, inside="none", outside=""
    )

    radGV = parameters_default["radGV"]

    afsi = 4 * aii

    perfect_gas = True
    if perfect_gas:
        prms_emb["bpress"] = parameters_default["bpress"]
    else:
        prms_emb["bpress"] = 0.0

    if objType == "gv":
        int_emb = mir.Interactions.MembraneForces(
            "int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free=True
        )
    else:
        int_emb = mir.Interactions.MembraneForces(
            "int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free=True
        )

    dpd = mir.Interactions.Pairwise(
        "dpd", rc, kind="DPD", a=0 * aii, gamma=0 * gamma_dpd, kBT=kbt, power=s
    )
    dpd_wat = mir.Interactions.Pairwise(
        "dpd_wat", rc, kind="DPD", a=aii, gamma=gamma_dpd, kBT=kbt, power=s
    )
    dpd_fsi = mir.Interactions.Pairwise(
        "dpd_fsi", rc, kind="DPD", a=afsi, gamma=gamma_fsi, kBT=kbt, power=k_fsi
    )
    if perfect_gas:
        dpd_gas = mir.Interactions.Pairwise(
            "dpd_gas", rc, kind="DPD", a=0 * aii, gamma=gamma_dpd_gas, kBT=kbt, power=s_g
        )
    else:
        dpd_gas = mir.Interactions.Pairwise(
            "dpd_gas", rc, kind="DPD", a=aii, gamma=gamma_dpd_gas, kBT=kbt, power=s_g
        )

    dpd_fsi_gas = mir.Interactions.Pairwise(
        "dpd_fsi_gas", rc, kind="DPD", a=0.0 * aii, gamma=gamma_fsi_gas, kBT=kbt, power=k_fsi
    )

    lj = mir.Interactions.Pairwise(
        "lj",
        rc,
        kind="RepulsiveLJ",
        epsilon=0.1,
        sigma=rc / (2 ** (1 / 6)),
        max_force=10.0,
        aware_mode="Object",
    )
    r_LJ = 1.0 * rc
    # lj_wall = mir.Interactions.Pairwise('lj_wall', r_LJ, kind = "DPD", a = 400.0, gamma = 0.0, kBT = 0.0, power = 1.0) #
    lj_wall = mir.Interactions.Pairwise(
        "lj_wall",
        r_LJ,
        kind="RepulsiveLJ",
        epsilon=1000.0,
        sigma=r_LJ / (2 ** (1 / 6)),
        max_force=1000.0,
    )  #
    # lj_wall = mir.Interactions.Pairwise('lj_wall', r_LJ, kind = "LJ", epsilon = 1000.0, sigma = r_LJ / (2**(1/6))) #
    lj_int = mir.Interactions.Pairwise(
        "lj_int",
        lj_fac,
        kind="RepulsiveLJ",
        epsilon=10000.0,
        sigma=0.99 * lj_fac / (2 ** (1 / 6)),
        max_force=10000.0,
    )
    # dpd_wall = mir.Interactions.Pairwise('dpd_water', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = 0.25)

    # print all interaction parameters
    # if u.isMasterTask():
    #     print("Water-water interaction parameters:")
    #     print(f"  aii = {aii},\tgamma_dpd = {gamma_dpd},\tkBT = {kbt},\ts = {s},\trc = {rc}")
    #     print("Gas-gas interaction parameters:")
    #     print(f"  aii = 0.0,\tgamma_dpd_gas = {gamma_dpd_gas},\tkBT = {kbt},\ts = {s_g},\trc = {rc}")
    #     print("Water-gas interaction parameters:")
    #     print(f"  aij = 0.0,\tgamma_dpd = 0.0,\tkBT = {kbt},\ts = {s},\trc = {rc}")
    #     print("EMB-water interaction parameters:")
    #     print(f"  aij = {afsi},\tgamma_fsi = {gamma_fsi},\tkBT = {kbt},\ts = {k_fsi},\trc = {rc}")
    #     print("EMB-gas interaction parameters:")
    #     print(f"  aij = 0.0,\tgamma_fsi_gas = {gamma_fsi_gas},\tkBT = {kbt},\ts = {k_fsi},\trc = {rc}")

    ######################################## WALLS ########################################
    offset = -0.05  # Initial offset between the EMB and the walls
    # avoids crumpling of the EMB on the contact points

    z0_bottom_emb = 0.5 * Lz - 1.0 * radp
    z0_top_emb = 0.5 * Lz + 1.0 * radp
    canti_width = 1.0

    z0_bottom_plate = z0_bottom_emb - 0.5 * canti_width - 1.0 * r_LJ - offset
    z0_canti = (
        z0_top_emb + 0.5 * canti_width + 1.0 * r_LJ + offset
    )  # Cantilever width is 1.0 ; interaciton radius is rLJ

    ######################################## INTEGRATOR ########################################
    # initialize integrator
    vv = mir.Integrators.VelocityVerlet("vv")

    # register integrator
    u.registerIntegrator(vv)

    # set integrator for various parts
    # if restart:
    #     u.setIntegrator(vv, emb)
    u.setIntegrator(vv, emb)
    u.setIntegrator(vv, water)
    u.setIntegrator(vv, gas)

    # Create the cantilever
    coords_canti = np.loadtxt(simu_path + "mesh/rigid_coords.txt")
    coords_plate = np.loadtxt(simu_path + "mesh/rigid_coords_reflected.txt")
    com_q_canti = [[Lx / 2, Ly / 2, z0_canti, 1.0, 0.0, 0.0, 0.0]]
    com_q_plate = [[Lx / 2, Ly / 2, z0_bottom_plate, 1.0, 0.0, 0.0, 0.0]]

    # if u.isMasterTask():
    #     print('Cantilever from z =', np.min(coords[:,2]) + z0_canti,
    #             'to z =', np.max(coords[:,2]) + z0_canti)

    m_canti = trimesh.load(simu_path + "mesh/cantilever.off")
    inertia = [row[i] for i, row in enumerate(m_canti.moment_inertia)]
    mesh_canti = mir.ParticleVectors.Mesh(m_canti.vertices.tolist(), m_canti.faces.tolist())

    pv_canti = mir.ParticleVectors.RigidObjectVector(
        "cantilever", fm_rigid * mvert, inertia, len(coords_canti), mesh_canti
    )
    ic_canti = mir.InitialConditions.Rigid(com_q_canti, coords_canti.tolist())

    pv_plate = mir.ParticleVectors.RigidObjectVector(
        "plate", fm_rigid * mvert, inertia, len(coords_plate), mesh_canti
    )
    ic_plate = mir.InitialConditions.Rigid(com_q_plate, coords_plate.tolist())

    # Create the integrator for the cantilever
    vv_rigid = mir.Integrators.RigidVelocityVerlet("vv_rigid")

    u.registerParticleVector(pv_canti, ic_canti)
    u.registerParticleVector(pv_plate, ic_plate)
    u.registerIntegrator(vv_rigid)

    u.setIntegrator(vv_rigid, pv_canti)
    u.setIntegrator(vv_rigid, pv_plate)

    ######################################## INTERACTIONS ########################################
    # register interactions
    u.registerInteraction(int_emb)
    u.registerInteraction(dpd_gas)
    u.registerInteraction(dpd)
    u.registerInteraction(dpd_wat)
    u.registerInteraction(dpd_fsi)
    u.registerInteraction(dpd_fsi_gas)
    u.registerInteraction(lj)
    u.registerInteraction(lj_int)

    u.registerInteraction(lj_wall)
    # u.registerInteraction(dpd_wall)

    # set interaction between particle vectors
    u.setInteraction(int_emb, emb, emb)
    u.setInteraction(dpd_wat, water, water)

    u.setInteraction(dpd_gas, gas, gas)
    u.setInteraction(dpd, water, gas)
    u.setInteraction(dpd_fsi, emb, water)
    u.setInteraction(dpd_fsi_gas, emb, gas)

    # Membrane interactions
    # u.setInteraction(lj, emb, emb) # not needed if using one EMB
    u.setInteraction(lj_int, emb, emb)

    ###### Set the interactions with the walls #######
    # EMB only interacts with cantilever and plate
    u.setInteraction(lj_wall, emb, pv_canti)
    u.setInteraction(lj_wall, emb, pv_plate)

    ######################################## REFLECTION BOUNDARIES ########################################
    # reflection boundaries of gas vesicle shells
    # name, coarse, fine, kernel, kBT
    bouncer = mir.Bouncers.Mesh("membrane_bounce", 500, 150, "bounce_maxwell", kBT=kbt)
    # bouncer = mir.Bouncers.Mesh("membrane_bounce", 1000, 150, "bounce_back")#, kBT = 0*kbt)
    u.registerBouncer(bouncer)
    u.setBouncer(bouncer, emb, water)
    u.setBouncer(bouncer, emb, gas)

    ######################################## PREVENT ROTATIONS ########################################

    # EMB should not move
    # Non-rigid objects can't be restricted rotationally
    velocity = [0.0, 0.0, 0.0]
    u.registerPlugins(
        mir.Plugins.createPinObject(
            name="pin_emb",
            ov=emb,
            dump_every=nevery,
            path=f"{simu_path}pinning/emb.csv",
            velocity=velocity,
            angular_velocity=velocity,  # This line is not needed but otherwise Plugins complains
        )
    )

    # Pin the diameter of the EMB
    # pos, ind = findpids(mesh.vertices, 0.0, 0.0)

    # def velocities(t):
    #     velo = np.array(emb.getVelocities())
    #     #print("velo =", velo)
    #     #print("ind =", ind)
    #     velo[ind,2] = 0.0 # Set the z-component of the velocity to 0
    #     return list(velo[ind,:])

    # def positions(t):
    #     posi = np.array(emb.getCoordinates())
    #     #print("posi.shape =", posi.shape, "\nposi =", posi)
    #     #print("len(ind) =", len(ind), "\nind =", ind)
    #     posi[ind,2] = 0.5*Lz # Set the z-component of the position to 0.5*Lz
    #     return list(posi[ind,:])

    # u.registerPlugins(
    #     mir.Plugins.createAnchorParticles(
    #         name          = "anchor_emb",
    #         pv            = emb,
    #         positions     = positions,
    #         velocities    = velocities,
    #         pids          = ind,
    #         report_every  = nevery,
    #         path          = f"{simu_path}anchor/emb.csv",
    #     )
    # )

    ########################### PLUGINS ###########################
    debug = config["debug"] >= 1

    if debug:
        # Output the stats of the cantilever
        u.registerPlugins(
            mir.Plugins.createDumpObjectStats(
                name="objStats_canti",
                ov=pv_canti,
                dump_every=1,
                filename=f"{simu_path}stats/cantilever.csv",
            )
        )

        # Output the stats of the EMB
        u.registerPlugins(
            mir.Plugins.createDumpObjectStats(
                name="objStats_emb", ov=emb, dump_every=1, filename=f"{simu_path}stats/emb.csv"
            )
        )

    # else:
    #     # Output the stats of the cantilever
    #     u.registerPlugins(
    #         mir.Plugins.createDumpObjectStats(
    #             name        = "objStats_canti",
    #             ov          = pv_canti,
    #             dump_every  = nevery,
    #             filename    = f"{simu_path}stats/cantilever.csv"
    #         )
    #     )

    #     # Output the stats of the plate
    #     u.registerPlugins(
    #         mir.Plugins.createDumpObjectStats(
    #             name        = "objStats_plate",
    #             ov          = pv_plate,
    #             dump_every  = nevery,
    #             filename    = f"{simu_path}stats/plate.csv"
    #         )
    #     )

    ################################ First apply the displacement ######################################
    # # DEBUGGING PLUGING
    # u.registerPlugins(
    #     mir.Plugins.createParticleChecker(
    #         name = "checker",
    #         check_every = 1,
    #     )
    # )

    dz_lin = disp
    vc = dz_lin / (
        2.0 * numsteps_eq * dt_eq
    )  # velocity to reach the deformation (each plate moves by half dz_lin)
    omega = [0.0, 0.0, 0.0]

    velocity = [0.0, 0.0, -vc]
    pin_canti_eq = mir.Plugins.createPinObject(
        "pin_canti_eq", pv_canti, nevery_eq, f"{simu_path}pinning_eq/", velocity, omega
    )
    u.registerPlugins(pin_canti_eq)

    # Bottom plate can move only along the z-axis
    omega = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, vc]
    pin_plate_eq = mir.Plugins.createPinObject(
        "pin_plate_eq", pv_plate, nevery_eq, f"{simu_path}pinning_eq", velocity, omega
    )
    u.registerPlugins(pin_plate_eq)

    if restart:
        # if u.isMasterTask():
        #     print(f"Restarting simulation from {restart_path} + 'restart/'")
        # u.deregisterPlugins(pin_canti)
        # u.deregisterPlugins(pin_plate)
        u.restart(restart_path + "restart/")

    if debug:
        u.registerPlugins(mir.Plugins.createStats("stats_debug", every=1))
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_emb_debug", emb, 1, f"{simu_path}trj_eq/sim{simnum}"
            )
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_water_debug", water, 1, f"{simu_path}trj_eq/sim{simnum}"
            )
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_gas_debug", gas, 1, f"{simu_path}trj_eq/sim{simnum}"
            )
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_canti_debug", pv_canti, 1, f"{simu_path}trj_eq/sim{simnum}"
            )
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_wall_debug", pv_plate, 1, f"{simu_path}trj_eq/sim{simnum}"
            )
        )

    elif dump:
        u.registerPlugins(mir.Plugins.createStats("stats_debug", every=nevery))
        u.registerPlugins(
            mir.Plugins.createDumpXYZ("xyz_dump", emb, nevery, f"{simu_path}trj_eq/sim{simnum}")
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_wall", pv_plate, nevery, f"{simu_path}trj_eq/sim{simnum}"
            )
        )
        u.registerPlugins(
            mir.Plugins.createDumpXYZ(
                "xyz_dump_canti", pv_canti, nevery, f"{simu_path}trj_eq/sim{simnum}"
            )
        )

    # Run with error handling that aborts all ranks in this worker on error
    try:
        u.run(numsteps_eq, dt=dt_eq)
    except RuntimeError as e:
        # Print error and abort this worker's MPI ranks to prevent deadlock
        # comm.Abort only kills ranks in this worker's communicator, not all Korali workers
        if comm.Get_rank() == 0:
            error_msg = str(e)
            print(f"[Mirheo Error] {error_msg}")
            print("[MPI] Aborting worker ranks due to simulation error")
        comm.Abort(1)

    ######################################## RUN SAMPLING ########################################
    omega = [0.0, 0.0, 0.0]
    velocity = [0.0, 0.0, 0.0]

    u.deregisterPlugins(pin_canti_eq)
    pin_canti = mir.Plugins.createPinObject(
        "pin_canti", pv_canti, nevery, f"{simu_path}pinning/", velocity, omega
    )
    u.registerPlugins(pin_canti)

    u.deregisterPlugins(pin_plate_eq)
    pin_plate = mir.Plugins.createPinObject(
        "pin_plate", pv_plate, nevery, f"{simu_path}pinning/", velocity, omega
    )
    u.registerPlugins(pin_plate)

    # Run with error handling that aborts all ranks in this worker on error
    try:
        u.run(numsteps, dt=dt)
    except RuntimeError as e:
        # Print error and abort this worker's MPI ranks to prevent deadlock
        if comm.Get_rank() == 0:
            error_msg = str(e)
            print(f"[Mirheo Error] {error_msg}")
            print("[MPI] Aborting worker ranks due to simulation error")
        comm.Abort(1)

    del u
    return 0


def main(argv):
    import argparse

    ######################################################
    # set-up simulation type: equilibration or restart

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--equil", action="store_true", default=None)
    group.add_argument("--restart", action="store_true", default=None)
    parser.add_argument("--simnum", dest="simnum", default="00001")

    args = parser.parse_args()

    run_equil(
        source_path="",
        simu_path="",
        simnum=args.simnum,
        equil=args.equil,
        restart=args.restart,
        restart_path="",
        comm=MPI.COMM_WORLD,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
