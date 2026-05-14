"""
Mirheo DPD simulation driver for EMB indentation experiments.

Requires: mirheo (Python 3.8, GPU), mpi4py, trimesh, numpy, pyyaml.
Source: Hierarchical_UQ_compression_dev/emb/indentation/src/equil.py
"""

import sys
from pathlib import Path

import mirheo as mir
import numpy as np
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
    vacuum: bool,
    comm: MPI.Comm,
):

    # print(f"equil: {equil}, restart: {restart}, vacuum: {vacuum}")

    ######################################################
    # set-up parameters

    filename_default = simu_path + "parameter/parameters-default" + simnum + ".yaml"
    with open(filename_default, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)

    from pathlib import Path

    emb_path = Path(__file__).resolve().parent / "parameters-default.emb.yaml"
    if (
        ("gamma_dpd_gas" not in parameters_default)
        or ("s_g" not in parameters_default)
        or ("k_fsi" not in parameters_default)
        or ("lj_fac" not in parameters_default)
        or ("bpress" not in parameters_default)
    ):
        with open(emb_path, "rb") as f:
            emb_defaults = yaml.load(f, Loader=yaml.CLoader)

        for k in ("gamma_dpd_gas", "s_g", "k_fsi", "lj_fac", "bpress"):
            if k not in parameters_default and k in emb_defaults:
                parameters_default[k] = emb_defaults[k]

    filename = simu_path + "parameter/parameters" + simnum + ".yaml"
    with open(filename, "rb") as f:
        parameters = yaml.load(f, Loader=yaml.CLoader)

    filename_prms = simu_path + "parameter/parameters.prms" + simnum + ".yaml"

    with open(filename_prms, "rb") as f:
        prms_emb = yaml.load(f, Loader=yaml.CLoader)

    def computeForces(vertices, fraction, force):
        vertices = np.array(vertices)

        k = int(fraction * 0.5 * len(vertices))

        ind_max = np.argpartition(+vertices[:, 2], -k)[-k:]
        ind_min = np.argpartition(-vertices[:, 2], -k)[-k:]

        # print(ind_max, ind_min)

        forces = np.zeros((len(vertices), 3))

        forces[ind_max, 2] = +force
        forces[ind_min, 2] = -force
        return forces

    def findPoles(vertices):
        vertices = np.array(vertices)

        id_max = np.argmax(vertices[:, 2])
        id_min = np.argmin(vertices[:, 2])

        return id_min, id_max

    objType = (parameters_default["objFile"])[0:-4]
    objFile = "mesh/" + objType + simnum + ".off"

    numsteps = parameters_default["numsteps"] if restart else parameters_default["numsteps_eq"]
    dt = parameters_default["dt"] if restart else parameters_default["dt_eq"]
    Lx = parameters_default["Lx"]
    Ly = parameters_default["Ly"]
    Lz = parameters_default["Lz"]
    nevery = parameters["nevery"] if restart else parameters["nevery_eq"]
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

    pos_q = np.reshape(np.loadtxt(simu_path + "posq.txt"), (-1, 7))

    ranks = (1, 1, 1)
    domain = (Lx, Ly, Lz)

    ######################################################
    checkpoint_step = numsteps - 1

    # mirheo coordinator
    mirheo_log_path = str(Path(simu_path) / "logs" / "mirheo")

    u = mir.Mirheo(
        nranks=ranks,
        domain=domain,
        debug_level=0,
        log_filename=mirheo_log_path,
        checkpoint_folder=simu_path + "restart/",
        checkpoint_every=checkpoint_step,
        no_splash=True,
        comm_ptr=MPI._addressof(comm),
    )

    # if u.isMasterTask():
    # 	print("force =", force)
    # print("Loading mesh from file:", simu_path + objFile)
    # loads the off script

    mesh = trimesh.load_mesh(simu_path + objFile)

    # sets the lj_fac to half the minimum edge length
    triangle = mesh.vertices[mesh.faces]
    edge1 = triangle[:, 1] - triangle[:, 0]
    edges = np.linalg.norm(edge1, axis=1)
    lj_fac = 0.8 * np.min(edges)

    # reads vertices, faces
    mesh_emb = mir.ParticleVectors.MembraneMesh(
        vertices=mesh.vertices.tolist(),
        stress_free_vertices=mesh.vertices.tolist(),
        faces=mesh.faces.tolist(),
    )

    emb = mir.ParticleVectors.MembraneVector("emb", mass=mvert, mesh=mesh_emb)

    # initial condition for GV
    ic_emb = mir.InitialConditions.Membrane(pos_q)

    # register
    u.registerParticleVector(emb, ic_emb)

    if not vacuum:
        # water
        water = mir.ParticleVectors.ParticleVector("water", mass=mw)
        ic_water = mir.InitialConditions.Uniform(number_density=rhow)
        u.registerParticleVector(water, ic_water)

        # solvent
        sol2 = mir.ParticleVectors.ParticleVector("sol2", mass=mg)
        ic_outer2 = mir.InitialConditions.Uniform(number_density=rhog)
        u.registerParticleVector(sol2, ic_outer2)

        # splits the particle vector into inner part and outer part defined by the membrane
        # only one can be not null, either inside or outside
        inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
        u.registerObjectBelongingChecker(inner_checker_1, emb)
        gas = u.applyObjectBelongingChecker(
            inner_checker_1, sol2, correct_every=0, inside="gas", outside=""
        )  # correct_every = 100000
        # https://mirheo.readthedocs.io/en/latest/user/tutorials.html

        inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
        u.registerObjectBelongingChecker(inner_checker_2, emb)
        u.applyObjectBelongingChecker(
            inner_checker_2, water, correct_every=0, inside="none", outside=""
        )

    # interactions

    afsi = 0.4 * aii
    # if(args.vacuum):
    #    prms_emb["gammaC"] = gamma_dpd
    prms_emb["bpress"] = parameters_default["bpress"]
    if objType == "gv":
        int_emb = mir.Interactions.MembraneForces(
            "int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free=True
        )
    else:
        int_emb = mir.Interactions.MembraneForces(
            "int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free=True
        )

    # if u.isMasterTask():
    # 	print("EMB parameters:", prms_emb)

    if vacuum:
        dpd0 = mir.Interactions.Pairwise(
            "dpd0", rc, kind="DPD", a=0.0, gamma=3 * gamma_dpd, kBT=0.015 * kbt, power=s
        )
    dpd = mir.Interactions.Pairwise(
        "dpd", rc, kind="DPD", a=0 * aii, gamma=0 * gamma_dpd, kBT=kbt, power=s
    )
    dpd_wat = mir.Interactions.Pairwise(
        "dpd_wat", rc, kind="DPD", a=aii, gamma=gamma_dpd, kBT=kbt, power=s
    )
    dpd_gas = mir.Interactions.Pairwise(
        "dpd_gas", rc, kind="DPD", a=0 * aii, gamma=gamma_dpd_gas, kBT=kbt, power=s_g
    )
    dpd_fsi = mir.Interactions.Pairwise(
        "dpd_fsi", rc, kind="DPD", a=afsi, gamma=gamma_fsi, kBT=kbt, power=k_fsi
    )
    dpd_fsi_gas = mir.Interactions.Pairwise(
        "dpd_fsi_gas", rc, kind="DPD", a=0 * afsi, gamma=gamma_fsi_gas, kBT=kbt, power=k_fsi
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
    lj_int = mir.Interactions.Pairwise(
        "lj_int",
        lj_fac * rc,
        kind="RepulsiveLJ",
        epsilon=10000.0,
        sigma=lj_fac * rc / (2 ** (1 / 6)),
        max_force=100000.0,
    )

    ######################################## INTEGRATOR ########################################
    # initialize integrator
    vv = mir.Integrators.VelocityVerlet("vv")

    # register integrator
    u.registerIntegrator(vv)

    # set integrator for various parts
    u.setIntegrator(vv, emb)

    if not vacuum:
        u.setIntegrator(vv, water)
        u.setIntegrator(vv, gas)

    ######################################## INTERACTIONS ########################################
    # register interactions
    u.registerInteraction(int_emb)
    if vacuum:
        u.registerInteraction(dpd0)
    u.registerInteraction(dpd)
    u.registerInteraction(dpd_wat)
    u.registerInteraction(dpd_gas)
    u.registerInteraction(dpd_fsi)
    u.registerInteraction(dpd_fsi_gas)
    u.registerInteraction(lj)
    u.registerInteraction(lj_int)

    # set interaction
    u.setInteraction(int_emb, emb, emb)
    u.setInteraction(lj, emb, emb)
    u.setInteraction(lj_int, emb, emb)
    if not vacuum:
        u.setInteraction(dpd_wat, water, water)
        u.setInteraction(dpd_gas, gas, gas)
        u.setInteraction(dpd, water, gas)
        u.setInteraction(dpd_fsi, emb, water)
        u.setInteraction(dpd_fsi_gas, emb, gas)

    if vacuum:
        u.setInteraction(dpd0, emb, emb)

    # Pin the diameter of the EMB
    pos, ind = findpids(mesh.vertices, 0.0, 0.0)
    # vel = np.zeros((len(pos), 3))
    # cen = np.array([0.5 * Lx, 0.5 * Ly, 0.5 * Lz])

    def velocities(t):
        velo = np.array(emb.getVelocities())
        # print("velo =", velo)
        # print("ind =", ind)
        velo[ind, 2] = 0.0  # Set the z-component of the velocity to 0
        return list(velo[ind, :])

    def positions(t):
        posi = np.array(emb.getCoordinates())
        # print("posi.shape =", posi.shape, "\nposi =", posi)
        # print("len(ind) =", len(ind), "\nind =", ind)
        posi[ind, 2] = 0.5 * Lz  # Set the z-component of the position to 0.5*Lz
        return list(posi[ind, :])

    u.registerPlugins(
        mir.Plugins.createAnchorParticles(
            name="anchor_emb",
            pv=emb,
            positions=positions,
            velocities=velocities,
            pids=ind,
            report_every=nevery,
            path=f"{simu_path}anchor/emb.csv",
        )
    )

    ######################################## REFLECTION BOUNDARIES ########################################
    # reflection boundaries of gas vesicle shells

    if not vacuum:
        bouncer = mir.Bouncers.Mesh("membrane_bounce", 1000, 150, "bounce_maxwell", kBT=kbt)
        u.registerBouncer(bouncer)
        u.setBouncer(bouncer, emb, water)
        u.setBouncer(bouncer, emb, gas)

    ######################################## RUN ########################################

    fraction = parameters_default["fraction"]
    force = parameters_default["force"] / int(0.5 * fraction * len(mesh.vertices))
    # if u.isMasterTask():
    forces = computeForces(mesh.vertices, fraction, force).tolist()
    ind_min, ind_max = findPoles(mesh.vertices)

    if equil:
        # print('equilibration')
        # print('Output files will be saved in:', simu_path + 'trj_eq/sim' + simnum)
        unr = mir.Plugins.PinObject.Unrestricted
        omega = [unr, unr, unr]
        velocity = [0.0, 0.0, 0.0]
        u.registerPlugins(
            mir.Plugins.createPinObject("pin", emb, nevery, f"{simu_path}pin", velocity, omega)
        )
        forces = computeForces(mesh.vertices, fraction, force).tolist()
        # u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
        # u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"{simu_path}trj_eq/sim{simnum}"))
        u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
        u.registerPlugins(
            mir.Plugins.createDumpParticles(
                name="dumpParticles",
                pv=emb,
                dump_every=nevery,
                channel_names=["positions"],
                path=f"{simu_path}particles/emb",
            )
        )
        u.run(numsteps, dt=dt)
        if u.isMasterTask():
            np.savetxt(
                simu_path + "posfin.txt", np.array(emb.getCoordinates())[[ind_min, ind_max], :]
            )
            np.savetxt(simu_path + "ind_poles.txt", np.array([ind_min, ind_max]))
        del u

    if restart:
        # print('restart from', restart_path + "restart/")
        u.restart(restart_path + "restart/")
        unr = mir.Plugins.PinObject.Unrestricted
        omega = [unr, unr, unr]
        velocity = [0.0, 0.0, 0.0]
        u.registerPlugins(
            mir.Plugins.createPinObject("pin", emb, nevery, f"{simu_path}pin", velocity, omega)
        )
        forces = computeForces(mesh.vertices, fraction, force).tolist()
        # u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))
        # u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"{simu_path}trj_eq/sim{simnum}"))
        u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))
        u.registerPlugins(
            mir.Plugins.createDumpParticles(
                name="dumpParticles",
                pv=emb,
                dump_every=nevery,
                channel_names=["positions"],
                path=f"{simu_path}particles/emb",
            )
        )
        u.run(numsteps, dt=dt)
        if u.isMasterTask():
            np.savetxt(
                simu_path + "posfin.txt", np.array(emb.getCoordinates())[[ind_min, ind_max], :]
            )
            np.savetxt(simu_path + "ind_poles.txt", np.array([ind_min, ind_max]))
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
    parser.add_argument("--vacuum", action="store_true", default=None)

    args = parser.parse_args()

    run_equil(
        source_path="",
        simu_path="",
        simnum=args.simnum,
        equil=args.equil,
        restart=args.restart,
        restart_path="",
        vacuum=args.vacuum,
        comm=MPI.COMM_WORLD,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
