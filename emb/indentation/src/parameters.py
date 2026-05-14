#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path


def write_parameters(source_path, simu_path, simnum):
    import numpy as np
    import trimesh
    import yaml
    from scipy.stats import qmc

    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "inference" / "configs" / "production" / "inference_config_indentation.yaml"
    with open(cfg_path, "rb") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    filename = simu_path + "parameter/parameters" + simnum + ".yaml"
    filename_prms = simu_path + "parameter/parameters.prms" + simnum + ".yaml"
    src_dir = Path(__file__).resolve().parent
    emb_defaults_path = src_dir / "parameters-default.emb.yaml"
    sim_defaults_path = Path(simu_path) / "parameter" / f"parameters-default{simnum}.yaml"
    with open(emb_defaults_path, "rb") as f:
        parameters_default = yaml.load(f, Loader=yaml.CLoader)
    with open(sim_defaults_path, "rb") as f:
        sim_defaults = yaml.load(f, Loader=yaml.CLoader)
    parameters_default.update(sim_defaults)

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
    ul = parameters_default["ul"]
    ue = energyFactor * kbol * t0
    um = rho_water * ul**3 / rhow
    ut = np.sqrt(um * ul**2 / ue)
    kbt = 1 / energyFactor
    uvis = um / ul / ut
    visw = parameters_default["visw"]
    visg = parameters_default["visg"]
    visw_dpd = visw / uvis
    visg_dpd = visg / uvis
    mw = rho_water * ul**3 / (rhow * um)
    mg = rho_gas * ul**3 / (rhog * um)

    objType = (parameters_default["objFile"])[0:-4]
    if objType != "emb":
        raise NotImplementedError("This public runtime-preparation slice currently supports emb object preparation only.")
    subprocess.run([
        sys.executable, "./sphere_icosphere.py", "--simnum", simnum,
        "-o", f"../mesh/emb{simnum}.off", "-s", str(int(parameters_default["subDiv"])),
        "-r", str(parameters_default["radp"]),
    ], check=True, cwd=os.path.join(simu_path, "microbubble"))

    objFile = simu_path + "mesh/" + objType + simnum + ".off"
    mesh = trimesh.load(objFile)
    nverts = len(mesh.vertices)
    rho_shell = parameters_default["rho_shell"]
    shell_th = parameters_default["th_fac"] * parameters_default["shell_th"]
    tot_area = mesh.area
    tot_volume = abs(mesh.volume)
    mvert = rho_shell * shell_th * tot_area * ul**2 / um / nverts
    mvert *= 5.0

    ka_tot = parameters_default["ka_tot"]
    kv_tot = parameters_default["kv_tot"]
    gammaC = 2 * parameters_default["gamma_dpd"]
    kBT = kbt
    fscale = parameters_default["fscale"]
    Yt = parameters_default["yt_fac"] * parameters_default["Yt"]
    Yl = parameters_default["Yl"]
    nu = parameters_default["nu"]
    rc = parameters_default["rc"]
    k_fsi = parameters_default["k_fsi"]
    rho_surf = nverts / tot_area
    gamma_fsi = fscale * 2.0 * visw_dpd * (2*k_fsi+1)*(2*k_fsi+2)*(2*k_fsi+3)*(2*k_fsi+4) / (3 * np.pi * rc**4 * rhow * rho_surf)
    gamma_fsi_gas = fscale * 2.0 * visg_dpd * (2*k_fsi+1)*(2*k_fsi+2)*(2*k_fsi+3)*(2*k_fsi+4) / (3 * np.pi * rc**4 * rhog * rho_surf)
    ka = fscale * Yt * shell_th / (2 * (1 - nu)) / (ue / ul**2)
    mu = fscale * Yt * shell_th / (2 * (1 + nu)) / (ue / ul**2)
    a3 = parameters_default["a3"]
    a4 = parameters_default["a4"]
    b1 = parameters_default["b1"]
    b2 = parameters_default["b2"]
    kb_fac = parameters_default["kb_fac"]
    kb = kb_fac * fscale * 2.0 / np.sqrt(3) * Yl * shell_th**3 / (12 * (1 - nu**2)) / ue

    prms_emb = {k: float(v) for k, v in {
        "ka_tot": ka_tot, "kv_tot": kv_tot, "gammaC": gammaC, "kBT": kBT,
        "tot_area": tot_area, "tot_volume": tot_volume, "kb": kb, "ka": ka,
        "mu": mu, "a3": a3, "a4": a4, "b1": b1, "b2": b2,
    }.items()}
    with open(filename_prms, "w") as f:
        yaml.dump(prms_emb, f)

    numObjects = parameters_default["numObjects"]
    m = int(np.log2(numObjects))
    sampler = qmc.Sobol(d=3, scramble=True)
    sample = sampler.random_base2(m=m)
    fac = 0.6
    sample[:, 0] = 0.5 * (1 - fac) * Lx + sample[:, 0] * fac * Lx
    sample[:, 1] = 0.5 * (1 - fac) * Ly + sample[:, 1] * fac * Ly
    sample[:, 2] = 0.5 * (1 - fac) * Lz + sample[:, 2] * fac * Lz
    pos_q = []
    if len(sample[:,0]) == 1:
        pos_q = [[0.5 * Lx, 0.5 * Ly, 0.5 * Lz, 1, 0, 0, 0]]
    else:
        for i in range(len(sample[:,0])):
            u = np.random.random(); v = np.random.random(); w = np.random.random()
            quatr = [np.sqrt(1-u)*np.sin(2*np.pi*v), np.sqrt(1-u)*np.cos(2*np.pi*v), np.sqrt(u)*np.sin(2*np.pi*w), np.sqrt(u)*np.cos(2*np.pi*w)]
            pos_q.append([sample[i,0], sample[i,1], sample[i,2], quatr[0], quatr[1], quatr[2], quatr[3]])
    np.savetxt(simu_path + "posq.txt", pos_q)

    nevery = int(parameters_default["numsteps"] / parameters_default["stslik"])
    nevery_eq = int(parameters_default["numsteps_eq"] / parameters_default["stslik_eq"])
    parameters = {
        "ut": float(ut), "ue": float(ue), "um": float(um), "uvis": float(uvis),
        "visw_dpd": float(visw_dpd), "visg_dpd": float(visg_dpd), "kbt": float(kbt),
        "nevery": int(nevery), "nevery_eq": int(nevery_eq), "nverts": int(nverts),
        "mw": float(mw), "mg": float(mg), "mvert": float(mvert), "tot_area": float(tot_area),
        "tot_volume": float(tot_volume), "gamma_fsi": float(gamma_fsi), "gamma_fsi_gas": float(gamma_fsi_gas),
        "ka": float(ka), "mu": float(mu), "kb": float(kb),
    }
    with open(filename, "w") as f:
        yaml.dump(parameters, f)


def main(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simnum", dest="simnum", default="00001")
    args = parser.parse_args()
    write_parameters(source_path="", simu_path="", simnum=args.simnum)


if __name__ == "__main__":
    main(sys.argv[1:])
