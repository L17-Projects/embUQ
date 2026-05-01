import numpy as np
#from parameters import *
import argparse
import yaml
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
args = parser.parse_args()

filename_default = '../parameter/parameters-default' + args.simnum + '.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

filename = 'parameters.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)

def make_off_file_for_cgal(filename,points):
    file = open(filename,'w')
    file.write('OFF\n')
    file.write(f'{int(len(points))} 0 0\n')
    for p in points:
        file.write(f'{p[0]} {p[1]} {p[2]}\n')
    file.close()

def flatten_list(obj):
    obj_tmp = [x for xs in obj for x in xs]
    return obj_tmp

def gen_cylinder(radius, height, k, rho, shift = 0.5):
    #shift = 0, 1 nonstagerred
    #shift = 0.5 staggered
    #k number of layers
    #rho density of points
    z0 = np.linspace(0, height, k)
    npts = int(rho * 2 * np.pi * radius)
    fi0 = np.linspace(0, 2 * np.pi, npts, endpoint = False)
    dfi = 2 * np.pi / npts
    arr0 = []
    for i in range(k):
        fi = fi0 + 0.5 * (1 - (-1)**i) * dfi * shift
        #fi = fi0 + i * dfi * shift
        x = radius * np.cos(fi)
        y = radius * np.sin(fi)
        z = z0[i] * np.ones(len(fi))
        circle = np.column_stack((x, y, z))
        arr0.append(circle)
        #print(np.column_stack((x, y, z)))
    #print(np.array(arr0)[1])
    return np.array(arr0).reshape((-1, 3))

def gen_cone(radius, height, k, rho, sign = 1, shift = 0.5):
    #shift = 0, 1 nonstagerred
    #shift = 0.5 staggered
    #k number of layers
    #rho density of points
    z0 = sign * np.linspace(0, height, k)
    dh = height / (k - 1)
    arr0 = []
    for i in range(k - 1):
        radtmp = radius * (1 - i / (k - 1))
        npts = max(int(2 * rho * 2 * np.pi * radtmp), 3)

        dfi = 2 * np.pi / npts
        fi0 = np.linspace(dfi, 2 * np.pi + dfi, npts, endpoint = False)

        fi = fi0 + 0.5 * (1 - (-1)**(i+1)) * dfi * shift
        #fi = fi0 + i * dfi * shift
        x = radtmp * np.cos(fi)
        y = radtmp * np.sin(fi)
        z = z0[i] * np.ones(len(fi))
        circle = np.column_stack((x, y, z))
        arr0.append(circle)
    arr_out = [x for xs in arr0 for x in xs]
    arr_out.append([0, 0, sign * (height - dh/2)])
    return np.array(arr_out).reshape((-1, 3))

radius0 = parameters_default["radGV"]
height0 = parameters_default["height"]
rho0 = parameters["den_fac"]
k0 = parameters["k0"]
frac = parameters["frac"]
scale = parameters["scale"]
af1 = parameters["af1"]
staggered = parameters["staggered"]

# this k0 works well. comment it out if you want your own k0 from parameters_default["k0"]
k0 = int(80 / 14.28 * height0)   #80/14.28 empirical factor that works well

shift0 = (0.5 if staggered else 0.0)

zvec = np.array([0, 0, 1])

k0_cyl = int(k0 * (1 - 2 * frac))
cone_height = height0 * frac
cyl_height = height0 * (1 - 2 * frac)

dh = height0 / (k0 - 1)
k0_cone = int(1.2 * np.sqrt(radius0**2 + (frac * height0)**2) / dh)
ratio = 1 / np.sqrt(1 + radius0**2 / (frac * height0)**2)
ktmp0 = k0 * (1-2*frac)/(2 * frac / ratio +1 - 2 * frac)


cone_bottom = gen_cone(radius = radius0, height = cone_height, k = k0_cone, rho = rho0, sign = -1, shift = shift0)
cylinder = gen_cylinder(radius = radius0, height = cyl_height, k = k0_cyl, rho = rho0, shift = shift0)
cylinder = cylinder + dh * zvec
cone_top = gen_cone(radius = radius0, height = cone_height, k = k0_cone, rho = rho0, sign = 1, shift = shift0)
cone_top = cone_top + cyl_height * zvec + 2 * dh * zvec

obj = []
obj.append(cone_bottom.tolist())
obj.append(cylinder.tolist())
obj.append(cone_top.tolist())
obj = flatten_list(obj)
arr = np.array(obj).reshape((-1, 3))
arr_center = np.mean(arr, axis = 0)
arr = arr - arr_center

make_off_file_for_cgal('test.off', arr)


def _find_scale_space_tool() -> Path:
    candidates = []
    env_binary = os.getenv("GV_SCALE_SPACE_BINARY")
    if env_binary:
        candidates.append(Path(env_binary).expanduser())

    env_root = os.getenv("GV_CGAL_TOOLS_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        candidates.append(root / "scale_space")
        candidates.append(root / "cgal_scripts" / "scale_space")

    script_root = Path(__file__).resolve().parent
    candidates.append(script_root / "cgal_scripts" / "scale_space")
    candidates.append(script_root / "build" / "cgal_scripts" / "scale_space")

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    system_path = shutil.which("scale_space")
    if system_path is not None:
        return Path(system_path)

    inspected_paths = ", ".join(str(path) for path in candidates)
    raise RuntimeError(
        "Required dependency `scale_space` was not found.\n"
        "Set GV_SCALE_SPACE_BINARY or GV_CGAL_TOOLS_ROOT, or install it in PATH.\n"
        f"Searched: {inspected_paths}"
    )


scale_space_exe = _find_scale_space_tool()

print('Running scale_space')

result = subprocess.run([str(scale_space_exe), str(scale), str(af1)])
if result.returncode != 0:
    raise RuntimeError(
        f"scale_space exited with status {result.returncode} when processing {af1}."
    )

Path("test.off").unlink(missing_ok=True)
