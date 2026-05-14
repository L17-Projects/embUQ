import argparse

import matplotlib.pyplot as plt
import trimesh
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--simnum", dest="simnum", default="00001")
args = parser.parse_args()

filename_default = "../parameter/parameters-default" + args.simnum + ".yaml"
with open(filename_default, "rb") as f:
    parameters_default = yaml.load(f, Loader=yaml.CLoader)

radius = parameters_default["radGV"]
height = parameters_default["height"]

m = trimesh.load("out.of")
vertices = m.vertices
T = m.faces

x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot_trisurf(x, y, z, triangles=T, edgecolor="gray", linewidth=1, alpha=0.99)
ax.scatter(x, y, z, s=1, color="red")

ax.set_box_aspect((radius, radius, height))

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.grid(False)
ax.set_axis_off()

# ax.set_xlim(-1,1)
# ax.set_ylim(-1,1)
# ax.set_zlim(-10,10)
plt.savefig("gv.png", dpi=300, bbox_inches="tight")

# plt.show()
