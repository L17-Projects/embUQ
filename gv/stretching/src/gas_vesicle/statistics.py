import numpy as np
from parameters import *
import collections
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
args = parser.parse_args()

filename_default = '../parameter/parameters-default' + args.simnum + '.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)


radius = parameters_default["radGV"]
height = parameters_default["height"]

def read_off_file(filename):
    points = []
    indices = []
    file = open(filename,'r')
    p = 0
    for line in file:
        p = p + 1
        if(p<3):
            continue
        linetmp = line.split()
        if(len(linetmp)==3):
            points.append([float(linetmp[0]),float(linetmp[1]),float(linetmp[2])])
        elif(len(linetmp)==4):
            indices.append([int(linetmp[1]),int(linetmp[2]),int(linetmp[3])])
    file.close()
    return np.array(points),np.array(indices)

vertices, T = read_off_file('out.off')

faces = vertices[T]

area = []
for f in faces:
    vec = np.cross(f[1]-f[0],f[2]-f[0])
    area.append(np.linalg.norm(vec)/2)

cms = np.array([0,0,0.5*height])
volume = []
for f in faces:
    loc_cms = (f[0]+f[1]+f[2])/3.0-cms      #triangle cms #loc_cms actually does not matter, as long as it is one of the vectors
    vec = np.cross(f[1]-f[0],f[2]-f[0])     #normal
    volume_tmp = np.dot(loc_cms,vec)/6      #signed volume tetrahedron (negative)
    #print(f'partial volume = {volume_tmp}')
    volume.append(volume_tmp)

def find_neighbors(T, points):
    stats = []
    neighbors = {}
    for point in range(points.shape[0]):
        neighbors[point] = []
    for simplex in T:
        neighbors[simplex[0]] += [simplex[1],simplex[2]]
        neighbors[simplex[1]] += [simplex[2],simplex[0]]
        neighbors[simplex[2]] += [simplex[0],simplex[1]]
    for n in neighbors.values():
        stats.append(len(list(dict.fromkeys(n))))
    counter = collections.Counter(stats)
    return stats, counter

stats,counter = find_neighbors(T,vertices)

import trimesh

mesh = trimesh.load('out.off')

vert = mesh.vertices

mesh.vertices  = mesh.vertices  - np.mean(mesh.vertices, axis=0)

mesh.export('out.off')

print(f'''Average triangle area = {np.mean(area)} +- {np.std(area)}.
Total area = {np.sum(area)}.
Total volume = {-np.sum(volume)}.
Orientation of triangles: clockwise (normals point inside).
Nt = {int(len(area))} triangles
Ne = {int(len(area))+int(len(vertices))-2} edges.
Nv = {int(len(vertices))} vertices.
Nv - Ne + Nt = 2.
Number of neighbors: {np.mean(stats)} +- {np.std(stats)}
Neighbor statistics {{No. of neighbors: frequency}}: {dict(counter)}''')
