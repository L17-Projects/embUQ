import numpy as np
from parameters import *

_PAPER_SIGNED_VOLUME = -1.0


def read_test_off_file(filename):
    points = []
    file = open(filename,'r')
    p = 0
    for line in file:
        p = p + 1
        if(p<3):
            continue
        linetmp = line.split()
        points.append([float(linetmp[0]),float(linetmp[1]),float(linetmp[2])])
    file.close()
    return np.array(points)

def read_out_off_file(filename):
    indices = []
    file = open(filename,'r')
    for line in file:
        linetmp = line.split()
        indices.append([int(linetmp[1]),int(linetmp[2]),int(linetmp[3])])
    file.close()
    return np.array(indices)

def signed_mesh_volume(points, indices):
    if len(indices) == 0:
        return 0.0
    triangle = points[indices]
    return float(np.einsum("ij,ij->", triangle[:, 0], np.cross(triangle[:, 1], triangle[:, 2])) / 6.0)

def orient_faces_to_paper_winding(indices, points):
    signed_volume = signed_mesh_volume(points, indices)
    if signed_volume == 0.0 or signed_volume * _PAPER_SIGNED_VOLUME > 0.0:
        return indices
    oriented = np.array(indices, copy=True)
    oriented[:, [1, 2]] = oriented[:, [2, 1]]
    return oriented

def write_out_off_file(filename,indices,points):
    indices = orient_faces_to_paper_winding(indices, points)
    file = open(filename,'w')
    file.write('OFF\n')
    file.write(f'{int(len(points))} {int(len(indices))} 0\n')
    for p in points:
        file.write(f'{p[0]} {p[1]} {p[2]}\n')
    for i in indices:
        file.write(f'3 {int(i[0])} {int(i[1])} {int(i[2])}\n')
    file.close()

def main():
    points = read_test_off_file('test.off')
    indices = read_out_off_file('out_tmp.off')
    write_out_off_file('out.off',indices,points)

if __name__ == "__main__":
    main()
