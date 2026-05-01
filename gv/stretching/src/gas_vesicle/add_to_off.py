import numpy as np
from parameters import *

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

def write_out_off_file(filename,indices,points):
    file = open(filename,'w')
    file.write('OFF\n')
    file.write(f'{int(len(points))} {int(len(indices))} 0\n')
    for p in points:
        file.write(f'{p[0]} {p[1]} {p[2]}\n')
    for i in indices:
        file.write(f'3 {int(i[0])} {int(i[1])} {int(i[2])}\n')
    file.close()

points = read_test_off_file('test.off')
indices = read_out_off_file('out_tmp.off')
write_out_off_file('out.off',indices,points)
