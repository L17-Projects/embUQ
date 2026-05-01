#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

frequencies = np.loadtxt('output/eigvalues.txt')

frequencies = frequencies[::-1]

n0 = 0
nmodes = 40

plt.plot(np.sort(np.sqrt(1/frequencies[n0:nmodes])), 'ob-')

plt.savefig(f'freq.png')
