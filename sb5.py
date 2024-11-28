import numpy as np
import matplotlib.pyplot as plt

import subprocess

from util import *

# data: topspin generated 13C

filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/fid'
data = extend2n(read_raw(filename))
n = data.size       ; N = n
d = int(np.log2(n)) ; D = d
shape = [2] * d
assert np.prod(shape) == n

print_data(data, True, 'out1.txt')
print_data(np.asarray(shape), False, 'out2.txt')

r = 3               ; R = r
ea = exec_als(R=R)

data_a = read_data('out_als.txt')
matrices = read_matrices(shape)


plot_spectrum(data, abs=True)
plot_spectrum(data_a, abs=True)
plt.show()

for k in range(R):
    vectors = []
    for i in range(d):
        vectors.append(np.copy(matrices[i][:,k]))
    plot_rank1(vectors, R)
