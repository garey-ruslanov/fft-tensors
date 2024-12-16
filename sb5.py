import numpy as np
import matplotlib.pyplot as plt

import subprocess

from util import *

# data: topspin generated 13C
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/ile_13C'

# data: real 13C
filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real13C/fid1'

# data: real 1H
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real1H/fid1'

data = extend2n(read_raw(filename))
n = data.size       ; N = n
d = int(np.log2(n)) ; D = d
shape = [2] * d
assert np.prod(shape) == n

print_data(data, True, 'out1.txt')
print_data(np.asarray(shape), False, 'out2.txt')

r = 6              ; R = r
skip = False
if not skip:
    ea = exec_als(R=R)
else:
    print('als skipped')

data_a = read_data('out_als.txt')
matrices = read_matrices(shape)

spec_lim = np.max(np.abs(np.fft.fft(data)))
plot_spectrum(data, abs=True)
plot_spectrum(data_a, abs=True)
plt.show()

g_per_row = 6
fig, axes = plt.subplots((2*R + g_per_row - 1) // g_per_row, g_per_row)
fig.suptitle('Rank 1 components given by ALS, rank=%i' % R)
for k in range(R):
    vectors = []
    for i in range(d):
        vectors.append(np.copy(matrices[i][:,k]))
    
    sig, inds, cst = info_rank1(vectors)
    print(k, cst, end=' ')
    inds2 = detect_bullshit(inds)

    axes[(2*k) // g_per_row, (2*k) % g_per_row].set_ylim((0, spec_lim))
    axes[(2*k) // g_per_row, (2*k) % g_per_row].plot(np.abs(np.fft.fftshift(np.fft.fft(sig))))

    axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.real(inds))
    axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.real(inds2))
    axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.imag(inds))
    axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.imag(inds2))



plt.show()
