import numpy as np
import matplotlib.pyplot as plt

import subprocess

from util import *

# data: real 13C
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real13C/fid1'

# data: real 1H
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real1H/fid1'

def experiment(data, rank, draw_info=True):


    n = data.size       ; N = n
    d = int(np.log2(n)) ; D = d
    shape = [2] * d
    assert np.prod(shape) == n

    print_data(data, True, 'out1.txt')
    print_data(np.asarray(shape), False, 'out2.txt')

    r = rank            ; R = r
    skip = False
    if not skip:
        ea = exec_als(R=R)
    else:
        print('als skipped')

    data_a = read_data('out_als.txt')  # unused
    matrices = read_matrices(shape)

    spec_lim = np.max(np.abs(np.fft.fft(data)))
    # plot_spectrum(data, abs=True)
    # plot_spectrum(data_a, abs=True)
    # plt.show()
    # пофиг на это

    items_to_draw = []
    component_sum = np.zeros_like(data)

    for k in range(R):
        vectors = []
        for i in range(d):
            vectors.append(np.copy(matrices[i][:,k]))

        sig, inds, nrm = info_rank1(vectors)
        inds2 = detect_bullshit(inds)

        items_to_draw.append((k, nrm, sig, inds, inds2))
        component_sum += sig

    plot_spectrum(data, abs=True)
    plot_spectrum(component_sum, abs=True)
    plt.show()

    items_to_draw.sort(key=lambda x: x[1], reverse=True)  # sorting by norm

    g_per_row = 4
    fig, axes = plt.subplots((2*R + g_per_row - 1) // g_per_row, g_per_row)
    fig.suptitle('Rank 1 components given by ALS, rank=%i' % R)
    for k in range(R):
        k, nrm, sig, inds, inds2 = items_to_draw[k]

        print(k, nrm)

        axes[(2*k) // g_per_row, (2*k) % g_per_row].set_ylim((0, spec_lim))
        axes[(2*k) // g_per_row, (2*k) % g_per_row].plot(np.abs(np.fft.fftshift(np.fft.fft(sig))))

        axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.real(inds))
        axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.real(inds2))
        axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.imag(inds))
        axes[(2*k+1) // g_per_row, (2*k+1) % g_per_row].plot(np.imag(inds2))
    
    plt.show()


# data: topspin generated 13C
filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/naphtalene_13C'
data = extend2n(read_raw(filename))

experiment(data, 5)

# data: generated, uniformly noised
#filename = same
data = extend2n(read_raw(filename))
snr = 0.1
noise_uniform(data, np.max(np.abs(data)) * snr)

experiment(data, 5)

# another signal

filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/ile_1H'
data = extend2n(read_raw(filename))

experiment(data, 10)

#filename = same
data = extend2n(read_raw(filename))
snr = 0.1
noise_uniform(data, np.max(np.abs(data)) * snr)

experiment(data, 10)
