import numpy as np
import matplotlib.pyplot as plt

import subprocess

from util import *

# data: real 13C
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real13C/fid1'

# data: real 1H
# filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real1H/fid1'

def experiment(data, rank, draw_info=True, skip=False):


    n = data.size       ; N = n
    d = int(np.log2(n)) ; D = d
    shape = [2] * d
    assert np.prod(shape) == n

    print_data(data, filename='out1.txt', complex=True)
    print_data(np.asarray(shape), filename='out2.txt', complex=False)

    r = rank            ; R = r

    if not skip:
        ea = exec_als(R=R)
    else:
        print('als skipped')

    data_a = read_data('out_als.txt')  # unused
    matrices = read_matrices(shape, filename="out_als_matrices.txt")

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

    g_per_row = 6
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


def experiment_2(data : np.ndarray, sR : int, R : int, draw_info=True, prev_matrices=None):
    n = data.size       ; N = n
    d = int(np.log2(n)) ; D = d
    shape = [2] * d
    assert np.prod(shape) == n

    data_c = np.copy(data)
    if prev_matrices is not None:
        data_c = data_c - cp_restore(shape, prev_matrices, sR-1).flatten()
    iterations = []
    residual_norms = []
    for r in range(sR, R+1):
        print("r =", r)
        g1 = ttsvd(d, data_c.reshape(shape), [1]*d, -1)
        cp1 = [np.copy(g1[i].reshape((shape[i], 1))) for i in range(d)]

        if draw_info:
            plot_spectrum(data, abs=True)
            plot_spectrum(cp_restore(shape, prev_matrices, r-1).flatten(), abs=True)
            plot_spectrum(cp_restore(shape, cp1, 1).flatten(), abs=True)
            plt.show()

        matrices_r = [np.zeros((shape[i], r), dtype=complex) for i in range(d)]
        if prev_matrices is not None:
            for i in range(d):
                matrices_r[i][:,:r-1] = prev_matrices[i][:,:]
        for i in range(d):
            matrices_r[i][:,r-1] = cp1[i][:,0]

        print_data(data, complex=True, filename="out1.txt")
        print_data(np.asarray(shape), complex=False, filename="out2.txt")
        print_matrices(matrices_r, shape, complex=True, filename="als_start_matrices.txt")

        iter, res_norm = exec_als(R=r, use_random_matrices=False)
        iterations.append(iter)
        residual_norms.append(res_norm)

        t_a = read_data(filename="out_als.txt")
        matrices_ls = read_matrices(shape, filename="out_als_matrices.txt")

        data_c = data - t_a
        prev_matrices = matrices_ls
        if draw_info:
            plot_spectrum(data, abs=True)
            plot_spectrum(t_a, abs=True)
            plt.show()
    print(iterations)
    print(residual_norms)

    plot_spectrum(data, abs=True)
    plot_spectrum(cp_restore(shape, prev_matrices, R).flatten(), abs=True)
    plt.show()

"""
# data: topspin generated 13C
filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/naphtalene_13C'
data = extend2n(read_raw(filename))

experiment(data, 10)

# data: generated, uniformly noised
#filename = same
data = extend2n(read_raw(filename))
snr = 0.1
noise_uniform(data, np.max(np.abs(data)) * snr)

experiment(data, 10)

# another signal

filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/ile_1H'
data = extend2n(read_raw(filename))

experiment(data, 10)

#filename = same
data = extend2n(read_raw(filename))
snr = 0.1
noise_uniform(data, np.max(np.abs(data)) * snr)

experiment(data, 10)
"""

filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/generated/ile_1H'
data = extend2n(read_raw(filename))

#experiment(data, 15)
experiment_2(data, 25, 30, draw_info=True, prev_matrices=read_matrices([2] * (int(np.log2(data.size))), filename="out_als_matrices 24.txt"))
