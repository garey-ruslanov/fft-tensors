import numpy as np
import matplotlib.pyplot as plt

import subprocess

from als import cp_restore
from fit_exp import detect_bullshit
from util import exec_als, extend2n, info_rank1, plot_spectrum, print_data, print_matrices, read_data, read_matrices, read_raw, ttsvd
from util2 import fit_exp_whole_signal, matrices_add1, signal_from_pivots

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


def experiment_2(data : np.ndarray, sR : int, R : int, draw_info=True, flag2=False, prev_matrices=None):
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
        
        if flag2:
            EPS = 0.1
            nrms = []
            for k in range(r):
                nrms.append(np.linalg.norm(cp_restore(shape, [mat[:,k:k+1] for mat in prev_matrices], 1)))
            nrms = sorted(nrms, reverse=True)
            print(nrms / nrms[0])
            if (nrms[-1] / nrms[0] < EPS):
                print("cringe components found,", len(list(filter(lambda x: x / nrms[0] < EPS, nrms))))

    print(iterations)
    print(residual_norms)

    plot_spectrum(data, abs=True)
    plot_spectrum(cp_restore(shape, prev_matrices, R).flatten(), abs=True)
    plt.show()

    return prev_matrices


def experiment_full(data : np.ndarray, rank : int, mode : str, D_im=0, draw_info=True, save_matrices_every=4):
    n = data.size       ; N = n
    d = int(np.log2(n)) ; D = d
    R = rank
    shape = tuple([2] * (D + D_im))
    assert np.prod(shape) == n

    start_rank = 0  # ?

    n_images = 2**D_im

    # returns indices of at most 2 elements that differ from max by 0.08 or less
    def max_2(spectrum):
        k = 2
        arr = np.abs(spectrum)
        n = len(arr)
        m = np.max(arr)
        arr = arr / m
        e = 0.08  # ???
        l = []
        for i in range(n):
            v = arr[i]
            if abs(m - v) < e:
                l.append((v, i))
        l = sorted(l, key=lambda x: x[0], reverse=True)
        if len(l) > k:
            l = l[:k]
        return tuple([ll[1] for ll in l])

    matrices = []
    if mode == "random pivots":
        exponent = fit_exp_whole_signal(data)
        signal, matrices, _ = signal_from_pivots(D, [exponent + 1j * np.random.rand() * n for _ in range(start_rank)], None)

        max_freq_data = np.max(np.abs(np.fft.fft(data)))
        max_freq_rand = np.max(np.abs(np.fft.fft(signal)))
        norms = [max_freq_data / max_freq_rand] * start_rank
        matrices[0] = matrices[0] @ np.diag(norms)

    if mode == "ttsvd":
        g1 = ttsvd(d, data.reshape(shape), [1]*d, -1)
        matrices = [np.copy(g1[i].reshape((shape[i], 1))) for i in range(d)]

    if mode == "random":
        from als import random_matrices
        matrices = random_matrices(shape, start_rank)
    
    iterations = []
    residual_norms = []

    for r in range(start_rank+1, R+1):
        print("r =", r)

        if mode == "ttsvd":
            g1 = ttsvd(d, data.reshape(shape), [1]*d, -1)
            cp1 = [np.copy(g1[i].reshape((shape[i], 1))) for i in range(d)]
            if len(matrices) == 0:
                pass
            # don't care what happens here, not going to execute it
            # поебать че тут происходит, все равно не буду это запускать
            pass

        if mode == "random pivots":
            pivo = exponent + 1j * np.random.rand() * n 
            matrices = matrices_add1(matrices, D - D_im, D_im, pivo)

        from util import filenames_dict

        print_data(data, complex=True, filename=filenames_dict["data"])
        print_data(np.asarray(shape), complex=False, filename=filenames_dict["shape"])
        print_matrices(matrices, shape, complex=True, filename="als_start_matrices.txt")
        
        iter, res_norm = exec_als(R=r, use_random_matrices=False, filename_in="als_start_matrices.txt", filename_out="out_als_matrices.txt")
        iterations.append(iter)
        residual_norms.append(res_norm)

        t_a = read_data(filename=filenames_dict["result"])
        matrices_ls = read_matrices(shape, filename="out_als_matrices.txt")

        matrices = matrices_ls

        if draw_info:
            plot_spectrum(data, abs=True)
            plot_spectrum(t_a, abs=True)
            plt.show()

    print(iterations)
    print(residual_norms)

    plot_spectrum(data, abs=True)
    plot_spectrum(cp_restore(shape, matrices, R).flatten(), abs=True)
    plt.show()

    return matrices

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
#filename = '/mnt/c/Users/Ruslan Gareev/Desktop/rehcfx/raw fids/real13C/fid1'
data = extend2n(read_raw(filename))

#experiment(data, 15)
#experiment_2(data, 51, 51, draw_info=False, prev_matrices=read_matrices([2] * (int(np.log2(data.size))), filename="out_als_matrices 50.txt"))

#data = np.concatenate((data, data))
#data = np.concatenate((data, data))
#data = np.concatenate((data, data))

D = int(np.log2(data.size))
shape = tuple([2] * D)

R = 10
#als_matrices = \
#experiment_2(data, 1, R, draw_info=False, prev_matrices=None, flag2=True)

#als_matrices, norms = normalize_matrices2(R, als_matrices)
#print(norms)
#als_matrices = \
#read_matrices(shape, "out_als_matrices.txt")
als_matrices = \
experiment_full(data=data, rank=R, mode="random pivots", draw_info=False)

#als_matrices, norms = normalize_matrices2(R, als_matrices)
#print(norms)


comps = []
for i in range(R):
    mat1 = [np.zeros((2,1), dtype=complex) for _ in range(D)]
    for k in range(D):
        mat1[k][0,0] = als_matrices[k][0,i]
        mat1[k][1,0] = als_matrices[k][1,i]
    comps.append(mat1)


#dims = tuple([m.shape[0] for m in als_matrices])

#rank1_sig = [cp_restore(dims, [m[:,r].reshape((2, 1)) for m in als_matrices], 1).flatten() for r in range(R)]  # >:
# true
#asdf = []
#for si in rank1_sig:
#    s = np.abs(si)
#    plot_signal(s, nolog=False)
#    asdf.append(np.polyfit(np.linspace(0, len(s) - 1, len(s)), s, 1)[0])
#    print(asdf[-1])
#plt.show()
#plt.plot(asdf)
#plt.show()

# E(R) = 2^14/15 ~= 1000
# ||n||_F / 30
