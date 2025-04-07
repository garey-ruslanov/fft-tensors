import numpy as np
import matplotlib.pyplot as plt

import subprocess

from fit_exp import try_fit, detect_bullshit
from reconstruct import reconstruct_from_pivots

def read_raw(filename='out.txt', b_endian=False):
    dt = ">i4" if b_endian else "<i4"
    with open(filename) as f:
        arr = np.fromfile(f, dtype=dt, count=-1, )
        arr = np.asarray([arr[2*i] + arr[2*i+1] * 1.j for i in range(len(arr) // 2)])

        return arr


def read_data(filename):
    with open(filename, 'r') as f:
        data = np.asarray([float(s.split()[0]) + 1j * float(s.split()[1]) for s in f.readlines()[1:]])

        return data


def read_matrices(tshape, filename):
    D = len(tshape)
    with open(filename, 'r') as f:
        R = int(f.readline())
        matrices = [np.zeros((tshape[i], R), dtype=complex) for i in range(D)]
        for i in range(D):
            for c in range(R):
                for r in range(tshape[i]):
                    l = f.readline()
                    while l.isspace():
                        l = f.readline()
                    
                    v = float(l.split()[0]) + 1j * float(l.split()[1])
                    matrices[i][r][c] = v
    return matrices


def print_data(data : np.ndarray, filename, complex=True):
    
    with open(filename, 'w') as f:
        f.write(str(data.size))
        f.write('\n')
        for i in range(data.size):
            if complex:
                f.write(str(np.real(data[i])) + ' ' + str(np.imag(data[i])))
            else:
                f.write(str(data[i]))
            f.write('\n')


def print_matrices(matrices : list, tshape, filename, complex=True):  # &
    D = len(tshape)
    R = matrices[0].shape[1]

    with open(filename, 'w') as f:
        f.write(str(R))
        f.write('\n')
        for i in range(len(matrices)):
            for c in range(R):
                for r in range(tshape[i]):
                    if complex:
                        f.write(str(np.real(matrices[i][r][c])) + " " + str(np.imag(matrices[i][r][c])) + "\n")
                    else:
                        pass
            f.write("\n")


def plot_signal(a : np.ndarray, abs=True, name=''):
    if abs:
        plt.title('abs ' + name)
        plt.plot(np.abs(a.ravel()))
    else:
        plt.title('logarithmic ' + name)
        plt.plot(np.real(np.log(a.ravel())))
        plt.plot(np.imag(np.log(a.ravel())))


def plot_spectrum(s : np.ndarray, abs=False, name=''):
    plt.title('spectrum ' + name)
    if abs:
        plt.plot(np.abs(np.fft.fftshift(np.fft.fft(s))))
    else:
        plt.plot(np.real(np.fft.fftshift(np.fft.fft(s))))
        plt.plot(np.imag(np.fft.fftshift(np.fft.fft(s))))


def plot_norm(a : np.ndarray):
    plt.plot(np.abs(a.ravel()) / np.max(np.abs(a.ravel())))
    plt.show()


def extend2n(array : np.ndarray):
    assert len(array.shape) == 1
    N = array.size
    N2 = 1
    while N2 < N:
        N2 *= 2
    array = np.concatenate((array, np.zeros((N2 - N,), dtype=array.dtype)))
    
    N = N2
    while np.linalg.norm(array[N // 2:]) == 0:
        array = array[:N // 2]
        N = N // 2
    return array


def ttsvd(d : int, tensor : np.ndarray, rg : list, EPS_cutout=1e-6):
    shape = tensor.shape
    
    if rg is None:
        rg = [1]
    gset = list()
    t_v = np.copy(tensor)

    #fig, axs = plt.subplots(3, 4)
    #plt.subplots_adjust(hspace=0.4)
    #plt.subplots_adjust(wspace=0.25)
    
    for j in range(1, d):
        st = 1
        for t in range(j, d):
            st *= shape[t]
    
        svd_res = np.linalg.svd(t_v.reshape((rg[j - 1] * shape[j - 1], st)), compute_uv=True, full_matrices=False)

        rj = len(svd_res[1])
        # replace len(svd_res[1])
        for k in range(1, rj):
            if (svd_res[1][k] / svd_res[1][0]) < EPS_cutout:
                rj = k
                break
        if j < len(rg):
            rj = rg[j]

        u = np.delete(svd_res[0], slice(rj, len(svd_res[1]), 1), 1)
        v = np.delete(svd_res[2], slice(rj, len(svd_res[1]), 1), 0)

        if j >= len(rg):
            rg.append(rj)

        #if j <= 12:
        #    axs[(j-1) // 4, (j-1) % 4].scatter(np.arange(rj), svd_res[1][:rj] / svd_res[1][0])
        #    axs[(j-1) // 4, (j-1) % 4].set_yscale('log')
        #    axs[(j-1) // 4, (j-1) % 4].set_title(str(j) + '-th fold')

        gj = u.reshape((rg[j - 1], shape[j - 1], rj))
        gset.append(gj)

        t_v = (np.diag(svd_res[1][:rj]) @ v).reshape(((rj,) + shape[j:]))

    print(rg)

    #plt.show()

    gset[0] = gset[0].reshape(gset[0].shape[1:])
    t_v = t_v.reshape(t_v.shape + (1,))
    gset.append(t_v)

    return gset


def ttsvd_restore(d, gset):
    letters = 'abcdefghijklmnopqrstuvwxy'
    gg = gset[0]
    for it in range(1,d):
        gg = np.einsum(letters[:it]+'z,z'+letters[it:it+2] + '->' + letters[:it+2], gg, gset[it], dtype=complex)
    gg = gg.reshape(gg.shape[:-1])
    return gg


from als import als_iteration_1, normalize_matrices, cp_restore, random_matrices


def exec_als(R=1, use_random_matrices=True, filename_in="als_start_matrices.txt", filename_out="out_als_matrices.txt"):
    data = read_data("out1.txt")

    # reading shape
    with open("out2.txt", "r") as f:
        dshape = [int(s) for s in f.readlines()]
        d, shape = dshape[0], dshape[1:]
    # done reading shape

    T = data.reshape(shape)
    if use_random_matrices:
        matrices = random_matrices(shape, R)
    else:
        try:
            matrices = read_matrices(shape, filename=filename_in)
        except:
            print("failed reading matrices.")
            return
    matrices, norms = normalize_matrices(R, matrices)
    epsilon_rel = 1e-6
    i = 0
    n1 = np.linalg.norm(T - cp_restore(shape, matrices, R, norms))
    n2 = 1.0
    normT = np.linalg.norm(T)
    while np.abs(n2 - n1) / normT > epsilon_rel:  # ?????????
        matrices, norms, n1, n2 = als_iteration_1(T, shape, matrices, norms, R)
        print(np.abs(n2 - n1) / normT, ', res:', n2 / normT)
        i += 1

    print("total iterations:", i)
    print("end relative residual:", n2 / normT)
    matrices[0] = matrices[0] @ np.diag(norms)
    print_matrices(matrices, shape, filename=filename_out)
    print_data(cp_restore(shape, matrices, R).flatten(), complex=True, filename="out_als.txt")

    return i, n2 / normT


def _exec_als(R=1):
    output = subprocess.run(['./als_bin1', str(R)], capture_output=False)
    #print(output.stdout.decode())


def gen_data(N : int, R : int, M : float):
    sig = np.zeros((N,), dtype=complex)
    for i in range(R):
        phi = np.random.randint(0, N) * 2 * np.pi
        sig += np.exp(np.arange(N) * phi * 1j) * M / (i+2)
    return sig


def plot_rank1(vectors : list, R : int):
    sig = np.ones((1))
    for v in vectors:
        sig = np.outer(sig, v)
    
    con = 1.0
    inds = []
    for v in vectors:
        con *= v[0]
        p = v[1] / v[0]

        inds.append(np.log(p))
    
    inds_r = np.real(inds)
    inds_i = np.imag(inds)

    phi_r = try_fit(inds_r, mod=False)
    phi_i = try_fit(inds_i, mod=True)
    
    plt.plot(inds_r)
    plt.plot([(phi_r * 2**k) for k in range(len(inds_r))][::-1])
    plt.show()
    plt.plot(inds_i)
    plt.plot([(phi_i * 2**k + np.pi) % (2*np.pi) - np.pi for k in range(len(inds_i))][::-1])
    plt.show()

    inds_new = [(phi_r * 2**k + 1j * phi_i * 2**k) for k in range(len(inds_r))][::-1]
    asdf = reconstruct_from_pivots(inds_new) * con
    plot_spectrum(sig.ravel(), abs=True)
    plot_spectrum(asdf, abs=True)
    plt.show()

    # !
    return asdf


def info_rank1(vectors : list):
    signal = np.ones((1,))
    for v in vectors:
        signal = np.outer(signal, v)
    
    indexes = []
    cst = 1.0
    for v in vectors:
        indexes.append(np.log(v[1] / v[0]))
        cst *= v[0]
    
    nrm = np.linalg.norm(signal)
    return signal.ravel(), indexes, nrm


def noise_uniform(sig : np.ndarray, a):
    no = np.zeros((sig.size,))
    for k in range(no.size):
        no[k] = np.random.uniform(-1.0, 1.0)
    sig += no.reshape(sig.shape) * a


def starting_matrices(tensor : np.ndarray, rank : int):
    tshape = tensor.shape
    d = len(tshape)
    gset = ttsvd()
    pass