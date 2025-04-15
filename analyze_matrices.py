import numpy as np

import matplotlib.pyplot as plt 

from util import exec_als, extend2n, info_rank1, plot_spectrum, print_data, print_matrices, read_data, read_matrices, read_raw, ttsvd
from als import cp_restore, normalize_matrices1

def read_matrices_files(shape, start, end, prefix):
    ret = []
    for k in range(start, end+1):
        filename = prefix + str(k) + ".txt"
        matrices = read_matrices(shape, filename)
        ret.append((matrices, k))
    return ret


if __name__ == "__main__":
    N = 16384 * 64
    D = int(np.log2(N))
    shape = tuple([2] * D)
    all_matrices = read_matrices_files(shape, 2, 10, "als_matrices_")
    #for matrices, rank in all_matrices:
    #    norms = np.abs(normalize_matrices1(rank, matrices)[1])
    #    plt.plot(norms)
    #    plt.show()
    plot_spectrum(cp_restore(shape, all_matrices[-1][0], all_matrices[-1][1]).ravel(), abs=True)
    plt.show()