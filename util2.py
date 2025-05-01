
import numpy as np

# ???
def fit_exp_whole_signal(data : np.array):
    abs_data = np.abs(data)
    N = data.size
    #plot.plot(np.log(abs_data))

    w = 24
    sm_data = np.asarray(list(filter(lambda x: x != 0.0, [np.max(abs_data[i:i+w]) for i in range(N - w)])))
    print(np.min(sm_data))
    #plot.plot(np.log(sm_data))
    #plot.show()

    p = np.polyfit(np.arange(sm_data.size), np.log(sm_data), 1)
    print(p)
    pp = np.polyval(p, np.arange(sm_data.size))
    #plot.plot(np.abs(data))
    #plot.plot(np.exp(pp))
    #plot.show()
    return p[0]

# gives a signal and its tensor-CP-matrices
# signal = sum of np.exp([0, p, 2p, 3p, ...]) for all p in pivots
# matrices = cp appr of signal.reshape(2^D)
# norms = coefficients
def signal_from_pivots(D, pivots, coefficients=None):
    R = len(pivots)
    #assert len(coefficients) == R == len(pivots)
    N = 2**D
    signal = np.zeros((N,), dtype=complex)
    matrices = [np.zeros((2, R), dtype=complex) for i in range(D)]
    k = 0
    for p in pivots:
        if np.real(p) > 0:
            print("real part of a pivot should be negative")
            continue
        signal += np.exp(np.arange(N) * p)
        
        for i in range(D):
            matrices[i][0, k] = 1.0
            matrices[i][1, k] = np.exp(p * 2**(D - i - 1))
        k += 1
    return signal, matrices, coefficients


def matrices_add_random(matrices, D_f, D_im, pivot):
    if matrices is None or len(matrices) == 0:
        new_matrices = [np.asarray([[1.0], [1.0]], dtype=complex) for i in range(D_im)] + [np.asarray([[1.0], [np.exp(pivot * 2**(D_f - i - 1))]]) for i in range(D_f)]
        return new_matrices

    R = matrices[0].shape[1]
    new_matrices = [np.zeros((matrices[i].shape[0], R+1), dtype=complex) for i in range(D_f + D_im)]
    for i in range(D_f + D_im):
        new_matrices[i][:,:R] = matrices[i][:,:]
        new_matrices[i][0,R] = 1.0
        new_matrices[i][1,R] = (np.random.randn() + np.random.randn()*1j) / 2**(0.5) * np.abs(pivot)
    return new_matrices


def matrices_add2(matrices, D_f, D_im, pivot1, pivot2):
    if matrices is None or len(matrices) == 0:
        new_matrices = [np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=complex) for i in range(D_im)] + [np.asarray([[1.0, 1.0], [np.exp(pivot1 * 2**(D_f - i - 1)), np.exp(pivot2 * 2**(D_f - i - 1))]]) for i in range(D_f)]
        return new_matrices
 
    R = matrices[0].shape[1]
    new_matrices = [np.zeros((matrices[i].shape[0], R+2), dtype=complex) for i in range(D_f + D_im)]
    for i in range(D_im):
        new_matrices[i][:,:R] = matrices[i][:,:]  # ones
        new_matrices[i][0,R] = 1.0
        new_matrices[i][1,R] = 1.0
    for i in range(D_im, D_f + D_im):
        new_matrices[i][:,:R] = matrices[i][:,:]
        new_matrices[i][0,R] = 1.0
        new_matrices[i][1,R] = np.exp(pivot1 * 2**(D_f - i - 1))
        new_matrices[i][0,R+1] = 1.0
        new_matrices[i][1,R+1] = np.exp(pivot2 * 2**(D_f - i - 1))
    return new_matrices
