
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

