import numpy as np
import matplotlib.pyplot as plt


def create_signal(params, n_sig=1):
    N = params['N']
    pivots = params['pivots']

    sig = np.zeros(N, dtype=complex)
    for p in pivots:
        sig += np.asarray([np.exp(k*p) for k in range(N)])
    return sig


def noise_uniform(sig : np.ndarray, a):
    no = np.zeros((sig.size,))
    for k in range(no.size):
        no[k] = np.random.uniform(-1.0, 1.0)
    sig += no.reshape(sig.shape) * a


if __name__ == '__main__':
    sig = create_signal({'N':2048, 'pivots':[-0.024+1.5j]})
    noise_uniform(sig, 0.05)

    plt.plot(np.abs(sig))
    plt.show()
    plt.plot(np.abs(np.fft.fftshift(np.fft.fft(sig))))
    plt.show()
