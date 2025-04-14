
import numpy as np
import matplotlib.pyplot as plt


def random_signal(n, n_peaks, max_ampl, exponent):
    d = int(np.log2(n));    D = d
    assert n == 2**d
    r = n_peaks;            R = r
    
    from als import cp_restore

    freqs = [np.random.rand() * n for _ in range(r)]
    norms = [np.random.rand() * max_ampl for _ in range(r)]
    
    matrices = [np.zeros((2,r), dtype=complex) for _ in range(d)]
    for i in range(d):
        for k in range(r):
            matrices[i][0,k] = 1.0;
            matrices[i][1,k] = np.exp((exponent + freqs[k] * 1j) * 2**(d - i - 1))
    tensor = cp_restore([2] * d, matrices, R, norms)
    return tensor.flatten(), matrices


def normal_noise(n, c):
    return (np.random.randn(n) + np.random.randn(n) * 1j) * c * 2**(-0.5)


if __name__ == '__main__':
    #sig = create_signal({'N':2048, 'pivots':[-0.024+1.5j]})
    #noise_uniform(sig, 0.05)

    #plt.plot(np.abs(sig))
    #plt.show()
    #plt.plot(np.abs(np.fft.fftshift(np.fft.fft(sig))))
    #plt.show()
    signal = random_signal(32768, 25, 5e5, 5e-5)
    from util import plot_spectrum
    plot_spectrum(signal, abs=True)
    plt.show()

