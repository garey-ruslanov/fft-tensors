import numpy as np
import matplotlib.pyplot as plt


def norm1(x):
    return np.sum(np.abs(x))


def norm2(x):
    return np.linalg.norm(x)


def count(inds : np.array, x : np.array, mod=False):
    if mod:
        y = (x + np.pi) % (2*np.pi) - np.pi
    else:
        y = x
    return norm2(inds - y)


def try_fit(inds, mod=False):
    n = len(inds)
    if mod:
        phi = inds[-1]
    else:
        phi = (inds[-1] + inds[0]/2**(len(inds)-1)) / 2
    x = np.asarray([(phi * 2**k) for k in range(n)][::-1])
    c = count(inds, x, mod)
    step = 0.76
    while step > 1e-10:
        x1 = np.asarray([(phi - step) * 2**k for k in range(n)][::-1])
        x2 = np.asarray([(phi + step) * 2**k for k in range(n)][::-1])
        c1 = count(inds, x1, mod)
        c2 = count(inds, x2, mod)

        if c1 >= c >= c2:
            c = c2
            phi = phi + step
        elif c2 >= c >= c1:
            c = c1
            phi = phi - step
        elif c >= c2 >= c1:
            c = c1
            phi = phi - step
        elif c >= c1 >= c2:
            c = c2
            phi = phi + step
        else:
            pass
        step /= 2.0
    return phi
    # вроде работает


def detect_bullshit(inds):
    phi_r = try_fit(np.real(inds), mod=False)
    phi_i = try_fit(np.imag(inds), mod=True)

    inds_new = [(phi_r * 2**k + 1j * ((phi_i * 2**k + np.pi) % (2*np.pi) - np.pi)) for k in range(len(inds))][::-1]
    #print(norm2(np.asarray(inds) - np.asarray(inds_new)))

    return inds_new