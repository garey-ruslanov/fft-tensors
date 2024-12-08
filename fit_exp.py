import numpy as np
import matplotlib.pyplot as plt


def count(inds : np.array, x : np.array, mod=False):
    if mod:
        y = (x + np.pi) % (2*np.pi) - np.pi
    else:
        y = x
    return np.linalg.norm(inds - y)


def try_fit(inds, mod=False):
    n = len(inds)
    phi = inds[-1]
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
