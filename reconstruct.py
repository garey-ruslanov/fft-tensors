import numpy as np
import matplotlib.pyplot as plt


def reconstruct_from_pivots(inds):
    d = len(inds)
    tensor = np.ones((1,))
    
    for k in range(d):
        vec = [1, np.exp(inds[k])]
        tensor = np.outer(tensor, vec)
    
    return tensor.ravel()
