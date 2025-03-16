
import numpy as np
import scipy.linalg as sclin

def least_squares(A: np.ndarray, F: np.ndarray):  # ||AX - F|| -> min
    return np.linalg.pinv(A) @ F


def cp_restore(dims, matrices, rank, norms=None):
    if norms is None:
        norms = np.ones((rank,), dtype=complex)
    T = np.zeros(dims, dtype=complex)
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    for j in range(rank):
        tj = matrices[0][:,j]
        for k in range(1, len(matrices)):
            tj = np.einsum(alpha[:k] + ',' + alpha[k] + '->' + alpha[:k+1], tj, matrices[k][:,j])
        T += tj * norms[j]
    return T


def als_iteration_simple(T: np.ndarray, dims, matrices: list, rank):
    d = len(dims)
    n = np.prod(dims)
    ind = tuple(np.arange(d))
    for k in range(d):
        tk = T.transpose(((k,) + ind[:k] + ind[k+1:])).reshape((dims[k], n // dims[k]))
        #print(tk)  # yes
        matrix_kh_r = np.ones((1, rank))
        for i in range(d):
            if i == k:
                continue
            matrix_kh_r = sclin.khatri_rao(matrix_kh_r, matrices[i])
        #print(matrix_kh_r.shape)
        norm_e = np.linalg.norm(tk - matrices[k] @ matrix_kh_r.T)
        # ||tk - A*mat_khr^T|| -> min
        # ||tk^T - mat_khr*A^T|| -> min
        # A^T = mat_khr^+*tk^T
        # A = tk*mat_khr^+^T

        mat_new = (np.linalg.pinv(matrix_kh_r) @ tk.T).T
        matrices[k] = mat_new
        norm_e_1 = np.linalg.norm(tk - mat_new @ matrix_kh_r.T)
        print(norm_e, norm_e_1)
    
    return matrices


def als_iteration_1(T: np.ndarray, dims, matrices: list, norms, rank):
    d = len(dims)
    n = np.prod(dims)
    ind = tuple(np.arange(d))
    norm_s = np.linalg.norm(T - cp_restore(dims, matrices, rank, norms))
    for k in range(d):
        tk = T.transpose(((k,) + ind[:k] + ind[k+1:])).reshape((dims[k], n // dims[k]))

        matrix_kh_r = np.ones((1, rank), dtype=complex)
        for i in range(d):
            if i == k:
                continue
            matrix_kh_r = sclin.khatri_rao(matrix_kh_r, matrices[i])
        #print(matrix_kh_r.shape)
        #norm_e = np.linalg.norm(tk - matrices[k] @ np.diag(norms) @ matrix_kh_r.T)
        # ||tk - A*N*mat_khr^T|| -> min
        # ||tk^T - mat_khr*N*A^T|| -> min
        # A^T = mat_khr^+*tk^T
        # A = tk*mat_khr^+^T

        mat_new = (np.linalg.pinv(matrix_kh_r) @ tk.T).T
        matrices[k] = mat_new
        matrices, norms = normalize_matrices(rank, matrices)

        #norm_e_1 = np.linalg.norm(tk - mat_new @ np.diag(norms) @ matrix_kh_r.T)
        #print(norm_e, norm_e_1)
    norm_e = np.linalg.norm(T - cp_restore(dims, matrices, rank, norms))
    return matrices, norms, norm_s, norm_e


def random_matrices(dims, rank):
    d = len(dims)
    matrices = [np.random.random((dims[k], rank)) for k in range(d)]
    return matrices


def normalize_matrices(rank, matrices):
    norms = np.ones((rank,))
    for k in range(len(matrices)):
        for j in range(rank):
            n = np.linalg.norm(matrices[k][:,j])
            norms[j] *= n
            matrices[k][:,j] = matrices[k][:,j] / n
    
    return matrices, norms


if __name__ == "__main__":
    dims = (3, 4, 5, 2)
    N = np.prod(dims)
    T = np.arange(N).reshape(dims)

    rank = 4
    matrices = random_matrices(dims, rank)
    matrices, norms = normalize_matrices(rank, matrices)

    for i in range(1000):
        matrices, norms, n1, n2 = als_iteration_1(T, dims, matrices, norms, rank)
        print("after " + str(i) + " iterations:\n" + str(abs(n2 - n1)))
