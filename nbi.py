import numpy as np
from scipy.linalg import inv

def NBI(F0):
    M, N = F0.shape
    R = np.diag(np.sum(F0, axis=1))
    H = np.diag(np.sum(F0, axis=0))
    W = (F0 @ inv(H)).T @ (inv(R) @ F0)
    return W

def NBI_symmetric(SymF0, Nd):
    return NBI(SymF0[:Nd, Nd:])

def denovoNBI(F0, Nd, Nt):
    W = np.zeros((Nd, Nt))
    for idx in range(Nd):
        degree = np.count_nonzero(F0[idx, :])
        if degree > 0:
            W[idx, :] = F0[idx, :] / degree
        else:
            W[idx, :] = np.zeros(Nt)
    return W
