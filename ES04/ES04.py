# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from qr_decomp import qr_hh


# %%
def QR_eigensolve(A, eps, kmax):
    A = A.copy()
    run = True
    c = 0
    dif = np.inf
    Ak = []
    while run:
        if c == kmax:
            run = False
        if c >= 2 and dif < eps:
            run = False

        A_h, r = qr_hh(A)
        n, n = np.shape(A_h)

        for k in range(n):
            Q = np.identity(n)
            v_k = np.zeros(n)
            v_k[k:n] = A_h[k:n, k]
            Q -= 2 * (v_k @ v_k.T) / (v_k.T @ v_k)

        A = Q.T @ A @ Q
        dif = sum(A[i, j] for i in range(n) for j in range(n) if i != j) / sum(
            A[i, i] for i in range(n)
        )

        Ak.append(A)
        c += 1


# c = iteration counter, run = boolean to control the while loop
