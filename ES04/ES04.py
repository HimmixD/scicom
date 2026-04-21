# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from qr_decomp import qr_hh


# %%
def QR_eigensolve(A, eps, kmax):
    run = True
    c = 0
    dif = np.inf
    Ak = []
    eigv = []
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
            Qk = np.identity(n) - 2 * np.outer(v_k, v_k.T) / np.dot(v_k, v_k)
            Q *= Qk

        A = Q.T @ A @ Q
        dif = sum(A[i, j] for i in range(n) for j in range(n) if i != j) / sum(
            A[i, i] for i in range(n)
        )

        Ak.append(A)
        c += 1
    print(A)
    print(c)
    for i in range(n):
        eigv.append(float(A[i, i]))
    eigv.sort()
    
    return eigv, Ak


# c = iteration counter, run = boolean to control the while loop


def test_QR_eigensolve(n, eps, kmax):
    A_h = np.random.rand(n, n)
    A = np.tril(A_h) @ np.tril(A_h).T  # make A symmetric positive definite
    eigvals, _ = np.linalg.eig(A)
    eigvals_qr, Ak = QR_eigensolve(A, eps, kmax)
    print(Ak)
    #print("Eigenvalues from numpy:", eigvals)
    #print("Eigenvalues from QR algorithm:", eigvals_qr)



# %%
test_QR_eigensolve(5, 0.005, 50)
# %%
