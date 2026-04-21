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
    Ak = [A]
    eigv = []
    while run:
        A_h, r = qr_hh(A)
        n, n = np.shape(A_h)

        Q = np.eye(n)
        for k in range(n):
            v_k = np.zeros(n)
            v_k[k:n] = A_h[k:n, k]
            Qk = np.eye(n) - 2 * np.outer(v_k, v_k.T) / np.dot(v_k, v_k)
            Q @= Qk

        A = Q.T @ A @ Q
        dif = sum(A[i, j] for i in range(n) for j in range(n) if i != j) / sum(
            A[i, i] for i in range(n)
        )

        Ak.append(A.copy())
        c += 1

        if c == kmax:
            run = False
        if c >= 2 and abs(dif) < eps:
            run = False

    for i in range(n):
        eigv.append(float(A[i, i]))
    eigv.sort()

    return eigv, Ak, c


def test_QR_eigensolve(n, eps, kmax):
    A_h = np.random.rand(n, n)
    A = np.tril(A_h) @ np.tril(A_h).T  # make A symmetric positive definite
    eigvals, _ = np.linalg.eig(A)
    eigvals.sort()
    eigvals_qr, Ak, c = QR_eigensolve(A, eps, kmax)
    err = np.linalg.norm(eigvals - eigvals_qr) / np.linalg.norm(eigvals)

    print(f"Relative Error with np.linalg.eig as comparison: {err}")
    #    print("Eigenvalues from numpy:", eigvals)
    #    print("Eigenvalues from QR algorithm:", eigvals_qr)

    for A in Ak:
        Am = np.min(abs(A))
        AM = np.max(abs(A))
        plt.figure(figsize=(4, 3))
        plt.imshow(A, cmap="ocean_r", vmin=Am, vmax=AM)
        plt.colorbar()
        plt.show()


# %%
test_QR_eigensolve(16, 0, 500)


# %%
def dir_veciter(A, eps):
    n, n = np.shape(A)
    #    eigenvalues = np.linalg.eig(A)
    #    lam_max = np.max(abs(eigenvalues))

    v_k = np.zeros((n, 1))
    v_k[0, 0] = 1
    V = v_k.copy()

    run = True
    c = 0
    lam = []

    while run:
        v_til = A @ v_k

        lam_k = np.dot(v_til.T, v_k)
        lam.append(float(lam_k))

        v_k = v_til / np.linalg.norm(v_til)
        V = np.hstack((V, v_k))

        c += 1

        if c > 2 and abs((lam[-1] - lam[-2]) / lam[-1]) < eps:
            run = False

    return lam, V


def inv_veciter(A, lest, eps):
    n, n = np.shape(A)
    v_k = np.zeros((n, 1))
    v_k[0, 0] = 1
    V = v_k.copy()

    run = True
    c = 0
    lam = []

    while run:
        v_til = np.linalg.inv(A - np.eye(n) * lest) @ v_k

        lam_k = lest + 1 / np.dot(v_til.T, v_k)
        lam.append(float(lam_k))

        v_k = v_til / np.linalg.norm(v_til)
        V = np.hstack((V, v_k))

        c += 1

        if c > 2 and abs((lam[-1] - lam[-2]) / lam[-1]) < eps:
            run = False

    return lam, V


# %%
A = np.array([[2, 0, 0.2], [0, -2, 1], [0.2, 1, -2]])

B = np.array([[2, 0, 0.2], [0, 2.02, 1], [0.2, 1, -2]])

A_eigenvalues = np.linalg.eig(A)[0]
B_eigenvalues = np.linalg.eig(B)[0]
real_A = max(A_eigenvalues, key=abs)
real_B = max(B_eigenvalues, key=abs)


lam_A, V_A = dir_veciter(A, 1e-5)
lam_B, V_B = dir_veciter(B, 1e-5)
lam_A_inv, V_A_inv = inv_veciter(A, 4, 1e-5)
lam_B_inv, V_B_inv = inv_veciter(B, 4, 1e-5)


print(real_A)
print(real_B)
print(lam_A[-1])
print(lam_B[-1])
print(lam_A_inv[-1])
print(lam_B_inv[-1])


# %%
