# %% generell imports
import numpy as np
import matplotlib.pyplot as plt
from qr_decomp import qr_hh

help(qr_hh)


# %% functions
def pol_fit(x, y, n):
    """Fit a polynomial of degree n to the data (x, y) and return the coefficients."""
    M = np.array([[x_i ** (j - 1) for j in range(1, n + 1)] for x_i in x])
    M_T = M.transpose()
    A = M_T @ M
    b = M_T @ y
    coeffs = np.linalg.solve(A, b)
    return coeffs, M


def qr_solve(A, r, b):
    m, n = A.shape
    Q_b = b.copy()
    for k in range(n):
        v_k = np.zeros(m)
        v_k[k] = 1
        v_k[k + 1 : m] = A[k + 1 : m, k]
        pj = 2 * np.dot(v_k, Q_b) / np.dot(v_k, v_k) * v_k
        Q_b -= pj
    print("Q^T b after applying Q^T:", Q_b)
    x = np.empty((n, 1))
    for i in reversed(range(n)):
        x[i] = (
            Q_b[i] - sum(x[j] * r[i] + x[j] * A[i, j] for j in range(i + 1, n))
        ) / r[i]
    print(x)
    # funkt nicht, muss noch checken warum


# %% 5. plot with x values from the data
Z = np.loadtxt("ex05.dat")
oders = [2, 3, 6, 9, 20]
plt.plot(Z[:, 0], Z[:, 1], "o", label="Data")
for n in oders:
    coeffs, M = pol_fit(Z[:, 0], Z[:, 1], n)
    values = M @ coeffs.transpose()
    plt.plot(Z[:, 0], values, label=f"n={n}")
plt.legend()
plt.show()


# %% 5. plot with 1000 evenly spaced x values between the minimum and maximum of the data
Z = np.loadtxt("ex05.dat")
oders = [2, 3, 6, 9, 20]
x = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 1000)
plt.plot(Z[:, 0], Z[:, 1], "o", label="Data")
for n in oders:
    coeffs, M = pol_fit(Z[:, 0], Z[:, 1], n)
    y = sum(coeffs[j - 1] * x ** (j - 1) for j in range(1, n + 1))
    y_5 = sum(coeffs[j - 1] * 5 ** (j - 1) for j in range(1, n + 1))
    plt.plot(x, y, label=f"n={n}")
    print(f"n={n}: p(5) = {y_5}")
plt.legend()
plt.show()
"""At higher ns, the polynomial fits the data better, but it also starts to oscillate wildly between the data points, 
especially near the edges of the interval. 
This is known as Runge's phenomenon. The value of p(5) also changes significantly with increasing n, 
which indicates that the polynomial is not stable and can give very different results for small changes in the input data."""


# %% 6.
A = np.array([[1, -1], [10 ** (-8), 0], [0, 10 ** (-8)]])
b = np.array([0, 0, 10 ** (-6)])
A_qr, r = qr_hh(A)
qr_solve(A_qr, r, b)
# expected output for x is (50,50)


# %%
