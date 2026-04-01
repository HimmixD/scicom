# %% generell imports
import numpy as np
import matplotlib.pyplot as plt
from qr_decomp import qr_hh
import random as rd


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
    for k in range(
        n
    ):  # loop over all columns of A to apply the Householder transformations to b
        v_k = np.zeros(m)
        v_k[k:m] = A[k:m, k]
        pj = 2 * np.dot(v_k, Q_b) / np.dot(v_k, v_k) * v_k
        Q_b -= pj

    x = np.zeros((n, 1))
    R = np.empty((n, n))
    for i in range(
        n
    ):  # construct R from the upper triangular part of A and the diagonal elements from r
        R[i, i] = r[i]
        R[i, i + 1 : n] = A[i, i + 1 : n]
    for i in reversed(range(n)):  # back substitution to solve R x = Q_b
        x[i] = (Q_b[i] - sum(x[j] * R[i, j] for j in range(i + 1, n))) / R[i, i]

    return x


def create_problem():
    n = rd.randint(1, 100)
    m = rd.randint(n, n + 10)
    A = np.random.rand(m, n)
    x_true = np.random.rand(n)
    b = A @ x_true

    return A, b, x_true, m, n


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


# %% 6. test for randomly generated A and b
A, b, x_true, m, n = create_problem()
A_qr, r = qr_hh(A)
x_qr = qr_solve(A_qr, r, b).transpose()
lstsq_solution = np.linalg.lstsq(A, b)[0]
# print("Least squares solution:", lstsq_solution)
# print("True solution:", x_true)
# print("QR solution:", x_qr)

print(f"Tested for randomly generated A and b with m = {m} and n = {n}")
if np.allclose(x_qr, x_true):
    print("The QR solution is close to the true solution.")
else:
    print("The QR solution is not close to the true solution.")


# %% 6. test qr_solve with the given A and b
A = np.array([[1, -1], [10 ** (-8), 0], [0, 10 ** (-8)]])
b = np.array([0, 0, 10 ** (-6)])
A_qr, r = qr_hh(A)
qr_solve(A_qr, r, b)
# expected output for x is (50,50)

# %% 6. extra: solving least squares problem from ex05.dat
Z = np.loadtxt("ex05.dat")
x = Z[:, 0]
y = Z[:, 1]
m = len(x)
n = 6
M = np.array([[x_i ** (j - 1) for j in range(1, n + 1)] for x_i in x])
A_qr, r = qr_hh(M)
coeffs_qr = qr_solve(A_qr, r, y)
print("Coefficients from QR decomposition:", coeffs_qr)
print("Coefficients from pol_fit function:", pol_fit(x, y, n)[0])
