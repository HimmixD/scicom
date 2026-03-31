# %% generell imports
import numpy as np
import matplotlib.pyplot as plt


# %% functions
def pol_fit(x, y, n):
    """Fit a polynomial of degree n to the data (x, y) and return the coefficients."""
    M = np.array([[x_i**(j-1) for j in range(1, n + 1)] for x_i in x])
    M_T = M.transpose()
    A = M_T @ M
    b = M_T @ y
    coeffs = np.linalg.solve(A, b)
    return coeffs

print(pol_fit([1, 2, 3], [1, 4, 9], 5))  # Should return coefficients close to [0, 0, 1]



# %% test data
Z = np.loadtxt("ex05.dat")
print(Z)
oders = [2, 3, 6, 9, 20]
for n in oders:
    coeffs = pol_fit(Z[:, 0], Z[:, 1], n)
    

# %%
