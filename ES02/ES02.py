# %% Generel imports
import numpy as np
from tabulate import tabulate


# %% Function and value definitions

k_B = 8.617e-5  # Boltzmann constant in eV/K
Temps = [10, 300, 1000]  # Temperatures in K
m = -1  # Chemical potential in eV
eps = np.linspace(-3, 0, 16)  # Energy range in eV


def f(x, T):
    return 1 / (np.exp((x - m) / (k_B * T)) + 1)


def p(x, T):
    return 1 - f(x, T)


def p_2(x, T):
    return (np.exp((x - m) / (k_B * T))) / (np.exp((x - m) / (k_B * T)) + 1)


# %% Table of values
for T in Temps:
    data = []
    for x in reversed(eps):
        data.append([x, p(x, T), p_2(x, T)])
    headers = [f"ε for Temperature: {T}", "p(ε)", "p_2(ε)"]
    print(tabulate(data, headers=headers, tablefmt="grid"))

# The values are different, because in the exercise sheet they probably used a higher precision. Could be redone!


# %% Definition of values
a = [x / 10 for x in range(1, 7)]
b = np.array([[x] for x in range(1, 7)])
b_add = np.zeros((6, 1))
b_add[4] = 0.001
b_s = b + b_add
V = np.array([[x ** (j - 1) for j in range(1, 7)] for x in a])


# %% solving the first system of equations (b)
print(np.linalg.solve(V, b))


# %% solving the second system of equations (b)
print(np.linalg.solve(V, b_s))

# %% comparing the relative errors
x = np.linalg.solve(V, b)
x_s = np.linalg.solve(V, b_s)

dx = np.linalg.norm(x_s - x) / np.linalg.norm(x)
db = np.linalg.norm(b_s - b) / np.linalg.norm(b)
print(f"Delta_x = {dx}", f"Delta_b = {db}")
