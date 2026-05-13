# %% initial imports
import numpy as np
import matplotlib.pyplot as plt


# %% defining of the needed functions for exercise 11
def get_equidistant(func, node_count, a, b, n):
    x = np.linspace(a, b, n)
    t = np.linspace(a, b, node_count)
    y = func(t)
    return t, y, x


def get_chebyshev(func, node_count, a, b, n):
    x = np.linspace(a, b, n)
    t = np.array([0.0] * node_count)
    for i in range(1, node_count + 1):
        t[i - 1] = (a + b) / 2 + (b - a) / 2 * np.cos(
            (2 * i - 1) / (2 * node_count) * np.pi
        )
    y = func(t)
    return t, y, x


def calc_L(x, t, i):
    n = len(t)
    L = 1
    for j in range(n):
        if j != i:
            L *= (x - t[j]) / (t[i] - t[j])
    return L


def Lagrange(t, f, x):
    n = len(t)
    result = np.array([0.0] * len(x))
    for i in range(n):
        for j, x_j in enumerate(x):
            result[j] += f[i] * calc_L(x_j, t, i)
    return result


# %% plotting the results of exercise 11
func = lambda x: np.sin(x) / (1 + x**2)
n = [4, 8, 12, 24]
a, b = (-2, 10)
x_numb = 100

plt.figure(figsize=(15, 10))
for i, n in enumerate(n):
    t, f, x = get_equidistant(func, n, a, b, x_numb)
    y = Lagrange(t, f, x)

    plt.subplot(2, 4, i + 1)
    plt.scatter(t, f, color="black", label="nodes")
    plt.plot(x, func(x), color="red", label="f(x)")
    plt.plot(x, y, linestyle="--", color="blue", label="P(x)")
    plt.title(f"Equidistant, nodes={n}")
    plt.axis([a - 0.5, b + 0.5, -0.5, 0.5])
    if i == 0:
        plt.legend()

    plt.subplot(2, 4, i + 5)
    t, f, x = get_chebyshev(func, n, a, b, x_numb)
    y = Lagrange(t, f, x)

    plt.scatter(t, f, color="black", label="nodes")
    plt.plot(x, func(x), color="red", label="f(x)")
    plt.plot(x, y, linestyle="--", color="blue", label="P(x)")
    plt.title(f"Chebyshev, nodes={n}")
    plt.axis([a - 0.5, b + 0.5, -0.5, 0.5])


# %%
