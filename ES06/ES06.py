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


# %% defining the aitken-neville scheme for exercise 12
 def aitken_neville(f, t, x):
    n = len(t)
    P = np.zeros((n, n))
    for i in range(n):
        P[i][0] = f[i]
    for i in range(0, n):
        for j in range(1, n):
            if i >= j:
                P[i][j] = (x - t[i - j]) * P[i][j - 1] - (x - t[i]) * P[i - 1][j - 1]
                P[i][j] /= t[i] - t[i - j]
    return P


# %% calculating the results for exercise 12 via aitken-neville scheme
t_5 = np.array([0, 1, 2, 3, 4])
f_5 = np.array([999.8425, 999.9017, 999.9416, 999.9653, 999.9720])

P_5 = aitken_neville(f_5, t_5, 3.98)

t_6 = np.array([0, 1, 2, 3, 4, 5])
f_6 = np.array([999.8425, 999.9017, 999.9416, 999.9653, 999.9720, 999.9645])

P_6 = aitken_neville(f_6, t_6, 3.98)

P_5_bigger = np.vstack((P_5, np.zeros((1, 5))))
P_5_bigger = np.hstack((P_5_bigger, np.zeros((6, 1))))
P_6_5_diff = P_6 - P_5_bigger

print(f"p(3.98) with 5 nodes: {P_5[-1][-1]}")
print(f"p(3.98) with 6 nodes: {P_6[-1][-1]}")
print(f"Difference: {P_6[-1][-1] - P_5[-1][-1]}")

print("P_5:" + str(P_5))
print("P_6:" + str(P_6))
print("Difference:" + str(P_6_5_diff))



# %% defining the needed functions for exercise 12
def get_b(t, f):
    n = len(t)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, 0] = 1
        for j in range(1, i + 1):
            product = 1
            for k in range(j):
                product *= (t[i] - t[k])
            A[i, j] = product
    b = np.linalg.solve(A, f)
    return b

def P(b, t, x):
    P = b[0]
    for i in range(1, len(b)):
        polynomial = 1
        for j in range(i):
            polynomial *= (x - t[j])
        P += b[i] * polynomial
    return P


# %% calculating the results for exercise 12 
print("p(3.98) with 6 nodes: " + str(P(get_b(t_6, f_6), t_6, 3.98)))

plt.plot(np.linspace(0, 5, 100), P(get_b(t_6, f_6), t_6, np.linspace(0, 5, 100)), color="blue", label="P(x)")
plt.scatter(t_6, f_6, color="black", label="nodes")
plt.legend()
plt.show()



# %%
