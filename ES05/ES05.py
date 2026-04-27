# %%
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import fsolve


# %%
def fixiter(phi, x0, eps, kmax):
    x = np.array([x0])
    run = True
    c = 0
    while run:
        x_new = phi(x[-1])
        x = np.vstack((x, x_new))
        c += 1
        if np.linalg.norm((x[-1] - x[-2]) / x[-2]) < eps or c >= kmax:
            run = False
    return x


def find_cobweb(phi, x):
    x_new = np.array([x for x in x for _ in range(2)])
    x_final = x_new[1:]
    xp = np.zeros((len(x_final), 1))
    for i in range(len(x_final)):
        if i % 2 == 0:
            xp[i] = fsolve(lambda t: t - x_final[i], x_final[i - 1 if i != 0 else i])[0]
        else:
            xp[i] = fsolve(
                lambda t: phi(t) - x_final[i], x_final[i - 1 if i != 0 else i]
            )[0]
    return xp, x_final


# %%
y = (
    lambda t: 10
    - 0.98 * t
    + 0.34 * t**2
    - 0.027 * t**3
    + 0.001 * t**4
    - 0.000016 * t**5
)
y_1 = lambda t: (y(t) + 0.98 * t) / 0.98
y_2 = lambda t: np.sqrt(abs(y(t) - 0.34 * t**2) / 0.34)
y_3 = lambda t: ((y(t) + 0.000016 * t**5) / 0.000016) ** (1 / 5)
y_4 = lambda t: t / (y(t) + 1)


# %%
x_1 = fixiter(y_1, 27, 1e-5, 5)

xs = np.linspace(15, 30, 100)
xp_1, x_new_1 = find_cobweb(y_1, x_1)
x_points = [27.319]
y_points = [27.319]

plt.plot(xs, xs, label="y=x")
plt.plot(xp_1[:6], x_new_1[:6], marker="o", label="y_1")
plt.plot(xs, y_1(xs), label="y_1")
plt.scatter(x_points, y_points, color="red", label="zero")
plt.legend()
plt.show()


# %%
x_2 = fixiter(y_2, 27, 1e-5, 5)

xs = np.linspace(26, 28, 100)
xp_2, x_new_2 = find_cobweb(y_2, x_2)
x_points = [27.319]
y_points = [27.319]

plt.plot(xs, xs, label="y=x")
plt.plot(xp_2, x_new_2, marker="o", label="y_2")
plt.plot(xs, y_2(xs), label="y_2")
plt.scatter(x_points, y_points, color="red", label="zero")
plt.legend()
plt.show()


# %%
x_3 = fixiter(y_3, 27, 1e-5, 25)

xs = np.linspace(26.8, 27.6, 100)
xp_3, x_new_3 = find_cobweb(y_3, x_3)
x_points = [27.319]
y_points = [27.319]

plt.plot(xs, xs, label="y=x")
plt.plot(xp_3, x_new_3, marker="o", label="y_3")
plt.plot(xs, y_3(xs), label="y_3")
plt.scatter(x_points, y_points, color="red", label="zero")
plt.legend()
plt.show()


# %%
def newton(f, fp, x0, eps, kmax):
    x = np.array([x0])
    run = True
    c = 0
    while run:
        x_new = x[-1] - f(x[-1]) / fp(x[-1])
        x = np.vstack((x, x_new))
        c += 1
        if np.linalg.norm((x[-1] - x[-2]) / x[-2]) < eps or c >= kmax:
            run = False
    return x


def find_newton_cobweb(f, fp, x):
    functions = []
    for i in x:
        slope = fp(i)
        interception = f(i) - slope * i
        functions.append(lambda t, m=slope, b=interception: m * t + b)
    return functions


# %%
f = lambda x: -np.sin(x) / x
fp = lambda x: (np.sin(x) - x * np.cos(x)) / x**2

x = newton(f, fp, 5, 1e-6, 100)
xs = np.linspace(4, 8, 100)
functions = find_newton_cobweb(f, fp, x)


for i, y in enumerate(functions[:3]):
    plt.plot(xs, y(xs), label=f"f_{i}'", color="green")
plt.plot(xs, f(xs), label="f(x)", color="blue")
plt.plot(xs, 0 * xs, label="y=0", color="grey")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# %%
