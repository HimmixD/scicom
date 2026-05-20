# %% general imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# %% exercise 13 def diffop(x):
def diffop(x):
    n = len(x)
    DDX = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            h = abs(x[1] - x[0])
            DDX[0, 0] = -1 / h
            DDX[0, 1] = 1 / h

        elif i == n - 1:
            h = abs(x[-1] - x[-2])
            DDX[-1, -1] = 1 / h
            DDX[-1, -2] = -1 / h

        else:
            h = abs(x[i + 1] - x[i - 1])
            DDX[i, i - 1] = -1 / h
            DDX[i, i + 1] = 1 / h
    return DDX


# %% test for given x
x = [0, 1, 2, 3, 4, 5]

print(diffop(x))


# %% aplly to a function
f = lambda x: -2 * np.exp(-(x**4))
inter = [-2, 2]
point_num = [10, 25, 50]


x_1000 = np.linspace(inter[0], inter[1], 1000)

plt.plot(x_1000, f(x_1000), label="f(x)", color="black", linestyle="dashed")
plt.plot(x_1000, np.gradient(f(x_1000), x_1000), label="f1(x)", color="black")
for num in point_num:
    x = np.linspace(inter[0], inter[1], num)
    DDX = diffop(x)
    fx = f(x)
    dfx = DDX @ fx
    plt.plot(
        x, dfx, label=f"f1(x), N = {num}", linestyle="--", marker="o", markersize=4
    )
plt.title("Numerical derivative of f(x)")
plt.legend()
plt.show()

# %% perturbed data
f = lambda x: -2 * np.exp(-(x**4))
inter = [-2, 2]
point_num = [500, 50]

rng = np.random.default_rng()
for num in point_num:
    x = np.linspace(inter[0], inter[1], num)
    fx = f(x)
    df = diffop(x) @ fx
    for i in range(len(fx)):
        fx[i] += rng.uniform(-0.002, 0.002)
    dfr = diffop(x) @ fx
    plt.scatter(x, (df - dfr), label=f"N = {num}")
plt.title("Num. der. of f(x) with perturbed data")
plt.legend()
plt.show()


# %% exercise 14
def trapezw(N, h):
    w = np.ones(N)
    w[0] = 0.5
    w[-1] = 0.5
    return w * h


def simpsonw(N, h):
    w = np.ones(N)
    w[0] = 1 / 6
    w[-1] = 1 / 6
    for i in range(1, N - 1):
        if i % 2 == 0:
            w[i] = 2 / 6
        else:
            w[i] = 4 / 6
    return w * h


# %% outputs for exercise 14
data = np.loadtxt("ex14.dat")
v = data[:, 1]
p = np.zeros(len(v))
for i in range(len(v)):
    p[i] = (1.2 / 4) * v[i] ** 3
p_60 = p[::2]

E_trapez_30 = (trapezw(len(p), 30 * 60) @ p) / 1e6
E_simpson_30 = (simpsonw(len(p), 30 * 60) @ p) / 1e6

E_trapez_60 = (trapezw(len(p_60), 60 * 60) @ p_60) / 1e6
E_simpson_60 = (simpsonw(len(p_60), 60 * 60) @ p_60) / 1e6


E_goal = 20 * 3.6e3  # MWh to MJ
area_trapez_30 = E_goal / E_trapez_30
area_simpson_30 = E_goal / E_simpson_30

area_trapez_60 = E_goal / E_trapez_60
area_simpson_60 = E_goal / E_simpson_60


entries = [
    [E_trapez_30, area_trapez_30],
    [E_trapez_60, area_trapez_60],
    [E_simpson_30, area_simpson_30],
    [E_simpson_60, area_simpson_60],
]
columns = ["Energy (MJ/m^2)", "Area for 20 MWh (m^2)"]
rows = ["Trapezoidal (30s)", "Trapezoidal (60s)", "Simpson's (30s)", "Simpson's (60s)"]

table = pd.DataFrame(entries, columns=columns, index=rows)
print(table)


plt.plot(data[:, 0], p, label="Power", color="orange")
plt.plot(data[::2, 0], p_60, label="Power - 60s", color="red")
plt.title("Power output of the wind turbine over time")
plt.xlabel("Time")
plt.ylabel("Power")
plt.legend()
plt.show()

# %%
