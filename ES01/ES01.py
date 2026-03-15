import numpy as np
import timeit as ti
import matplotlib.pyplot as plt


def create_problem(n):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = np.random.rand()
    b = np.random.rand(n, 1)
    return (L, b)


def tri_solve(L, b):
    x = np.empty((len(b), 1))
    for i in range(len(x)):
        x[i] = (b[i] - sum(x[j] * L[i, j] for j in range(i))) / L[i, i]
    return b


def tri_solve2(L, b):
    x = np.empty((len(b), 1))
    for i in range(len(x)):
        x[i] = b[i] / L[i, i]
        for j in range(i + 1, len(x)):
            b[j] = b[j] - (x[i] * L[j, i])
    return x


L = np.array([[3, 0, 0], [1, 5, 0], [2, 3, 1]])
b = np.array([[3], [2], [1]])


def measure_time(func, *args):
    c = 0
    for i in range(100):
        start = ti.default_timer()
        _ = func(*args)
        end = ti.default_timer()
        c += end - start
    return c / 100


def plot_times():
    n = [5, 10, 20, 50, 100, 200, 400]
    time_own = np.zeros(len(n))
    time_own2 = np.zeros(len(n))
    time_np = np.zeros(len(n))
    for i, x in enumerate(n):
        L, b = create_problem(x)

        time_own[i] = measure_time(tri_solve, L, b)

        time_own2[i] = measure_time(tri_solve2, L, b)

        time_np[i] = measure_time(np.linalg.solve, L, b)

    print(time_own)
    print(time_np)

    plt.plot(n, time_own, label="tri_solve")
    plt.plot(n, time_own2, label="tri_solve2")
    plt.plot(n, time_np, label="linalg.solve")
    plt.xlabel("Mat. dim. n")
    plt.ylabel("Comput. time (s)")
    plt.legend()
    plt.show()


plot_times()
