import numpy as np
import timeit as ti
import matplotlib.pyplot as plt
import lu_decomp as lu


def create_problem(n):
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = np.random.rand()
    b = np.random.rand(n, 1)
    return (L, b)


def create_problem_full(n):
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)
    return (A, b)


def tri_solve(L, b):
    x = np.empty((len(b), 1))
    for i in range(len(x)):
        x[i] = (b[i] - sum(x[j] * L[i, j] for j in range(i))) / L[i, i]
    return x


def tri_solve_upper(R, z):
    n = len(z)
    x = np.empty((n, 1))
    for i in reversed(range(n)):
        x[i] = (z[i] - sum(x[j] * R[i, j] for j in range(i, n))) / R[i, i]
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


A = np.array([[10 ** (-17), 1], [2, 1]])
b = np.array([[1], [3]])


def lu_solve(A, b, pv):
    M, z = lu.lu(A, pv)
    n, n = M.shape
    L = np.zeros((n, n))
    R = np.zeros((n, n))
    b = b[z]
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = M[i, j]
            elif i == j:
                L[i, j] = 1
                R[i, j] = M[i, j]
            else:
                R[i, j] = M[i, j]
    z_sol = tri_solve(L, b)
    x = tri_solve_upper(R, z_sol)
    x = x[z]
    return x


def test_lu_solve(n):
    A, b = create_problem_full(n)
    x_own = lu_solve(A, b, False)
    x_np = np.linalg.solve(A, b)
    print("Own solution:\n", x_own)
    print("Numpy solution:\n", x_np)
    print(np.allclose(x_own, x_np))


def plot_times_2():
    n = [5, 10, 20, 50, 100, 200, 400]
    time_own = np.zeros(len(n))
    time_own2 = np.zeros(len(n))
    time_np = np.zeros(len(n))
    for i, x in enumerate(n):
        A, b = create_problem_full(x)

        time_own[i] = measure_time(lu_solve, A, b, False)

        time_np[i] = measure_time(np.linalg.solve, A, b)

    plt.plot(n, time_own, label="lu_solve")
    plt.plot(n, time_np, label="linalg.solve")
    plt.xlabel("Mat. dim. n")
    plt.ylabel("Comput. time (s)")
    plt.legend()
    plt.show()


B = np.array([[1, 2, 3], [3, 1, 4], [6, 9, 1]])
u = np.array([[6], [8], [16]])

plot_times_2()
