# %%
import numpy as np
import matplotlib.pyplot as plt


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


print(fixiter(y_4, 27, 1e-5, 50))


# %%
x_1 = fixiter(y_1, 27, 1e-5, 5)
x_2 = fixiter(y_2, 27, 1e-5, 5)
x_3 = fixiter(y_3, 27, 1e-5, 5)


xs = np.linspace(15, 30, 100)
plt.plot(xs, xs, label="y=x")
plt.plot(xs, y_1(xs), label="y_1")
plt.plot(xs, y_2(xs), label="y_2")
plt.plot(xs, y_3(xs), label="y_3")
plt.legend()
plt.show()

# %%
