# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def fixiter(phi, x0, eps, kmax):
    x = np.zeros(kmax)
    x[0] = x0
    run = True
    c = 0
    while run:
        
