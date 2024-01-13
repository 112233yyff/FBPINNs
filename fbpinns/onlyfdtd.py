# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 08:48:01 2024

@author: shuw
"""

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
# def FDTD1DD():
eps0 = 1#8.854e-12
eps = 1
mu0 = 1#4 * np.pi * 1e-7
imp = np.sqrt(mu0/(eps0*eps))
c = 1 / np.sqrt(eps0 * mu0)

f = 1#3e10
w = 2 * np.pi * f
lam = c / f
dz = lam / 20
dt = dz / c

Distance = 10 * lam
Zmax=int(np.ceil(10*lam/dz))
Nmax=500

imp = np.sqrt(mu0/(eps0*eps))

Hy = np.zeros((Zmax, Nmax))
Ex = np.zeros((Zmax, Nmax))

x_len = dz * Zmax
x_cood = np.linspace(-x_len /2., x_len /2., Zmax)

source_x = int(Zmax/2.0)
for t in range(0, Nmax-1):
    # 内层循环：在空间维度上的循环
    Ex[:, t] = np.exp(-0.5 * (x_cood ** 2 + t ** 2) / (0.5 ** 2))
    Hy[-1, t + 1] = Hy[-2, t]  # abc
    for z in range(0, Zmax - 1):
        Hy[z, t + 1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 *dz)


    Ex[0, t + 1] = Ex[1, t]  # abc
    for z in range(1, Zmax):
        Ex[z, t + 1] = Ex[z, t] + (Hy[z, t + 1] - Hy[z - 1, t + 1])*dt / (eps0*eps*dz)

    if t % 20 == 0:
        plt.clf()
        plt.title("Ex after t=%i" % t)
        plt.plot(x_cood, Ex[:,t+1], x_cood,Hy[:,t+1])
        # plt.ylim([-1, 1])
        plt.show()
        plt.clf()
        plt.close()