# # -*- coding: utf-8 -*-
# """
# Created on Sat Jan 13 05:28:00 2024
#
# @author: shuw1
# """
#
# # -*- coding: utf-8 -*-
# """
# Created on Sat Jan  6 08:48:01 2024
#
# @author: shuw
# """
#
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
# # def FDTD1DD():
# eps0 = 1#8.854e-12
# eps = 1
# mu0 = 1#4 * np.pi * 1e-7
# imp = np.sqrt(mu0/(eps0*eps))
# c = 1 / np.sqrt(eps0 * mu0)
#
# f = 1#3e10
# w = 2 * np.pi * f
# lam = c / f
# dz = 0.008368200836820083
# dt = 0.001680672268907563
#
# Distance = 10 * lam
# Zmax=240
# Nmax=596
#
# imp = np.sqrt(mu0/(eps0*eps))
#
# Hy = np.zeros((Zmax, Nmax))
# Ex = np.zeros((Zmax, Nmax))
#
# # x_len = dz * Zmax
# x_cood = np.linspace(-1, 1, 240)
#
# # t_len = dt * Nmax
# t_cood = np.linspace(0, 1, 596)
#
# def source(x,t,sd):
#     e=-0.5 * (x ** 2 + t ** 2) / (sd ** 2)
#     return 2e1 * np.exp(e) * (1+e) #ricker source
#     # return np.exp(e) # gaussian source
#
# source_x = range(0,Zmax)#int(Zmax/2.0)
# for t in range(0, Nmax-1):
#     # 内层循环：在空间维度上的循环
#     Ex[source_x,t] = Ex[source_x,t]+source(x_cood[source_x],t_cood[t],0.1)
#     Hy[-1, t + 1] = Hy[-2, t]  # abc
#     for z in range(0, Zmax - 1):
#         Hy[z, t + 1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 *dz)
#
#
#     Ex[0, t + 1] = Ex[1, t]  # abc
#     for z in range(1, Zmax):
#         Ex[z, t + 1] = Ex[z, t] + (Hy[z, t + 1] - Hy[z - 1, t + 1])*dt / (eps0*eps*dz)
#
#     if t % 20 == 0:
#         plt.clf()
#         plt.title("Ex after t=%i" % t)
#         plt.plot(x_cood, Ex[:,t+1], x_cood,Hy[:,t+1])
#         # plt.ylim([-8, 8])
#         plt.show()
#         plt.clf()
#         plt.close()
if __name__ == "__main__":

    xmin = -2
    xmax = 2
    tmin = 0
    tmax = 1
    sd = 0.08
    NX = 240
    NSTEPS = 596
    DELTAX = 0.008368200836820083
    DELTAT = 0.001680672268907563
    eps0 = 1#8.854e-12
    eps = 1
    mu0 = 1#4 * np.pi * 1e-7
    imp = np.sqrt(mu0/(eps0*eps))
    c = 1 / np.sqrt(eps0 * mu0)

    f = 1#3e10
    w = 2 * np.pi * f
    lam = c / f
    imp = np.sqrt(mu0/(eps0*eps))

    #MODEL
    dz = DELTAX
    dt = DELTAT
    Zmax=NX
    Nmax=NSTEPS


    Hy = np.zeros((Zmax, Nmax))
    Ex = np.zeros((Zmax, Nmax))

    x_cood = np.linspace(xmin, xmax, Zmax)
    t_cood = np.linspace(tmin, tmax, Nmax)

    #source
    def source(x,t,sd):
        e=-0.5 * (x ** 2 + t ** 2) / (sd ** 2)
        return 1200 * np.exp(e) * (1 + e) #ricker source

    source_x = range(0, Zmax)  # int(Zmax/2.0)
    # x_cood = np.linspace(xmin, xmax, Zmax)  # 离散化后的空间坐标
    # Ex[:, 0:1] = np.exp(-(((x_cood / sd) ** 2) / 2)).reshape(-1, 1)
    # Hy[0:-1, 0:1] = -Ex[1:, 0:1] / imp  # shift by 1 because of stagging grid
    for t in range(0, Nmax - 1):
        # 加源
        Ex[source_x, t] = Ex[source_x, t] + source(x_cood[source_x], t_cood[t], sd) * dt / (eps0 * eps)
        # 迭代过程
        Hy[-1, t + 1] = Hy[-2, t]  # abc
        for z in range(0, Zmax - 1):
            Hy[z, t + 1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 * dz)

        Ex[0, t + 1] = Ex[1, t]  # abc
        for z in range(1, Zmax):
            Ex[z, t + 1] = Ex[z, t] + (Hy[z, t + 1] - Hy[z - 1, t + 1]) * dt / (eps0 * eps * dz)

        if t % 20 == 0:
            plt.clf()
            plt.title("Ex after t=%i" % t)
            plt.plot(x_cood, Ex[:, t + 1], x_cood, Hy[:, t + 1])
            plt.ylim([-8, 8])
            plt.show()
            plt.clf()
            plt.close()