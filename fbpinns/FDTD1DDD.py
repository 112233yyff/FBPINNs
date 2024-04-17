# # -*- coding: utf-8 -*-
# """
# Created on Sat Jan  6 08:48:01 2024
#
# @author: shuw
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
# from time import sleep
# def FDTD1DD(
#         xmin,
#         xmax,
#         tmin,
#         tmax,
#         sd,
#         NX,
#         NSTEPS,
#         DELTAX,
#         DELTAT,
# ):
#     eps0 = 1  # 8.854e-12
#     eps = 1
#     mu0 = 1  # 4 * np.pi * 1e-7
#     imp = np.sqrt(mu0 / (eps0 * eps))
#     c = 1 / np.sqrt(eps0 * mu0)
#
#     f = 1  # 3e10
#     w = 2 * np.pi * f
#     dz = DELTAX #空间离散化的步长，被设置为波长的20分之一
#     dt = DELTAT #时间步长，由空间步长 dz 和光速 c 计算而得
#
#     Zmax=NX #离散化空间后的单元个数
#     Nmax=NSTEPS #时间步数
#
#     Hy = np.zeros((Zmax, Nmax)) # 初始化磁场的数组
#     Ex = np.zeros((Zmax, Nmax)) #初始化电场的数组
#
#     x_cood = np.linspace(xmin, xmax, Zmax)
#
#     t_cood = np.linspace(tmin, tmax, Nmax)
#
#     def source(x, t, sd):
#         return np.exp(-0.5 * (x ** 2 + t ** 2) / (sd ** 2))
#     source_x = int(Zmax / 2.0)
#
#     for t in range(0, Nmax - 1):
#         # 内层循环：在空间维度上的循环
#         Ex[source_x, t] = source(x_cood[source_x], t_cood[t], sd)
#         Hy[-1, t + 1] = Hy[-2, t]  # abc
#         # 内层循环：在空间维度上的循环
#         for z in range(0, Zmax - 1):
#             Hy[z, t+1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 * dz)
#
#         Ex[0, t + 1] = Ex[1, t]  # abc
#         for z in range(1, Zmax):
#             Ex[z, t+1] = Ex[z, t] + (Hy[z, t+1] - Hy[z-1, t+1]) * dt / (eps0 *eps* dz)
#
#     return Hy,Ex
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 05:28:00 2024

@author: shuw1
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 08:48:01 2024

@author: shuw
"""

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
def FDTD1DD(
        xmin,
        xmax,
        tmin,
        tmax,
        sd,
        NX,
        NSTEPS,
        DELTAX,
        DELTAT,
):
    eps0 = 1#8.854e-12
    eps = 1
    mu0 = 1#4 * np.pi * 1e-7
    imp = np.sqrt(mu0/(eps0*eps))
    c = 1 / np.sqrt(eps0 * mu0)

    f = 1#3e10
    w = 2 * np.pi * f
    lam = c / f
    dz = DELTAX
    dt = DELTAT

    Distance = 10 * lam
    Zmax=NX
    Nmax=NSTEPS

    imp = np.sqrt(mu0/(eps0*eps))

    Hy = np.zeros((Zmax, Nmax))
    Ex = np.zeros((Zmax, Nmax))

    x_cood = np.linspace(xmin, xmax, Zmax)
    t_cood = np.linspace(tmin, tmax, Nmax)

    # def source(x,t,sd):
    #     e=-0.5 * (x ** 2 + t ** 2) / (sd ** 2)
    #     return 1e6 * np.exp(e) * (1 + e) #ricker source
    #     # return np.exp(e) # gaussian source

    # source_x = range(0,Zmax)#int(Zmax/2.0)
    Ex[:, 0:1] = np.exp(-(((x_cood / sd) ** 2) / 2)).reshape(-1, 1)
    # Hy[0:-1, 0:1] = -Ex[1:, 0:1] / imp  # shift by 1 because of stagging grid
    for t in range(0, Nmax-1):
        #加源
        # Ex[source_x, t] = Ex[source_x, t] + source(x_cood[source_x], t_cood[t], sd) * dt / (eps0 * eps)
        #迭代过程
        # Hy[-1, t + 1] = -Hy[-2, t]  # abc
        for z in range(0, Zmax - 1):
            Hy[z, t + 1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 *dz)

        # Ex[0, t + 1] = -Ex[1, t]  # abc
        for z in range(1, Zmax):
            Ex[z, t + 1] = Ex[z, t] + (Hy[z, t + 1] - Hy[z - 1, t + 1])*dt / (eps0*eps*dz)
        Ex[59, t + 1] = 0  # pec
        # if t % 20 == 0:
        #     plt.clf()
        #     plt.title("Ex after t=%i" % t)
        #     plt.plot(x_cood, Ex[:, t + 1], x_cood, Hy[:, t + 1])
        #     plt.ylim([-8, 8])
        #     plt.show()
        #     plt.clf()
        #     plt.close()
    return Hy, Ex

