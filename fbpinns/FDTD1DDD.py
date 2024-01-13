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
        NX,
        NSTEPS,
        DELTAX,
        DELTAT,
):
    eps0 = 1  # 8.854e-12
    eps = 1
    mu0 = 1  # 4 * np.pi * 1e-7
    imp = np.sqrt(mu0 / (eps0 * eps))
    c = 1 / np.sqrt(eps0 * mu0)

    f = 1  # 3e10
    w = 2 * np.pi * f
    dz = DELTAX #空间离散化的步长，被设置为波长的20分之一
    dt = DELTAT #时间步长，由空间步长 dz 和光速 c 计算而得

    Zmax=NX #离散化空间后的单元个数
    Nmax=NSTEPS #时间步数

    Hy = np.zeros((Zmax, Nmax)) # 初始化磁场的数组
    Ex = np.zeros((Zmax, Nmax)) #初始化电场的数组

    # x_len = dz * Zmax #空间长度
    x_cood = np.linspace(xmin, xmax, Zmax) #离散化后的空间坐标
    source_x = int(Zmax / 2.0)

    for t in range(0, Nmax-1): #t∈[0,Nmax-2] 循环不包括最后一个时间点
        Ex[source_x, t] = np.sin(w * t * dt)
        Hy[-1, t + 1] = Hy[-2, t]  # abc
        # 内层循环：在空间维度上的循环
        for z in range(0, Zmax - 1):
            Hy[z, t+1] = Hy[z, t] + (Ex[z + 1, t] - Ex[z, t]) * dt / (mu0 * dz)

        Ex[0, t + 1] = Ex[1, t]  # abc
        for z in range(1, Zmax):
            Ex[z, t+1] = Ex[z, t] + (Hy[z, t+1] - Hy[z-1, t+1]) * dt / (eps0 *eps* dz)

    return Hy,Ex



