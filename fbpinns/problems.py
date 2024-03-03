#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:11:31 2021

@author: bmoseley
"""

# This module defines a set of PDE problems to solve
# Each problem is defined by a problem class, which must inherit from the _Problem base class
# Each problem class must define the NotImplemented methods which compute the PINN physics loss,
# the gradients required to evaluate the PINN physics loss, the hard boundary conditions applied to
# the ansatz and the exact solution, if it exists.
## 该模块定义了一组要解决的偏微分方程问题
# 每个问题都由一个问题类定义，该类必须继承自 _Problem 基类
# 每个问题类必须定义未实现的方法，这些方法用于计算 PINN 物理损失、计算评估 PINN 物理损失所需的梯度、应用于假设函数的硬边界条件以及精确解（如果存在）

# Problem classes are used by constants.py when defining FBPINN / PINN problems (and subsequently main.py)

import numpy as np
import torch

import boundary_conditions
import losses

import sys

sys.path.insert(0, '../analytical_solutions/')
from burgers_solution import burgers_viscous_time_exact1

sys.path.insert(0, '../seismic-cpml')
from seismic_CPML_2D_pressure_second_order import seismicCPML2D
import pdb
from FDTD1DDD import FDTD1DD


class _Problem:
    "Base problem class to be inherited by different problem classes"

    @property
    def name(self):
        "Defines a name string (only used for labelling automated training runs)"
        raise NotImplementedError

    def __init__(self):
        raise NotImplementedError

    def physics_loss(self, x, *yj):
        "Defines the PINN physics loss to train the NN"
        raise NotImplementedError

    def get_gradients(self, x, y):
        "Returns the gradients yj required for this problem"

    def boundary_condition(self, x, *yj, args):
        "Defines the hard boundary condition to be applied to the NN ansatz"
        raise NotImplementedError

    def exact_solution(self, x, batch_size):
        "Defines exact solution if it exists"
        return None


# 1D problems

class Cos1D_1(_Problem):
    """Solves the 1D ODE:
        du
        -- = cos(wx)
        dx

        Boundary conditions:
        u (0) = A
    """

    @property
    def name(self):
        return "Cos1D_1_w%s" % (self.w)

    def __init__(self, w, A=0):
        # input params
        self.w = w

        # boundary params
        self.A = A

        # dimensionality of x and y
        self.d = (1, 1)

    def physics_loss(self, x, y, j):
        physics = j - torch.cos(self.w * x)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j

    def boundary_condition(self, x, y, j, sd):
        y, j = boundary_conditions.A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        Ap = self.A
        y = (1 / self.w) * torch.sin(self.w * x) + Ap
        j = torch.cos(self.w * x)
        return y, j


class Cos_multi1D_1(_Problem):
    """Solves the 1D ODE:
        du
        -- = w1*cos(w1x) + w2*cos(w2x)
        dx

        Boundary conditions:
        u (0) = A
    """

    @property
    def name(self):
        return "Cos_multi1D_1_w%sw%s" % (self.w1, self.w2)

    def __init__(self, w1, w2, A=0):
        # input params
        self.w1, self.w2 = w1, w2

        # boundary params
        self.A = A

        # dimensionality of x and y
        self.d = (1, 1)

    def physics_loss(self, x, y, j):
        physics = j - (self.w1 * torch.cos(self.w1 * x) + self.w2 * torch.cos(self.w2 * x))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        return y, j

    def boundary_condition(self, x, y, j, sd):
        y, j = boundary_conditions.A_1D_1(x, y, j, self.A, 0, sd)
        return y, j

    def exact_solution(self, x, batch_size):
        Ap = self.A
        y = torch.sin(self.w1 * x) + torch.sin(self.w2 * x) + Ap
        j = self.w1 * torch.cos(self.w1 * x) + self.w2 * torch.cos(self.w2 * x)
        return y, j


class Sin1D_2(_Problem):
    """Solves the 1D ODE:
        d^2 u
        ----- = sin(wx)
        dx^2

        Boundary conditions:
        u (0) = A
        u'(0) = B
    """

    @property
    def name(self):
        return "Sin1D_2_w%s" % (self.w)

    def __init__(self, w, A=0, B=0):
        # input params
        self.w = w

        # boundary params
        self.A = A
        self.B = B

        # dimensionality of x and y
        self.d = (1, 1)

    def physics_loss(self, x, y, j, jj):
        physics = jj - torch.sin(self.w * x)
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        jj = torch.autograd.grad(j, x, torch.ones_like(j), create_graph=True)[0]
        return y, j, jj

    def boundary_condition(self, x, y, j, jj, sd):
        y, j, jj = boundary_conditions.AB_1D_2(x, y, j, jj, self.A, self.B, 0, sd)
        return y, j, jj

    def exact_solution(self, x, batch_size):
        Ap = self.A
        Bp = self.B + (1 / self.w)
        y = -(1 / self.w ** 2) * torch.sin(self.w * x) + Bp * x + Ap
        j = -(1 / self.w) * torch.cos(self.w * x) + Bp
        jj = torch.sin(self.w * x)
        return y, j, jj


# 2D problems

class Cos_Cos2D_1(_Problem):
    """Solves the 2D PDE:
        du   du
        -- + -- = cos(wx) + cos(wy)
        dx   dy

        Boundary conditions:
        u(0,y) = (1/w)sin(wy) + A
    """

    @property
    def name(self):
        return "Cos_Cos2D_1_w%s" % (self.w)

    def __init__(self, w, A=0):
        # input params
        self.w = w

        # boundary params
        self.A = A

        # dimensionality of x and y
        self.d = (2, 1)

    def physics_loss(self, x, y, j0, j1):
        physics = (j0[:, 0] + j1[:, 0]) - (torch.cos(self.w * x[:, 0]) + torch.cos(
            self.w * x[:, 1]))  # be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]

        return y, j0, j1

    def boundary_condition(self, x, y, j0, j1, sd):
        # Apply u = tanh((x-0)/sd)*NN + A + (1/w)sinwy   ansatz

        A, w = self.A, self.w

        t0, jt0 = boundary_conditions.tanh_1(x[:, 0:1], 0, sd)

        sin = (1 / w) * torch.sin(w * x[:, 1:2])
        cos = torch.cos(w * x[:, 1:2])

        y_new = t0 * y + A + sin
        j0_new = jt0 * y + t0 * j0
        j1_new = t0 * j1 + cos

        return y_new, j0_new, j1_new

    def exact_solution(self, x, batch_size):
        Ap = self.A
        y = (1 / self.w) * torch.sin(self.w * x[:, 0:1]) + (1 / self.w) * torch.sin(self.w * x[:, 1:2]) + Ap
        j0 = torch.cos(self.w * x[:, 0:1])
        j1 = torch.cos(self.w * x[:, 1:2])
        return y, j0, j1


class Sin2D_1(_Problem):
    """Solves the 2D PDE:
        du   du
        -- + -- = -sin(w(x+y))
        dx   dy

        Boundary conditions:
        u(x,x) = (1/w)cos^2(wx) + A
    """

    @property
    def name(self):
        return "Sin2D_1_w%s" % (self.w)

    def __init__(self, w, A=0):
        # input params
        self.w = w

        # boundary params
        self.A = A

        # dimensionality of x and y
        self.d = (2, 1)

    def physics_loss(self, x, y, j0, j1):
        physics = (j0[:, 0] + j1[:, 0]) + (torch.sin(
            self.w * (x[:, 0] + x[:, 1])))  # be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]

        return y, j0, j1

    def boundary_condition(self, x, y, j0, j1, sd):
        # Apply u = tanh((x+y)/sd)*NN + A + (1/w)cos^2wx   ansatz

        A, w = self.A, self.w

        t, jt = boundary_conditions.tanh_1(x[:, 0:1] + x[:, 1:2], 0, sd)

        cos2 = (1 / w) * torch.cos(w * x[:, 0:1]) ** 2
        sin2 = -2 * torch.sin(w * x[:, 0:1]) * torch.cos(w * x[:, 0:1])

        y_new = t * y + A + cos2
        j0_new = jt * y + t * j0 + sin2
        j1_new = jt * y + t * j1

        return y_new, j0_new, j1_new

    def exact_solution(self, x, batch_size):
        Ap = self.A
        y = (1 / self.w) * torch.cos(self.w * x[:, 0:1]) * torch.cos(self.w * x[:, 1:2]) + Ap
        j0 = -torch.sin(self.w * x[:, 0:1]) * torch.cos(self.w * x[:, 1:2])
        j1 = -torch.cos(self.w * x[:, 0:1]) * torch.sin(self.w * x[:, 1:2])
        return y, j0, j1

class Burgers2D(_Problem):
    """Solves the time-dependent 1D viscous Burgers equation
        du       du        d^2 u
        -- + u * -- = nu * -----
        dt       dx        dx^2

        for -1.0 < x < +1.0, and 0 < t

        Boundary conditions:
        u(x,0) = - sin(pi*x)
        u(-1,t) = u(+1,t) = 0
    """

    @property
    def name(self):
        return "Burgers2D_nu%.3f" % (self.nu)

    def __init__(self, nu=0.01 / np.pi):
        # input params
        self.nu = nu

        # dimensionality of x and y
        self.d = (2, 1)

    def physics_loss(self, x, y, j0, j1, jj0):
        physics = (j1[:, 0] + y[:, 0] * j0[:, 0]) - (self.nu * jj0[:,
                                                               0])  # be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):
        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1 = j[:, 0:1], j[:, 1:2]
        jj = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0]
        jj0 = jj[:, 0:1]

        return y, j0, j1, jj0

    def boundary_condition(self, x, y, j0, j1, jj0, sd):
        # Apply u = tanh((x+1)/sd)*tanh((x-1)/sd)*tanh((y-0)/sd)*NN - sin(pi*x)   ansatz

        t0, jt0, jjt0 = boundary_conditions.tanhtanh_2(x[:, 0:1], -1, 1, sd)
        t1, jt1 = boundary_conditions.tanh_1(x[:, 1:2], 0, sd)

        sin = -torch.sin(np.pi * x[:, 0:1])
        cos = -np.pi * torch.cos(np.pi * x[:, 0:1])
        sin2 = (np.pi ** 2) * torch.sin(np.pi * x[:, 0:1])

        y_new = t0 * t1 * y + sin
        j0_new = jt0 * t1 * y + t0 * t1 * j0 + cos
        j1_new = t0 * jt1 * y + t0 * t1 * j1
        jj0_new = jjt0 * t1 * y + 2 * jt0 * t1 * j0 + t0 * t1 * jj0 + sin2

        return y_new, j0_new, j1_new, jj0_new

    def exact_solution(self, x, batch_size):
        # use the burgers_solution code to compute analytical solution
        xmin, xmax = x[:, 0].min().item(), x[:, 0].max().item()
        tmin, tmax = x[:, 1].min().item(), x[:, 1].max().item()
        vx = np.linspace(xmin, xmax, batch_size[0])
        vt = np.linspace(tmin, tmax, batch_size[1])
        vu = burgers_viscous_time_exact1(self.nu, len(vx), vx, len(vt), vt)
        y = torch.tensor(vu.flatten(), device=x.device).unsqueeze(1)

        return y, y, y, y  # skip computing analytical gradients

class FDTD1D(_Problem):
        """Solves the FDTD 1D equation
            u = [Hy, Ez]
            d Hy    dEz
            ---- - ----  =  0
            dt       dx
            d Ez    1   dHy
            ---- - ---  ---  =  0
            dt      c    dx

            Boundary conditions:
            Ez(x,0) = exp( -(1/2)((x/sd)^2) )
            Hy(x,0) = -exp( -(1/2)((x-dt/sd)^2) )
            du
            --(x,0) = 0
            dt
        """

        @property
        # def name(self):
        #     return "FDTD1D_%s" % (self.source_sd)
        # def __init__(self,source_sd=0.2):
        #     self.source_sd = source_sd
        #     # dimensionality of x and y
        #     self.d = (2, 2)
        def name(self):
            return "FDTD1D_%s" % (self._cname)

        def __init__(self, c=1, source_sd = 0.5):

            # input params
            if isinstance(c, (float, int)):
                self.c, self._c0, self._cname = self._constant_c, c, "s%sc%s" % (source_sd, c)
            elif c == "gaussian":
                self.c, self._c0, self._cname = self._gaussian_c, 1, "s%sc%s" % (source_sd, c)
            else:
                raise Exception("ERROR: c input not recognised! %s" % (c))
            self.source_sd = source_sd

            # dimensionality of x and y
            self.d = (2, 2)

        def physics_loss(self, x, y, j0_hy, j2_hy, j0_ez, j2_ez):
            Hy = y[:, 0:1]
            Ez = y[:, 1:2]
            physics_hy = (j0_hy[:, 0]) - j2_ez[:,
                                         0]   # be careful to slice correctly (transposed calculations otherwise (!))
            physics_ez = (j0_ez[:, 0]) - j2_hy[:,
                                         0]  # be careful to slice correctly (transposed calculations otherwise (!))
            return losses.l2_loss(physics_hy, 0) + losses.l2_loss(physics_ez, 0)

        def get_gradients(self, x, y):
            j_hy = torch.autograd.grad(y[:, 0:1], x, torch.ones_like(y[:, 0:1]), create_graph=True)[0]
            j_ez = torch.autograd.grad(y[:, 1:2], x, torch.ones_like(y[:, 1:2]), create_graph=True)[0]
            j0_hy, j2_hy = j_hy[:, 0:1], j_hy[:, 1:2]
            j0_ez, j2_ez = j_ez[:, 0:1], j_ez[:, 1:2]
            return y, j0_hy, j2_hy, j0_ez, j2_ez

        def boundary_condition(self, x, y, j0_hy, j2_hy, j0_ez, j2_ez, sd):

            s0 = self.source_sd
            t, jt, jtt = boundary_conditions.tanh2_2(x[:, 1:2],0,(sd / 2.5))
            t1, jt1, jtt1 = boundary_conditions.exp_decay(x[:, 1:2], (sd / 1.5))
            s, js, jss = boundary_conditions.exp_decay(x[:, 0:1], s0)
            y_new = y.clone().detach()

            y_new[:, 0:1] = t * y[:, 0:1] - t1 * s #磁场值更新
            y_new[:, 1:2] = t * y[:, 1:2] + t1 * s  # 电场值更新
            j0_hy_new = t * j0_hy - t1 * js
            j0_ez_new = t * j0_ez + t1 * js
            j2_hy_new = jt * y[:, 0:1] + t * j2_hy - s * jt1
            j2_ez_new = jt * y[:, 1:2] + t * j2_ez + s * jt1
            return y_new, j0_hy_new, j2_hy_new, j0_ez_new, j2_ez_new

        # t2, jt2, jjt2 = boundary_conditions.tanh2_2(x[:, 1:2], 0, sd)
        # s, js, jjs = boundary_conditions.exp_decay(x[:, 1:2],  sd)
        # # m0 = 0
        # s0 = self.source_sd
        # # xn0 = (x[:,0:1] - m0) / s0
        # # exp = torch.exp(-0.5 * (xn0 ** 2) )
        # # f = exp
        # s1, js1, jjs1 = boundary_conditions.exp_decay(x[:, 0:1], s0)
        # y_new = y.clone().detach()
        # y_new[:, 1:2] = t2 * y[:, 1:2] + s1 * s  # 电场值更新
        # y_new[:, 0:1] = t2 * y[:, 0:1] - s1 * s #磁场值更新
        # j0_hy_new = t2 * j0_hy - s * js1
        # j2_hy_new = jt2 * y[:, 0:1] + t2 * j2_hy - js * s1
        # j0_ez_new = t2 * j0_ez + s * js1
        # j2_ez_new = jt2 * y[:, 1:2] + t2 * j2_ez + js * s1
        # def boundary_condition(self, x, y, j0_hy, j2_hy, j0_ez, j2_ez, sd):
        #
        #     source = torch.tensor(self.source_sd, device=x.device)
        #     x, t = x[:, 0:1], x[:, 1:2]
        #     # get starting wavefield
        #     p = source.unsqueeze(1)  # (k, 1, 3)
        #     x = x.unsqueeze(0)  # (1, n, 1)
        #     f1 = (p[:, :, 2:3] * torch.exp(-0.5 * ((x - p[:, :, 0:1]) ** 2).sum(2, keepdims=True) / (p[:, :, 1:2] ** 2))).sum(
        #         0)  # (n, 1)
        #
        #     t1 = source[:,1].min()/self._c0
        #     xn0 = 1.5 * t / t1
        #     exp = torch.exp(-0.5 * (xn0 ** 2) )
        #     f =  exp * f1
        #
        #     xn01 = 2.5*t
        #     t, jt, jtt = boundary_conditions.tanh2_2(xn01,0,t1)
        #     y_new = y.clone().detach()
        #     y_new[:, 1:2] = t * y[:, 1:2] + f  # 电场值更新
        #     y_new[:, 0:1] = t * y[:, 0:1] - f #磁场值更新
        #     return y_new, y_new, y_new, y_new, y_new

        # def boundary_condition(self, x, y, j0_hy, j2_hy, j0_ez, j2_ez, sd):
        #
        #     # Apply u = tanh^2((t-0)/sd)*NN + sigmoid((d-t)/sd)*exp( -(1/2)((x/sd)^2) )  ansatz
        #
        #     t2, jt2, jjt2 = boundary_conditions.tanh2_2(x[:, 1:2], 0, sd)
        #     s, js, jjs = boundary_conditions.sigmoid_2(-x[:, 1:2], -2 * sd,
        #                                                0.2 * sd)  # beware (!) this gives correct 2nd order gradients but negative 1st order (sign flip!)
        #     m0 = 0;
        #     # s0 = self.source[:,1]
        #     xn0 = (x[:, 0:1] - m0) / sd
        #     exp = torch.exp(-0.5 * (xn0 ** 2))
        #     f = exp
        #     y_new = y.clone().detach()
        #     y_new[:, 1:2] = t2 * y[:, 1:2] + s * f #电场值更新
        #     y_new[:, 0:1] = t2 * y[:, 0:1] - s * f #磁场值更新
        #     j0_hy_new = t2 * j0_hy - s * f * (-xn0 / sd)
        #     j2_hy_new = jt2 * y[:, 0:1] + t2 * j2_hy - js * f
        #     j0_ez_new = t2 * j0_ez + s * f * (-xn0 / sd)
        #     j2_ez_new = jt2 * y[:, 1:2] + t2 * j2_ez + js * f
        #     return y_new, j0_hy_new, j2_hy_new, j0_ez_new, j2_ez_new

        def exact_solution(self, x, batch_size):
            c, c0, source_sd = self.c, self._c0, self.source_sd
            xmin, xmax = x[:, 0].min().item(), x[:, 0].max().item()
            tmin, tmax = x[:, 1].min().item(), x[:, 1].max().item()
            deltax, deltat = (xmax - xmin) / (batch_size[0] - 1), (tmax - tmin) / (batch_size[1] - 1)
            f0 = c0 / source_sd  # approximate frequency of wave
            DELTAX = 1 / (f0 * 10)  # target fine sampled deltas
            DELTAT = DELTAX / (4 * np.sqrt(2))  # target fine sampled deltas
            dx, dt = int(np.ceil(deltax / DELTAX)), int(
                np.ceil(deltat / DELTAT))  # make sure deltas are a multiple of test deltas
            DELTAX, DELTAT = deltax / dx, deltat / dt
            NX, NSTEPS = batch_size[0] * dx - (dx - 1), batch_size[1] * dt - (dt - 1)
            Hy, Ex = FDTD1DD(
                xmin,
                xmax,
                NX,
                NSTEPS,
                DELTAX,
                DELTAT,
            )
            Hy = Hy[::dx, ::dt]
            Hy = torch.tensor(Hy.flatten(), device=x.device)
            Hy = Hy.view(-1, 1)
            Ex = Ex[::dx, ::dt]
            Ex = torch.tensor(Ex.flatten(), device=x.device)
            Ex = Ex.view(-1, 1)
            # 拼接 Hy 和 Ex，沿着列方向（dim=1）进行拼接
            y = torch.cat((Hy, Ex), dim=1)
            return y, y, y, y, y  # skip computing analytical gradients

        def _gaussian(self, x, mu, sd, a):
            return a * torch.exp(-0.5 * (((x[:, 0:1] - mu[0]) / sd[0]) ** 2))

        def _gaussian_c(self, x):
            "Defines a hard-coded mixture of gaussians velocity model over [-10,10]"

            mus = np.array([[3],
                            [-5],
                            [-2],
                            [3]])
            sds = np.array([[3],
                            [4],
                            [2],
                            [3]])
            aas = self._c0 * np.array([-0.7, -0.6, 0.7, 0.6])

            cs = []
            for mu, sd, a in zip(mus, sds, aas):
                cs.append(self._gaussian(x, mu, sd, a))
            c = self._c0 + torch.sum(torch.stack(cs, -1), -1)

            return c

        def _constant_c(self, x):
            "Defines a constant velocity model"

            return self._c0 * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
#################################################################################################################################
class WaveEquation2D(_Problem):
    """Solves the time-dependent 2D wave equation
        d^2 u     1  d^2 u
        -----  - --- ----- = 0
        dx^2     c^2 dt^2

        Boundary conditions:
        u(x,0) = exp( -(1/2)((x/sd)^2)
        du
        --(x,0) = 0
        dt
    """

    @property
    def name(self):
        return "WaveEquation2D_%s" % (self._cname)

    def __init__(self, c=1, source_sd=0.2):

        # input params
        if isinstance(c, (float, int)):
            self.c, self._c0, self._cname = self._constant_c, c, "s%sc%s" % (source_sd, c)
        elif c == "gaussian":
            self.c, self._c0, self._cname = self._gaussian_c, 1, "s%sc%s" % (source_sd, c)
        else:
            raise Exception("ERROR: c input not recognised! %s" % (c))
        self.source_sd = source_sd

        # dimensionality of x and y
        self.d = (2, 2)

    def physics_loss(self, x, y, j2,jj0, jj2):

        physics = jj0[:, 0] - (1 / (self.c(x)[:, 0] ** 2)) * jj2[:,0]  # be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):

        # j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j = torch.autograd.grad(y[:, 0:1], x, torch.ones_like(y[:, 0:1]), create_graph=True)[0]
        j0, j2 = j[:, 0:1], j[:, 1:2]
        jj = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0]
        jj0 = jj[:, 0:1]
        jj2 = jj[:, 1:2]

        return y,j2, jj0, jj2

    def boundary_condition(self, x, y, j2, jj0, jj2, sd):

        t2, jt2, jjt2 = boundary_conditions.tanh2_2(x[:, 1:2], 0, sd)
        s, _, jjs = boundary_conditions.sigmoid_2(-x[:, 1:2], -2 * sd,
                                                  0.2 * sd)  # beware (!) this gives correct 2nd order gradients but negative 1st order (sign flip!)

        m0 = 0;
        s0 = self.source_sd
        xn0 = (x[:, 0:1] - m0) / s0
        exp = torch.exp(-0.5 * (xn0 ** 2 ))
        f = exp
        jjf0 = (1 / s0 ** 2) * ((xn0 ** 2) - 1) * exp

        y_new = t2 * y + s * f
        jj0_new = t2 * jj0 + s * jjf0
        jj2_new = jjt2 * y + 2 * jt2 * j2 + t2 * jj2 + jjs * f

        return y_new, j2, jj0_new,  jj2_new  # skip updating first order gradients (not needed for loss)

    def exact_solution(self, x, batch_size):
        # use the burgers_solution code to compute analytical solution
        xmin, xmax = x[:, 0].min().item(), x[:, 0].max().item()
        tmin, tmax = x[:, 1].min().item(), x[:, 1].max().item()
        vx = np.linspace(xmin, xmax, batch_size[0])
        vt = np.linspace(tmin, tmax, batch_size[1])
        vu = burgers_viscous_time_exact1(self.nu, len(vx), vx, len(vt), vt)
        y = torch.tensor(vu.flatten(), device=x.device).unsqueeze(1)

        return y, y, y, y  # skip computing analytical gradients

    def _gaussian(self, x, mu, sd, a):
        return a * torch.exp(-0.5 * (((x[:, 0:1] - mu[0]) / sd[0]) ** 2))

    def _gaussian_c(self, x):
        "Defines a hard-coded mixture of gaussians velocity model over [-10,10]"

        mus = np.array([[3],
                        [-5],
                        [-2],
                        [3]])
        sds = np.array([[3],
                        [4],
                        [2],
                        [3]])
        aas = self._c0 * np.array([-0.7, -0.6, 0.7, 0.6])

        cs = []
        for mu, sd, a in zip(mus, sds, aas):
            cs.append(self._gaussian(x, mu, sd, a))
        c = self._c0 + torch.sum(torch.stack(cs, -1), -1)

        return c

    def _constant_c(self, x):
        "Defines a constant velocity model"

        return self._c0 * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
class WaveEquation3D(_Problem):
    """Solves the time-dependent 2D wave equation
        d^2 u   d^2 u    1  d^2 u
        ----- + ----- - --- ----- = 0
        dx^2    dy^2    c^2 dt^2

        Boundary conditions:
        u(x,y,0) = exp( -(1/2)((x/sd)^2+(y/sd)^2) )
        du
        --(x,y,0) = 0
        dt
    """

    @property
    def name(self):
        return "WaveEquation3D_%s" % (self._cname)

    def __init__(self, c=1, source_sd=0.2):

        # input params
        if isinstance(c, (float, int)):
            self.c, self._c0, self._cname = self._constant_c, c, "s%sc%s" % (source_sd, c)
        elif c == "gaussian":
            self.c, self._c0, self._cname = self._gaussian_c, 1, "s%sc%s" % (source_sd, c)
        else:
            raise Exception("ERROR: c input not recognised! %s" % (c))
        self.source_sd = source_sd

        # dimensionality of x and y
        self.d = (3, 1)

    def physics_loss(self, x, y, j2, jj0, jj1, jj2):

        physics = (jj0[:, 0] + jj1[:, 0]) - (1 / (self.c(x)[:, 0] ** 2)) * jj2[:,0]  # be careful to slice correctly (transposed calculations otherwise (!))
        return losses.l2_loss(physics, 0)

    def get_gradients(self, x, y):

        j = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        j0, j1, j2 = j[:, 0:1], j[:, 1:2], j[:, 2:3]
        jj0 = torch.autograd.grad(j0, x, torch.ones_like(j0), create_graph=True)[0][:, 0:1]
        jj1 = torch.autograd.grad(j1, x, torch.ones_like(j1), create_graph=True)[0][:, 1:2]
        jj2 = torch.autograd.grad(j2, x, torch.ones_like(j2), create_graph=True)[0][:, 2:3]

        return y, j2, jj0, jj1, jj2

    def boundary_condition(self, x, y, j2, jj0, jj1, jj2, sd):

        # Apply u = tanh^2((t-0)/sd)*NN + sigmoid((d-t)/sd)*exp( -(1/2)((x/sd)^2+(y/sd)^2) )  ansatz

        t2, jt2, jjt2 = boundary_conditions.tanh2_2(x[:, 2:3], 0, sd)
        s, _, jjs = boundary_conditions.sigmoid_2(-x[:, 2:3], -2 * sd,
                                                  0.2 * sd)  # beware (!) this gives correct 2nd order gradients but negative 1st order (sign flip!)

        m0 = m1 = 0;
        s0 = s1 = self.source_sd
        xn0, xn1 = (x[:, 0:1] - m0) / s0, (x[:, 1:2] - m1) / s1
        exp = torch.exp(-0.5 * (xn0 ** 2 + xn1 ** 2))
        f = exp
        jjf0 = (1 / s0 ** 2) * ((xn0 ** 2) - 1) * exp
        jjf1 = (1 / s1 ** 2) * ((xn1 ** 2) - 1) * exp

        y_new = t2 * y + s * f
        jj0_new = t2 * jj0 + s * jjf0
        jj1_new = t2 * jj1 + s * jjf1
        jj2_new = jjt2 * y + 2 * jt2 * j2 + t2 * jj2 + jjs * f

        return y_new, j2, jj0_new, jj1_new, jj2_new  # skip updating first order gradients (not needed for loss)

    def exact_solution(self, x, batch_size):

        # use the seismicCPML2D FD code with very fine sampling to compute solution

        c, c0, source_sd = self.c, self._c0, self.source_sd
        xmin, xmax = x[:, 0].min().item(), x[:, 0].max().item()
        ymin, ymax = x[:, 1].min().item(), x[:, 1].max().item()
        tmin, tmax = x[:, 2].min().item(), x[:, 2].max().item()
        deltax, deltay, deltat = (xmax - xmin) / (batch_size[0] - 1), (ymax - ymin) / (batch_size[1] - 1), (
                    tmax - tmin) / (batch_size[2] - 1)

        # get f0, target deltas of FD simulation
        f0 = c0 / source_sd  # approximate frequency of wave
        DELTAX = DELTAY = 1 / (f0 * 10)  # target fine sampled deltas
        DELTAT = DELTAX / (4 * np.sqrt(2) * c0)  # target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax / DELTAX)), int(np.ceil(deltay / DELTAY)), int(
            np.ceil(deltat / DELTAT))  # make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax / dx, deltay / dy, deltat / dt
        NX, NY, NSTEPS = batch_size[0] * dx - (dx - 1), batch_size[1] * dy - (dy - 1), batch_size[2] * dt - (dt - 1)

        # get starting wavefield and velocity model
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, NX), np.linspace(ymin, ymax, NY), indexing="ij")
        p0 = np.exp(-0.5 * ((xx / source_sd) ** 2 + (yy / source_sd) ** 2))
        c = c(torch.from_numpy(np.stack([xx, yy], -1).reshape((NX * NY, 2)))).numpy().reshape(NX, NY)
        ##np.linspace(xmin, xmax, NX) 和 np.linspace(ymin, ymax, NY) 分别在 x 和 y 轴上生成均匀分布的 NX 和 NY 个点。
        # np.meshgrid 接收这两个轴上的坐标向量，并生成一个网格矩阵。
        # indexing="ij" 表示使用索引顺序为 "ij"，即第一个索引对应行，第二个索引对应列。
        # xx 和 yy 是两个二维数组，分别包含了平面上均匀分布的 NX 行和 NY 列的坐标点。

        # add padded CPML boundary
        NPOINTS_PML = 10
        p0 = np.pad(p0, [(NPOINTS_PML, NPOINTS_PML), (NPOINTS_PML, NPOINTS_PML)], mode="edge")
        c = np.pad(c, [(NPOINTS_PML, NPOINTS_PML), (NPOINTS_PML, NPOINTS_PML)], mode="edge")

        # run simulation
        wavefields, _ = seismicCPML2D(
            NX + 2 * NPOINTS_PML,
            NY + 2 * NPOINTS_PML,
            NSTEPS,
            DELTAX,
            DELTAY,
            DELTAT,
            NPOINTS_PML,
            c,
            np.ones((NX + 2 * NPOINTS_PML, NY + 2 * NPOINTS_PML)),
            (p0.copy(), p0.copy()),
            f0,
            np.float32,
            output_wavefields=True,
            gather_is=None)

        # get croped, decimated, flattened wavefields
        wavefields = wavefields[:, NPOINTS_PML:-NPOINTS_PML, NPOINTS_PML:-NPOINTS_PML]
        wavefields = wavefields[::dt, ::dx, ::dy]
        wavefields = np.moveaxis(wavefields, 0, -1)
        assert wavefields.shape == batch_size
        y = torch.tensor(wavefields.flatten(), device=x.device).unsqueeze(1)
        #::dt、::dx 和 ::dy 表示以步长为 dt、dx 和 dy 对 wavefields 进行取样。这些值在先前的模拟设置中被计算出来，用于确定采样的间隔。
        # 这种采样方式相当于对 wavefields 进行了降采样，从而减少了数据量。
        return y, y, y, y, y  # skip computing analytical gradients

    def _gaussian(self, x, mu, sd, a):#_gaussian 函数计算了给定均值、标准差和振幅的高斯分布，并返回相应的值，该值添加到 cs 列表中。
        return a * torch.exp(-0.5 * (((x[:, 0:1] - mu[0]) / sd[0]) ** 2 + ((x[:, 1:2] - mu[1]) / sd[1]) ** 2))

    import torch

    def _gaussian_c(self, x):
        "Defines a hard-coded mixture of gaussians velocity model over [-10,10]"
        #定义了一个硬编码的混合高斯速度模型，范围在 [-10, 10] 内。
        #尝试几个新的c的，比如方形的，

        mus = np.array([[3, 3],
                        [-5, -5],
                        [-2, 2],
                        [3, -4]])
        #定义了高斯分布的均值矩阵。这里有四个高斯分布，每个分布有一个二维均值向量。
        sds = np.array([[3, 3],
                        [4, 4],
                        [2, 2],
                        [3, 3]])
        #sds：定义了高斯分布的标准差矩阵。与 mus 类似，每个分布都有一个二维标准差向量。
        aas = self._c0 * np.array([-0.7, -0.6, 0.7, 0.6])
        #定义了高斯分布的振幅。这个数组乘以 _c0，得到四个高斯分布的振幅值。

        cs = []#cs 是一个空列表，用于存储每个高斯分布在给定输入 x 下的值。
        for mu, sd, a in zip(mus, sds, aas):
            cs.append(self._gaussian(x, mu, sd, a))
            #for 循环遍历 mus，sds 和 aas 中的元素。
            # 对于每个高斯分布，它调用了 _gaussian 函数，将 x、对应的均值 mu、标准差 sd 和振幅 a 传递给这个函数。_gauss
        c = self._c0 + torch.sum(torch.stack(cs, -1), -1)
        #最后，计算了四个高斯分布在 x 处的总和，并将 _c0 加到结果上。这个总和即为混合高斯模型的值，并作为返回值返回。

        return c
    import torch
    def _rectangle_c(self, x):
        rows, cols = x.size()
        cs=torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        # cs = []
        for i in range(rows):
                if x[i][0] < 0 :
                    cs[i] = 1
                elif x[i][0] >= 0 :
                    cs[i] = 1.5
                # elif x[i][0] >= 0 and x[i][1] < 0:
                #     cs[i] = 1.5
                # elif x[i][0] >= 0 and x[i][1] >= 0:
                #     cs[i] = 2
        return self._c0*cs
    def _constant_c(self, x):
        "Defines a constant velocity model"
        return self._c0 * torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    import problems
    from main import _x_mesh

    # check velocity models for WaveEquation3D

    P = problems.WaveEquation3D(c=1, source_sd=0.2)
    subdomain_xs = [np.array([-10, -5, 0, 5, 10]), np.array([-10, -5, 0, 5, 10]), np.array([0, 5, 10])]
    batch_size_test = (50, 50, 15)
    x = _x_mesh(subdomain_xs, batch_size_test, "cpu")

    for f in P._rectangle_c, P._constant_c:
        y = f(x)
        print(y.shape)
        y = y[:, 0].numpy().reshape(batch_size_test)

        plt.figure()
        plt.imshow(y[:, :, 0].T, origin="lower")
        plt.colorbar()
        plt.figure()
        plt.imshow(y[:, :, -1].T, origin="lower")
        plt.colorbar()
        plt.show()