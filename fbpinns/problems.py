"""
Defines PDE problems to solve

Each problem class must inherit from the Problem base class.
Each problem class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax.nn
import jax.numpy as jnp
import numpy as np

from fbpinns.util.logger import logger
from fbpinns.traditional_solutions.analytical.burgers_solution import burgers_viscous_time_exact1
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_2D_pressure_second_order import seismicCPML2D
from FDTD import FDTD2D

class Problem:
    """Base problem class to be inherited by different problem classes.

    Note all methods in this class are jit compiled / used by JAX,
    so they must not include any side-effects!
    (A side-effect is any effect of a function that doesn’t appear in its output)
    This is why only static methods are defined.
    """

    # required methods

    @staticmethod
    def init_params(*args):
        """Initialise class parameters.
        Returns tuple of dicts ({k: pytree}, {k: pytree}) containing static and trainable parameters"""

        # below parameters need to be defined
        static_params = {
            "dims":None,# (ud, xd)# dimensionality of u and x
            }
        raise NotImplementedError

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes):
        """Samples all constraints.
        Returns [[x_batch, *any_constraining_values, required_ujs], ...]. Each list element contains
        the x_batch points and any constraining values passed to the loss function, and the required
        solution and gradient components required in the loss function, for each constraint."""
        raise NotImplementedError

    @staticmethod
    def constraining_fn(all_params, x_batch, u):
        """Applies optional constraining operator"""
        return u

    @staticmethod
    def loss_fn(all_params, constraints):
        """Computes the PINN loss function, using constraints with the same structure output by sample_constraints"""
        raise NotImplementedError

    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape=None):
        """Defines exact solution, if it exists"""
        raise NotImplementedError


class Maxwell2DTE(Problem):
    """Solves the time-dependent (1+1)D Maxwell equation with constant velocity

        u = [Hx, Hy, Ez]
        dHx     dEz
        ---- + -----  =  0
        dt      dy

        dHy     dEz
        ---- - -----  =  0
        dt      dx

        dEz     1    dHy    dHx
        ---- - -- ( ---- - ----)   =  0
        dt      Σ    dx     dy

        Boundary conditions:

    """

    @staticmethod
    def init_params(eps_bg=1.0, eps_obj=2.0, pulse_sd=0.1, alpha=110.0):
        static_params = {
            "dims": (3, 3),
            "eps_bg": eps_bg,
            "eps_obj": eps_obj,
            "pulse_sd": pulse_sd,
            "epsilon_fn": Maxwell2DTE.epsilon_fn,
            # 新增界面参数
            "interface": {
                "circle": {
                    "center": (-0.5, 0.5),
                    "radius": 0.45,
                },
                "triangle": {
                    "v1": (0.2, -0.4),
                    "v2": (0, -0.8),
                    "v3": (0.4, -0.8),
                },
                # 圆心坐标（与epsilon_fn一致）
                "alpha": alpha,
            }
        }
        return static_params, {}

    @staticmethod
    def sample_constraints(all_params, domain, key, sampler, batch_shapes, start_batch_shapes, boundary_batch_shapes):
        params = all_params["static"]["problem"]
        pulse_sd = params["pulse_sd"]
        # physics loss
        x_batch_phys = domain.sample_interior(all_params, key, sampler, batch_shapes[0])
        required_ujs_phys = (
            (0, (1,)),  # dHx / dy
            (0, (2,)),  # dHx / dt
            (1, (0,)),  # dHy / dx
            (1, (2,)),  # dHy / dt
            (2, (0,)),  # dE / dx
            (2, (1,)),  # dE / dy
            (2, (2,)),  # dE / dt
        )
        # start loss
        x_batch_start = domain.sample_start(all_params, key, "grid", start_batch_shapes[0])
        x = x_batch_start[:, 0:1] # 提取 x 坐标
        y = x_batch_start[:, 1:2]
        # 计算E_start
        E_start = jnp.exp(-0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / (pulse_sd ** 2))
        # 对x求偏导，结果保存在Hx_start
        Hx_start = (y - 0.5) / (pulse_sd ** 2) * E_start
        # 对y求偏导，结果保存在Hy_start
        Hy_start = -(x - 0.5) / (pulse_sd ** 2) * E_start

        required_ujs_start = (
            (0, ()),
            (1, ()),
            (2, ()),
        )
        # boundary loss
        x_batch_boundary = domain.sample_boundaries(all_params, key, sampler, boundary_batch_shapes[0])
        required_ujs_boundary = (
            (0, (1,)),  # dHx / dy
            (0, (2,)),  # dHx / dt
            (1, (0,)),  # dHy / dx
            (1, (2,)),  # dHy / dt
            (2, (0,)),  # dE / dx
            (2, (1,)),  # dE / dy
            (2, (2,)),  # dE / dt
        )
        return [[x_batch_phys, required_ujs_phys], [x_batch_start, Hx_start, Hy_start, E_start, required_ujs_start],
                [x_batch_boundary, required_ujs_boundary]]
    @staticmethod
    def loss_fn(all_params, constraints):

        epsilon_fn = all_params["static"]["problem"]["epsilon_fn"]
        # physics loss
        x_batch, dHxdy, dHxdt, dHydx, dHydt, dEdx, dEdy, dEdt = constraints[0]

        phys1 = jnp.mean((dHxdt + dEdy) ** 2)
        phys2 = jnp.mean((dHydt - dEdx) ** 2)
        phys3 = jnp.mean((dEdt - (1 / epsilon_fn(all_params, x_batch)) * (dHydx - dHxdy)) ** 2)
        phys = phys1 + phys2 + phys3

        # start loss
        x_batch_start, Hxc, Hyc, Ec, Hx, Hy, E = constraints[1]
        if len(Ec):
            start = jnp.mean((E - Ec) ** 2) + jnp.mean((Hx - Hxc) ** 2) + jnp.mean((Hy - Hyc) ** 2)
        else:
            start = 0

        # boundary loss
        x_batch_boundary, dHxdy_boundary, dHxdt_boundary, dHydx_boundary, dHydt_boundary, dEdx_boundary, dEdy_boundary, dEdt_boundary = \
            constraints[2]

        boundary1 = jnp.mean((dHxdt_boundary + dEdy_boundary) ** 2)
        boundary2 = jnp.mean((dHydt_boundary - dEdx_boundary) ** 2)
        boundary3 = jnp.mean(
            (dEdt_boundary - (1 / epsilon_fn(all_params, x_batch_boundary)) * (dHydx_boundary - dHxdy_boundary)) ** 2)
        boundary = boundary1 + boundary2 + boundary3

        return 1e1 * phys + 1e2 * start + 1e1 * boundary
    @staticmethod
    def exact_solution(all_params, x_batch, batch_shape):
        params = all_params["static"]["problem"]
        pulse_sd = params["pulse_sd"]
        epsilon_fn = params["epsilon_fn"]

        (xmin, ymin, tmin), (xmax, ymax, tmax) = np.array(x_batch.min(0)), np.array(x_batch.max(0))
        # get grid spacing
        deltax, deltay, deltat = (xmax - xmin) / (batch_shape[0] - 1), (ymax - ymin) / (batch_shape[1] - 1), (
                    tmax - tmin) / (batch_shape[2] - 1)

        # get f0, target deltas of FD simulation
        f0 = 1 / pulse_sd  # approximate frequency of wave
        DELTAX = DELTAY = 1 / (f0 * 10)  # target fine sampled deltas
        DELTAT = DELTAX / (4 * np.sqrt(2) * 1)  # target fine sampled deltas
        dx, dy, dt = int(np.ceil(deltax / DELTAX)), int(np.ceil(deltay / DELTAY)), int(
            np.ceil(deltat / DELTAT))  # make sure deltas are a multiple of test deltas
        DELTAX, DELTAY, DELTAT = deltax / dx, deltay / dy, deltat / dt
        NX, NY, NSTEPS = batch_shape[0] * dx - (dx - 1), batch_shape[1] * dy - (dy - 1), batch_shape[2] * dt - (dt - 1)

        xx, yy = np.meshgrid(np.linspace(2 * xmin, 2 * xmax, 2 * NX), np.linspace(2 * ymin, 2 * ymax, 2 * NY),
                             indexing="ij")

        # get velocity model
        x = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n, 2)
        velocity = np.array(epsilon_fn(all_params, x))
        if velocity.shape[0] > 1:
            velocity = velocity.reshape((2 * NX, 2 * NY))
        else:
            velocity = velocity * np.ones_like(xx)
        Ez = FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, pulse_sd, velocity, )
        Ez = Ez[::dx, ::dy, ::dt]
        Ez = jnp.ravel(Ez)
        Ez = jnp.reshape(Ez, (-1, 1))

        # 拼接 Hy 和 Ex，沿着列方向（dim=1）进行拼接
        return Ez

    @staticmethod
    def epsilon_fn(all_params, x_batch):
        # 提取参数
        ebs_bg = all_params["static"]["problem"]["eps_bg"]
        ebs_obj = all_params["static"]["problem"]["eps_obj"]
        interface_params = all_params["static"]["problem"]["interface"]
        circle_params = interface_params["circle"]
        triangle_params = interface_params["triangle"]

        x0, y0 = circle_params["center"]
        r = circle_params["radius"]
        alpha = interface_params["alpha"]

        # 新增三角形参数
        triangle_vertices = triangle_params["v1"],triangle_params["v2"],triangle_params["v3"]

        # 参数解析
        x = x_batch[:, 0]
        y = x_batch[:, 1]

        # 定义圆形的 level set 函数
        def level_set_function_circle(x, y):
            return r - jnp.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # 与圆心的距离减去半径

            # 定义三角形的 level set 函数
            # 定义三角形的 level set 函数

        def level_set_function_triangle(x, y, vertices):
            # 三角形的三个顶点
            (x1, y1), (x2, y2), (x3, y3) = vertices

            # 计算点到三角形的最小距离
            def point_to_segment_distance(px, py, x1, y1, x2, y2):
                # 计算点到线段的最小距离
                dx, dy = x2 - x1, y2 - y1
                t = ((px - x1) * dx + (py - y1) * dy) / (dx ** 2 + dy ** 2 + 1e-10)
                t = jnp.clip(t, 0.0, 1.0)
                closest_x = x1 + t * dx
                closest_y = y1 + t * dy
                return jnp.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

            # 计算点到三角形的最小距离
            d1 = point_to_segment_distance(x, y, x1, y1, x2, y2)
            d2 = point_to_segment_distance(x, y, x2, y2, x3, y3)
            d3 = point_to_segment_distance(x, y, x3, y3, x1, y1)
            min_distance = jnp.minimum(jnp.minimum(d1, d2), d3)

            # 判断点是否在三角形内部
            def cross_product(a, b):
                return a[0] * b[1] - a[1] * b[0]

            def point_in_triangle(px, py):
                # 向量 AB, BC, CA
                AB = (x2 - x1, y2 - y1)
                BC = (x3 - x2, y3 - y2)
                CA = (x1 - x3, y1 - y3)

                # 向量 AP, BP, CP
                AP = (px - x1, py - y1)
                BP = (px - x2, py - y2)
                CP = (px - x3, py - y3)

                # 计算叉积
                cross0 = cross_product(AB, AP)
                cross1 = cross_product(BC, BP)
                cross2 = cross_product(CA, CP)

                # 判断点是否在三角形内部
                inside = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0) | (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0)
                return ~inside

            # 有符号距离函数
            sign = jnp.where(point_in_triangle(x, y), -1.0, 1.0)
            return sign * min_distance

            # 计算圆形和三角形的 level set 函数

        F_circle = level_set_function_circle(x, y)
        F_triangle = level_set_function_triangle(x, y, triangle_vertices)

        # 计算 Heaviside 函数
        H_hat_circle = 1 / (1 + jnp.exp(-alpha * F_circle))
        H_hat_triangle = 1 / (1 + jnp.exp(-alpha * F_triangle))

        # 调整逻辑：如果三角形内的点使用 ebs_obj，否则使用 ebs_bg
        epsilon = ebs_bg * (1 - jnp.maximum(H_hat_circle, H_hat_triangle)) + ebs_obj * jnp.maximum(H_hat_circle, H_hat_triangle)

        # 返回 epsilon 值
        return jnp.expand_dims(epsilon, axis=1)

    # @staticmethod
    # def epsilon_fn(all_params, x_batch):
    #     # 提取参数
    #     interface_params = all_params["static"]["problem"]["interface"]
    #     circle_params = interface_params["circle"]
    #     triangle_params = interface_params["triangle"]
    #
    #     # 圆形的参数
    #     x0, y0 = circle_params["center"]
    #     r = circle_params["radius"]
    #
    #     # 三角形的三个顶点
    #     triangle_vertices = [
    #         triangle_params["v1"],
    #         triangle_params["v2"],
    #         triangle_params["v3"]
    #     ]
    #
    #     # 参数解析
    #     x = x_batch[:, 0]
    #     y = x_batch[:, 1]
    #
    #     # 初始化介电常数，默认值为 1
    #     epsilon = jnp.ones_like(x)
    #
    #     # 判断点是否在圆形内部
    #     def point_in_circle(px, py, x0, y0, radius):
    #         distance = jnp.sqrt((px - x0) ** 2 + (py - y0) ** 2)
    #         return distance <= radius
    #
    #     # 判断点是否在三角形内部
    #     def point_in_triangle(px, py, vertices):
    #         # 三角形的三个顶点
    #         (x1, y1), (x2, y2), (x3, y3) = vertices
    #
    #         # 计算叉积
    #         def cross_product(a, b):
    #             return a[0] * b[1] - a[1] * b[0]
    #
    #         # 向量 AB, BC, CA
    #         AB = (x2 - x1, y2 - y1)
    #         BC = (x3 - x2, y3 - y2)
    #         CA = (x1 - x3, y1 - y3)
    #
    #         # 向量 AP, BP, CP
    #         AP = (px - x1, py - y1)
    #         BP = (px - x2, py - y2)
    #         CP = (px - x3, py - y3)
    #
    #         # 计算叉积
    #         cross0 = cross_product(AB, AP)
    #         cross1 = cross_product(BC, BP)
    #         cross2 = cross_product(CA, CP)
    #
    #         # 判断点是否在三角形内部
    #         inside = (cross0 >= 0) & (cross1 >= 0) & (cross2 >= 0) | (cross0 <= 0) & (cross1 <= 0) & (cross2 <= 0)
    #         return inside
    #
    #     # 判断每个点是否在圆形内部
    #     in_circle = point_in_circle(x, y, x0, y0, r)
    #
    #     # 判断每个点是否在三角形内部
    #     in_triangle = point_in_triangle(x, y, triangle_vertices)
    #
    #     # 设置圆形和三角形内部的介电常数为 2
    #     epsilon = jnp.where(in_circle | in_triangle, 2.0, epsilon)
    #
    #     # 将 epsilon 重新调整为预期的输出形状 (n, 1)
    #     epsilon = jnp.expand_dims(epsilon, axis=1)
    #
    #     return epsilon



