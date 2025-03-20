"""
Defines problem domains

Each domain class must inherit from the Domain base class.
Each domain class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

from fbpinns import networks


class Domain:
    """Base domain class to be inherited by different domain classes.

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
            "xd":None,# dimensionality of x
            }
        raise NotImplementedError

    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):
        """Samples interior of domain.
        Returns x_batch points in interior of domain"""
        raise NotImplementedError

    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        """Samples boundaries of domain.
        Returns (x_batch, ...) tuple of points for each boundary"""
        raise NotImplementedError

    @staticmethod
    def norm_fn(all_params, x):
        """"Applies norm function, for a SINGLE point with shape (xd,)"""# note only used for PINNs, FBPINN norm function defined in Decomposition
        raise NotImplementedError




class RectangularDomainND(Domain):

    @staticmethod
    def init_params(xmin, xmax):

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)

        static_params = {
            "xd":xd,
            "xmin":jnp.array(xmin),
            "xmax":jnp.array(xmax),
            }
        return static_params, {}

    @staticmethod
    def sample_interior(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler_start(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        """
        在xyt三维空间中生成圆形和三角形边界附近的采样点
        batch_shapes: (x_samples, y_samples, t_samples) 各维度采样数
        """
        # 解析域参数
        domain = all_params["static"]["domain"]
        xmin, xmax = domain["xmin"], domain["xmax"]
        xd = domain["xd"]
        assert xd == 3, "当前仅支持xyt三维空间"

        # 硬编码圆形参数
        CIRCLE_CENTER = (-0.5, 0.5)  # 圆心 (x,y)
        CIRCLE_RADIUS = 0.45  # 半径
        DELTA = 0.05  # 边界扰动范围
        TIME_RANGE = (0.0, 1.5)  # 时间范围

        # 硬编码三角形参数
        triangle_vertices = [(0.2, -0.4), (0, -0.8), (0.4, -0.8)]  # 三角形的三个顶点

        # 解析采样数量
        x_samples, y_samples, t_samples = batch_shapes

        # ==================== 生成圆形边界采样 ====================
        # 1. 生成极坐标扰动
        key, angle_key, radius_key, t_key = jax.random.split(key, 4)

        # 角度采样 (x_samples*y_samples个点)
        angles = jax.random.uniform(
            angle_key, (x_samples * y_samples,),
            minval=0,
            maxval=2 * jnp.pi
        )

        # 半径扰动 (在半径±DELTA范围内)
        radii = CIRCLE_RADIUS + jax.random.uniform(
            radius_key, (x_samples * y_samples,),
            minval=-DELTA,
            maxval=DELTA
        )

        # 2. 转换为笛卡尔坐标
        x_circle = CIRCLE_CENTER[0] + radii * jnp.cos(angles)
        y_circle = CIRCLE_CENTER[1] + radii * jnp.sin(angles)

        # 3. 生成时间采样 (每个空间点对应t_samples个时间点)
        t_circle = jax.random.uniform(
            t_key, (x_samples * y_samples, t_samples),
            minval=TIME_RANGE[0],
            maxval=TIME_RANGE[1]
        )

        # 4. 组合三维坐标
        x_circle = jnp.repeat(x_circle, t_samples)
        y_circle = jnp.repeat(y_circle, t_samples)
        t_circle = t_circle.reshape(-1)
        circle_points = jnp.stack([x_circle, y_circle, t_circle], axis=1)

        # ==================== 生成三角形边界采样 ====================
        # 1. 生成三角形边界的采样点
        key, edge_key, normal_key, t_key = jax.random.split(key, 4)

        # 三角形的三条边
        edges = [
            (triangle_vertices[0], triangle_vertices[1]),
            (triangle_vertices[1], triangle_vertices[2]),
            (triangle_vertices[2], triangle_vertices[0])
        ]

        # 每条边的采样点数量
        edge_samples = x_samples * y_samples // 3

        # 初始化三角形边界采样点
        x_triangle = jnp.array([])
        y_triangle = jnp.array([])

        for (x1, y1), (x2, y2) in edges:
            # 在边上生成均匀分布的采样点
            t_edge = jax.random.uniform(
                edge_key, (edge_samples,),
                minval=0,
                maxval=1
            )
            x_edge = x1 + t_edge * (x2 - x1)
            y_edge = y1 + t_edge * (y2 - y1)

            # 对采样点施加法线方向的扰动
            dx = y2 - y1  # 边的法线方向 (dy, -dx)
            dy = -(x2 - x1)
            norm = jnp.sqrt(dx ** 2 + dy ** 2)
            dx /= norm
            dy /= norm

            # 法线方向的扰动
            normal_perturb = jax.random.uniform(
                normal_key, (edge_samples,),
                minval=-DELTA,
                maxval=DELTA
            )
            x_edge += normal_perturb * dx
            y_edge += normal_perturb * dy

            # 收集采样点
            x_triangle = jnp.concatenate([x_triangle, x_edge])
            y_triangle = jnp.concatenate([y_triangle, y_edge])

        # 2. 生成时间采样 (每个空间点对应t_samples个时间点)
        t_triangle = jax.random.uniform(
            t_key, (edge_samples * 3, t_samples),
            minval=TIME_RANGE[0],
            maxval=TIME_RANGE[1]
        )

        # 3. 组合三维坐标
        x_triangle = jnp.repeat(x_triangle, t_samples)
        y_triangle = jnp.repeat(y_triangle, t_samples)
        t_triangle = t_triangle.reshape(-1)
        triangle_points = jnp.stack([x_triangle, y_triangle, t_triangle], axis=1)

        # ==================== 合并圆形和三角形边界采样 ====================
        points = jnp.concatenate([circle_points, triangle_points], axis=0)

        # ==================== 边界剪裁 ====================
        points = jnp.clip(
            points,
            jnp.array([xmin[0], xmin[1], TIME_RANGE[0]]),
            jnp.array([xmax[0], xmax[1], TIME_RANGE[1]])
        )
        return points

    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax + xmin) / 2, (xmax - xmin) / 2
        x = networks.norm(mu, sd, x)
        return x
    @staticmethod
    def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin,xmax,b in zip(xmin, xmax, batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)# (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":
                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1,-1)), xmax.reshape((1,-1))
            x_batch = xmin + (xmax - xmin)*s

        return jnp.array(x_batch)

    def _rectangle_sampler_start(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
                  enumerate(batch_shape)]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
            x_batch = xx.reshape((-1, xd))
        else:
            if sampler == "halton":
                # use scipy as not implemented in jax (!)
                r = scipy.stats.qmc.Halton(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "sobol":

                r = scipy.stats.qmc.Sobol(xd)
                s = r.random(np.prod(batch_shape))
            elif sampler == "uniform":
                # Generate uniform samples for the first two dimensions
                s = jax.random.uniform(key, (np.prod(batch_shape), 2))
                # Append the minimum value for the third dimension
                s = jnp.hstack((s, jnp.full((np.prod(batch_shape), 1), xmin[2])))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        return jnp.array(x_batch)




