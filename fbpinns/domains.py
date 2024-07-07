# """
# Defines problem domains
#
# Each domain class must inherit from the Domain base class.
# Each domain class must define the NotImplemented methods.
#
# This module is used by constants.py (and subsequently trainers.py)
# """
#
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from jax import random
from scipy.spatial.distance import cdist
from scipy.stats import truncnorm
from scipy.stats import qmc
from collections import defaultdict
from fbpinns import networks
from matplotlib.path import Path
from shapely.geometry import Point, Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

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

#
#
#
#
# class RectangularDomainND(Domain):
#
#     @staticmethod
#     def init_params(xmin, xmax):
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#
#         static_params = {
#             "xd":xd,
#             "xmin":jnp.array(xmin),
#             "xmax":jnp.array(xmax),
#             }
#         return static_params, {}
#
#     @staticmethod
#     def sample_interior(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_start1d(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary1d(all_params, key, sampler, batch_shape, loc):
#
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc)
#     @staticmethod
#     def sample_start2d(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary2d(all_params, key, sampler, batch_shape, dim, loc):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, dim, loc)
#
#     @staticmethod
#     def sample_boundaries(all_params, key, sampler, batch_shapes):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         xd = all_params["static"]["domain"]["xd"]
#
#         assert len(batch_shapes) == 2*xd# total number of boundaries
#         x_batches = []
#         for i in range(xd):
#             ic = jnp.array(list(range(i))+list(range(i+1,xd)), dtype=int)
#             for j,v in enumerate([xmin[i], xmax[i]]):
#                 batch_shape = batch_shapes[2*i+j]
#                 if len(ic):
#                     xmin_, xmax_ = xmin[ic], xmax[ic]
#                     key, subkey = jax.random.split(key)
#                     x_batch_ = RectangularDomainND._rectangle_samplerND(subkey, sampler, xmin_, xmax_, batch_shape)# (n, xd-1)
#                     x_batch = v*jnp.ones((jnp.prod(jnp.array(batch_shape)),xd), dtype=float)
#                     x_batch = x_batch.at[:,ic].set(x_batch_)
#                 else:
#                     assert len(batch_shape) == 1
#                     x_batch = v*jnp.ones(batch_shape+(1,), dtype=float)
#                 x_batches.append(x_batch)
#         return x_batches
#
#     @staticmethod
#     def norm_fn(all_params, x):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
#         x = networks.norm(mu, sd, x)
#         return x
#
#     @staticmethod
#     def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin, xmax, b) for xmin,xmax,b in zip(xmin, xmax, batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)# (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1,-1)), xmax.reshape((1,-1))
#             x_batch = xmin + (xmax - xmin)*s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 1 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
#             xs = [
#                 jnp.array([loc]) if i == 0 else  # 对于第一个维度，在loc处取值
#                 jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
#                 for i, b in enumerate(batch_shape)
#             ]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#     @staticmethod
#     def _rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     @staticmethod
#     def _rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, dim, loc):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
#             xs = [
#                 jnp.array([loc]) if i == dim else  # 对于第一个维度，在loc处取值
#                 jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
#                 for i, b in enumerate(batch_shape)
#             ]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     # def _rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape):
#     #     "Get flattened samples of x in a rectangle, either on mesh or random"
#     #
#     #     assert xmin.shape == xmax.shape
#     #     assert xmin.ndim == 1
#     #     xd = len(xmin)
#     #
#     #     if not sampler in ["grid", "uniform", "sobol", "halton"]:
#     #         raise ValueError("ERROR: unexpected sampler")
#     #
#     #     if sampler == "grid":
#     #         xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
#     #         xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#     #         x_batch = xx.reshape((-1, xd))
#     #         x_batch = x_batch[(x_batch[:, 0] == xmin[0]) | (x_batch[:, 0] == xmax[0]) |
#     #                                   (x_batch[:, 1] == xmin[1]) | (x_batch[:, 1] == xmax[1])]
#     #     else:
#     #         if sampler == "halton":
#     #             # use scipy as not implemented in jax (!)
#     #             r = scipy.stats.qmc.Halton(xd)
#     #             s = r.random(np.prod(batch_shape))
#     #         elif sampler == "sobol":
#     #             r = scipy.stats.qmc.Sobol(xd)
#     #             s = r.random(np.prod(batch_shape))
#     #         elif sampler == "uniform":
#     #             s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#     #
#     #         xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#     #         x_batch = xmin + (xmax - xmin) * s
#     #
#     #     return jnp.array(x_batch)
# class RectangularDomainND(Domain):
#
#     @staticmethod
#     def init_params(xmin, xmax):
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#
#         static_params = {
#             "xd":xd,
#             "xmin":jnp.array(xmin),
#             "xmax":jnp.array(xmax),
#             }
#         return static_params, {}
#
#     @staticmethod
#     def sample_interior(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_samplerND(key, sampler, xmin, xmax, batch_shape)
#     def sample_start2d(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape)
#     @staticmethod
#     def norm_fn(all_params, x):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
#         x = networks.norm(mu, sd, x)
#         return x
#
#     @staticmethod
#     def _rectangle_samplerND(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#
#     def _rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         return jnp.array(x_batch)
#     def sample_interior_decircle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_samplerND_circle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_start2d_circle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDD_circle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary2d_circle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def _rectangle_samplerND_circle(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         # After generating x_batch
#         xmin_xy = xmin[:2]
#         xmax_xy = xmax[:2]
#         x_batch_xy = x_batch[:, :2]
#         x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
#         y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
#         xy_center = np.array([[x_center, y_center]])
#         # Compute the center of the rectangle
#         side_lengths = xmax_xy - xmin_xy
#         radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
#
#         # 首先，根据欧氏距离筛选出圆外的点
#         distances = cdist(x_batch_xy, xy_center, metric='euclidean')
#         mask_distance = distances > radius  # 选择圆外的点
#         # 然后，检查最后一个维度是否为0
#         last_dim_zero_mask = x_batch[:, -1] != 0  # 选出最后一个维度不为0的点
#         # 结合两个条件，使用逻辑与操作，保留同时满足两个条件的点
#         combined_mask = np.logical_and(mask_distance.all(axis=1), last_dim_zero_mask)
#         # 应用组合后的掩码进行过滤
#         x_filtered = x_batch[combined_mask]
#
#         return jnp.array(x_filtered)
#         # return jnp.array(x_batch)
#     @staticmethod
#     def _rectangle_sampler2NDD_circle(key, sampler, xmin, xmax, batch_shape):
#         "Get flattened samples of x in a rectangle, either on mesh or random"
#
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#
#         if not sampler in ["grid", "uniform", "sobol", "halton"]:
#             raise ValueError("ERROR: unexpected sampler")
#
#         if sampler == "grid":
#             xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 2 else jnp.array([xmin[i]]) for i, b in
#                   enumerate(batch_shape)]
#             xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
#             x_batch = xx.reshape((-1, xd))
#         else:
#             if sampler == "halton":
#                 # use scipy as not implemented in jax (!)
#                 r = scipy.stats.qmc.Halton(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "sobol":
#                 r = scipy.stats.qmc.Sobol(xd)
#                 s = r.random(np.prod(batch_shape))
#             elif sampler == "uniform":
#                 s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#
#             xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#             x_batch = xmin + (xmax - xmin) * s
#
#         # After generating x_batch
#         xmin_xy = xmin[:2]
#         xmax_xy = xmax[:2]
#         x_batch_xy = x_batch[:, :2]
#         x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
#         y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
#         xy_center = np.array([[x_center, y_center]])
#         # Compute the center of the rectangle
#         side_lengths = xmax_xy - xmin_xy
#         radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
#
#         # Filter out points that fall within the circle
#         distances = cdist(x_batch_xy, xy_center, metric='euclidean')
#         mask = distances > radius  # Points outside the circle
#         x_filtered = x_batch[mask.all(axis=1)]
#
#         return jnp.array(x_filtered)
#
#     def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
#         assert xmin.shape == xmax.shape
#         assert xmin.ndim == 1
#         xd = len(xmin)
#         assert len(batch_shape) == xd
#         # After generating x_batch
#         xmin_xy = xmin[:2]
#         xmax_xy = xmax[:2]
#         x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
#         y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
#         side_lengths = xmax_xy - xmin_xy
#         radius = np.min(side_lengths) / 4.0
#         def generate_circular_boundary_points(center, r, num_points):
#             cx = center[0]
#             cy = center[1]
#             theta = np.linspace(0, 2 * np.pi, num_points)
#             x = cx + r * np.cos(theta)
#             y = cy + r * np.sin(theta)
#             circle_points = np.column_stack((x, y))
#             return circle_points.tolist()
#         pboundary = generate_circular_boundary_points([x_center, y_center], radius, batch_shape[0] * batch_shape[1])
#         def foo(input_list,time_start, time_end, num_time_samples):
#             expanded_list = []
#             hw = np.linspace(time_start, time_end, num_time_samples)
#             for sublist in input_list:
#                 for i in hw:
#                     expanded_list.append(sublist + [i])
#             return expanded_list
#
#         x_filtered = foo(pboundary,xmin[2],xmax[2],batch_shape[2])
#
#         return jnp.array(x_filtered)
#
#
#     # def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
#     #     # generating x_batch
#     #     xmin_xy = xmin[:2]
#     #     xmax_xy = xmax[:2]
#     #     x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
#     #     y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
#     #     side_lengths = xmax_xy - xmin_xy
#     #     radius = np.min(side_lengths) / 4.0
#     #     def generate_circular_points(center, r, num_points):
#     #         theta = np.random.uniform(0, 2 * np.pi, num_points)
#     #         radius_samples = r * np.sqrt(np.random.uniform(0, 1, num_points))  # 使用均匀分布的半径样本
#     #         x = center[0] + radius_samples * np.cos(theta)
#     #         y = center[1] + radius_samples * np.sin(theta)
#     #         circle_points = np.column_stack((x, y))
#     #         return circle_points.tolist()
#     #     pboundary = generate_circular_points([x_center, y_center], radius, batch_shape[0] * batch_shape[1])
#     #
#     #     #       pdb.set_trace()
#     #     def foo(input_list,time_start, time_end, num_time_samples):
#     #         expanded_list = []
#     #         hw = np.linspace(time_start, time_end, num_time_samples)
#     #         for sublist in input_list:
#     #             for i in hw:
#     #                 expanded_list.append(sublist + [i])
#     #         return expanded_list
#     #
#     #     x_filtered = foo(pboundary,xmin[2],xmax[2],batch_shape[2])
#     #     #      pdb.set_trace()
#     #     return jnp.array(x_filtered)
#
#     # def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
#     #     assert xmin.shape == xmax.shape
#     #     assert xmin.ndim == 1
#     #     xd = len(xmin)
#     #     assert len(batch_shape) == xd
#     #
#     #     if not sampler in ["grid", "uniform", "sobol", "halton"]:
#     #         raise ValueError("ERROR: unexpected sampler")
#     #
#     #     # 空间采样点的数量
#     #     num_space_samples = batch_shape[0] * batch_shape[1]
#     #     # 时间范围和采样点的数量
#     #     time_start = xmin[2]
#     #     time_end = xmax[2]
#     #     num_time_samples = batch_shape[2]
#     #     #get x、y_center radius
#     #     xmin = xmin[:2]
#     #     xmax = xmax[:2]
#     #     x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
#     #     y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
#     #     xy_center = np.array([[x_center, y_center]])
#     #     # Compute the center of the rectangle
#     #     side_lengths = xmax - xmin
#     #     radius = np.min(side_lengths) / 4.0  # Use the shorter side's forth as radius
#     #
#     #     if sampler == "grid":
#     #
#     #         # 生成等间距的角度值，范围从0到2π
#     #         angles = np.linspace(0, 2 * np.pi, num_space_samples)
#     #         # 生成等间距的时间值
#     #         time_samples = np.linspace(time_start, time_end, num_time_samples)
#     #         # 初始化列表以存储所有采样点
#     #         all_samples = []
#     #         # 对每个时间点进行空间采样
#     #         for t in time_samples:
#     #             # 计算圆周上的 (x, y) 坐标
#     #             x_samples = x_center + radius * np.cos(angles)
#     #             y_samples = y_center + radius * np.sin(angles)
#     #
#     #             # 将时间坐标添加到每个空间采样点
#     #             samples_at_t = np.column_stack((x_samples, y_samples, np.full(num_space_samples, t)))
#     #             # 添加到所有采样点的列表
#     #             all_samples.append(samples_at_t)
#     #         # 将所有采样点组合成一个数组
#     #         x_batch = np.vstack(all_samples)
#     #     else:
#     #         if sampler == "halton":
#     #             # use scipy as not implemented in jax (!)
#     #             r = scipy.stats.qmc.Halton(xd)
#     #             s = r.random(np.prod(batch_shape))
#     #         elif sampler == "sobol":
#     #             r = scipy.stats.qmc.Sobol(xd)
#     #             s = r.random(np.prod(batch_shape))
#     #         elif sampler == "uniform":
#     #             s = jax.random.uniform(key, (np.prod(batch_shape), xd))
#     #
#     #         xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
#     #         x_batch = xmin + (xmax - xmin) * s
#     #
#     #     return jnp.array(x_batch)

def from_file_getdata(filename):
    """
    Reads a mesh file in the .mesh format and returns the node coordinates and triangle indices.
    """
    with open(filename, 'r') as file:
        # Skip the first line
        file.readline()

        # Read the number of points
        num_points = int(file.readline().strip())

        # Initialize the array to hold the points' coordinates
        node_coords = np.zeros((num_points, 2))

        # Read the points' coordinates
        for i in range(num_points):
            line = file.readline().strip()
            coords = list(map(float, line.split()[:2]))
            node_coords[i] = coords

        # Skip the empty line
        file.readline()

        # Read the number of triangles
        num_triangles = int(file.readline().strip())

        # Initialize the array to hold the triangles' indices
        triangles = np.zeros((num_triangles, 3), dtype=int)

        # Read the triangles' indices
        for i in range(num_triangles):
            line = file.readline().strip()
            indices = list(map(int, line.split()[:3]))
            triangles[i] = indices
    return node_coords, triangles

def get_boundary_points(node_coords, triangles):
    # Step 1: Find all edges
    edges = defaultdict(int)
    for triangle in triangles:
        for i in range(3):
            edge = tuple(sorted((triangle[i], triangle[(i + 1) % 3])))
            edges[edge] += 1

    # Step 2: Find boundary edges (edges that appear only once)
    boundary_edges = [edge for edge, count in edges.items() if count == 1]

    # Step 3: Find boundary points
    boundary_points_indices = set()
    for edge in boundary_edges:
        boundary_points_indices.update(edge)

    boundary_points = node_coords[list(boundary_points_indices)]

    return boundary_points


###########2222
def create_polygon_from_mesh(points, triangles):
    """
    从点坐标矩阵和三角形顶点索引构建多边形。

    参数:
    points (list of tuples): 点坐标矩阵，例如 [(x1, y1), (x2, y2), ...]
    triangles (list of tuples): 三角形顶点索引，例如 [(i1, i2, i3), (i4, i5, i6), ...]

    返回:
    Polygon: 由三角形网格构成的多边形
    """
    polygons = [Polygon([points[i] for i in triangle]) for triangle in triangles]
    unified_polygon = unary_union(polygons)
    return unified_polygon

def filter_point(point, polygon):
    shapely_point = Point(point)
    return shapely_point.within(polygon) or shapely_point.touches(polygon)

def filter_points(points, polygon, max_workers=4):
    points_xy = points[:, :2]

    filtered_points = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(filter_point, point, polygon): i for i, point in enumerate(points_xy)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                if not future.result():
                    filtered_points.append(points[index])  # Append the original 3D point
            except Exception as exc:
                print(f'Point {points_xy[index]} generated an exception: {exc}')

    return np.array(filtered_points)

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

    def sample_interior_depec(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_interior_depec(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start_depec(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_start_depec(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_boundary(key, sampler, xmin, xmax, batch_shape)
    @staticmethod
    def norm_fn(all_params, x):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        mu, sd = (xmax+xmin)/2, (xmax-xmin)/2
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
            xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
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
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        return jnp.array(x_batch)

    @staticmethod
    def _rectangle_samplerND_interior_depec(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
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
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        points, triangles = from_file_getdata("five.tri")
        polygon = create_polygon_from_mesh(points, triangles)
        x_filtered = filter_points(x_batch, polygon)
        # plt.scatter(x_filtered[:, 0], x_filtered[:, 1], c='r', marker='o')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('2D Points')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        return x_filtered

        #
        # def plot_points(boundary_points, filtered_points):
        #     plt.figure(figsize=(10, 10))
        #
        #     # Plot boundary points
        #     boundary_x, boundary_y = boundary_points[:, 0], boundary_points[:, 1]
        #     plt.plot(boundary_x, boundary_y, 'b-', label='Boundary')
        #
        #     # Plot filtered points
        #     if len(filtered_points) > 0:
        #         filtered_x, filtered_y = filtered_points[:, 0], filtered_points[:, 1]
        #         plt.scatter(filtered_x, filtered_y, color='red', label='Filtered Points')
        #
        #     plt.xlabel('X')
        #     plt.ylabel('Y')
        #     plt.legend()
        #     plt.title('Boundary and Filtered Points')
        #     plt.grid(True)
        #     plt.show()
        #
        # # 绘制边界点和经过过滤的点
        # plot_points(boundary_points, x_filtered)
    @staticmethod
    def _rectangle_samplerND_start_depec(key, sampler, xmin, xmax, batch_shape):
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
                s = jax.random.uniform(key, (np.prod(batch_shape), xd))

            xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
            x_batch = xmin + (xmax - xmin) * s

        points, triangles = from_file_getdata("five.tri")
        polygon = create_polygon_from_mesh(points, triangles)
        x_filtered = filter_points(x_batch, polygon)

        return x_filtered

        # return jnp.array(x_batch)
    def _rectangle_samplerND_boundary(key, sampler, xmin, xmax, batch_shape):

        points, triangles = from_file_getdata("five.tri")
        boundary_points = get_boundary_points(points, triangles)
        # 获取batch_shape前两个值
        num_xy_points = batch_shape[0] * batch_shape[1]

        # 从boundary_points中选取指定个数的点
        if num_xy_points > len(boundary_points):
            selected_boundary_points = boundary_points
        else:
            selected_boundary_points = boundary_points[:num_xy_points]

        # 根据xmin, xmax和batch_shape第三个维度的值对t进行划分
        t_values = np.linspace(xmin[2], xmax[2], batch_shape[2])

        # 将选定的xy的点与t值进行组合
        combined_points = []
        for t in t_values:
            for xy in selected_boundary_points:
                combined_points.append([xy[0], xy[1], t])

        return jnp.array(combined_points)


if __name__ == "__main__":

    # import matplotlib.pyplot as plt
    #
    # key = jax.random.PRNGKey(0)
    #
    #
    # domain = RectangularDomainND
    # sampler = "halton"
    #
    #
    # # 1D
    #
    # xmin, xmax = jnp.array([-1,]), jnp.array([2,])
    # batch_shape = (10,)
    # batch_shapes = ((3,),(4,))
    #
    # ps_ = domain.init_params(xmin, xmax)
    # all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    # x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    # x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)
    #
    # plt.figure()
    # plt.scatter(x_batch, jnp.zeros_like(x_batch))
    # for x_batch in x_batches:
    #     print(x_batch.shape)
    #     plt.scatter(x_batch, jnp.zeros_like(x_batch))
    # plt.show()
    #
    #
    # # 2D
    #
    # xmin, xmax = jnp.array([0,1]), jnp.array([1,2])
    # batch_shape = (10,20)
    # batch_shapes = ((3,),(4,),(5,),(6,))
    #
    # ps_ = domain.init_params(xmin, xmax)
    # all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    # x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    # x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)
    #
    # plt.figure()
    # plt.scatter(x_batch[:,0], x_batch[:,1])
    # for x_batch in x_batches:
    #     print(x_batch.shape)
    #     plt.scatter(x_batch[:,0], x_batch[:,1])
    # plt.show()
    #
    # # 3D
    # xmin, xmax = jnp.array([0, 1, 2]), jnp.array([1, 2, 3])  # 三维空间的最小和最大边界
    # batch_shape = (10, 20, 30)
    # batch_shapes = ((3, 4), (5, 6), (7, 8))  # 三维空间中每个维度的批次形状
    #
    # # 生成示例参数
    # ps_ = domain.init_params(xmin, xmax)
    # all_params = {"static": {"domain": ps_[0]}, "trainable": {"domain": ps_[1]}}
    #
    # # 采样内部点集
    # x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    # # 采样边界点集
    # x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)
    #
    # # 绘图
    # plt.figure()
    # plt.scatter(x_batch[:, 0], x_batch[:, 1], x_batch[:, 2])
    # for x_batch in x_batches:
    #     print(x_batch.shape)
    #     plt.scatter(x_batch[:, 0], x_batch[:, 1], x_batch[:, 2])
    # plt.show()

    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)

    domain = RectangularDomainND
    sampler = "grid"

    # 3D
    xmin, xmax = jnp.array([-1, -1, 0]), jnp.array([1, 1, 2])  # 三维空间的最小和最大边界

    ns = ((50, 50, 30),),
    n_start = ((100, 100, 1),),
    n_boundary = ((100, 100, 80),),
    n_test = (100, 100, 20),

    batch_shapes = (50, 50, 30)
    start_batch_shapes = (100, 100, 1)
    boundary_batch_shapes = (100, 100, 80)
    test_batch_shapes = (100, 100, 20)
    # 生成示例参数
    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static": {"domain": ps_[0]}, "trainable": {"domain": ps_[1]}}

    x_batch_phys = domain.sample_interior_depec(all_params, key, sampler, batch_shapes)
    x_batch_start = domain.sample_start_depec(all_params, key, sampler, start_batch_shapes)
    x_batch_boundary = domain.sample_boundary(all_params, key, sampler, boundary_batch_shapes)
    x_batch_test = domain.sample_interior(all_params, key, sampler, test_batch_shapes)
    x_batch_test_depec = domain.sample_interior_depec(all_params, key, sampler, test_batch_shapes)

    # 将数组保存到文件
    np.save('x_batch_phys.npy', np.array(x_batch_phys))
    np.save('x_batch_start.npy', np.array(x_batch_start))
    np.save('x_batch_boundary.npy', np.array(x_batch_boundary))
    np.save('x_batch_test.npy', np.array(x_batch_test))
    np.save('x_batch_test_depec.npy', np.array(x_batch_test_depec))



