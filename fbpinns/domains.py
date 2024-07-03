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
    def sample_interior_dense(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_dense(key, sampler, xmin, xmax, batch_shape)
    def sample_start2d(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape)
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
    # two_parts_dense
    def _rectangle_samplerND_dense(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random, with dense sampling in a specified x range"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        def generate_samples(sampler, key, xmin, xmax, batch_shape):
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

            return x_batch

        # Generate uniform samples for the entire range
        x_batch_uniform = generate_samples(sampler, key, xmin, xmax, batch_shape)
        dense_x_range = (-0.1, 0.1)
        dense_multiplier = 5
        # Generate dense samples for the specified x range
        xmin_dense = xmin.at[0].set(dense_x_range[0])
        xmax_dense = xmax.at[0].set(dense_x_range[1])
        batch_shape_dense = [dense_multiplier * b if xmin[i] == dense_x_range[0] and xmax[i] == dense_x_range[1] else b
                             for i, b in enumerate(batch_shape)]

        x_batch_dense = generate_samples(sampler, key, xmin_dense, xmax_dense, batch_shape_dense)

        # Combine uniform and dense samples
        x_batch = jnp.concatenate([x_batch_uniform, x_batch_dense], axis=0)

        return jnp.array(x_batch)

    @staticmethod
    # four_parts_dense
    def _rectangle_samplerND_dense(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random, with dense sampling in specified ranges"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if sampler not in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        def generate_samples(sampler, key, xmin, xmax, batch_shape):
            if sampler == "grid":
                xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
                xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
                x_batch = xx.reshape((-1, xd))
            else:
                if sampler == "halton":
                    r = scipy.stats.qmc.Halton(xd)
                    s = r.random(np.prod(batch_shape))
                elif sampler == "sobol":
                    r = scipy.stats.qmc.Sobol(xd)
                    s = r.random(np.prod(batch_shape))
                elif sampler == "uniform":
                    s = jax.random.uniform(key, (np.prod(batch_shape), xd))

                xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
                x_batch = xmin + (xmax - xmin) * s

            return x_batch

        # Generate uniform samples for the entire range
        x_batch_uniform = generate_samples(sampler, key, xmin, xmax, batch_shape)

        # Dense sampling ranges and multipliers
        dense_ranges = [((-0.1, 0.1), (-1, 1)), ((-1, 1), (-0.1, 0.1))]
        dense_multipliers = [3, 3]

        x_batch_dense_all = []
        for dense_range, multiplier in zip(dense_ranges, dense_multipliers):
            xmin_dense = xmin.copy()
            xmax_dense = xmax.copy()
            batch_shape_dense = batch_shape.copy()

            for i, (dense_min, dense_max) in enumerate(dense_range):
                xmin_dense = xmin_dense.at[i].set(dense_min)
                xmax_dense = xmax_dense.at[i].set(dense_max)
                batch_shape_dense[i] *= multiplier

            x_batch_dense = generate_samples(sampler, key, xmin_dense, xmax_dense, batch_shape_dense)
            x_batch_dense_all.append(x_batch_dense)

        # Combine uniform and dense samples
        x_batch = jnp.concatenate([x_batch_uniform] + x_batch_dense_all, axis=0)

        return jnp.array(x_batch)


    def _rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape):
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

        return jnp.array(x_batch)
    def sample_interior_decircle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_samplerND_circle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start2d_circle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDD_circle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary2d_circle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def _rectangle_samplerND_circle(key, sampler, xmin, xmax, batch_shape):
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

        # # After generating x_batch
        # xmin_xy = xmin[:2]
        # xmax_xy = xmax[:2]
        # x_batch_xy = x_batch[:, :2]
        # x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
        # y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
        # xy_center = np.array([[x_center, y_center]])
        # # Compute the center of the rectangle
        # side_lengths = xmax_xy - xmin_xy
        # radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
        #
        # # 首先，根据欧氏距离筛选出圆外的点
        # distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        # mask_distance = distances > radius  # 选择圆外的点
        # # 然后，检查最后一个维度是否为0
        # last_dim_zero_mask = x_batch[:, -1] != 0  # 选出最后一个维度不为0的点
        # # 结合两个条件，使用逻辑与操作，保留同时满足两个条件的点
        # combined_mask = np.logical_and(mask_distance.all(axis=1), last_dim_zero_mask)
        # # 应用组合后的掩码进行过滤
        # x_filtered = x_batch[combined_mask]

        # return jnp.array(x_filtered)
        return jnp.array(x_batch)
    @staticmethod
    def _rectangle_sampler2NDD_circle(key, sampler, xmin, xmax, batch_shape):
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

        # # After generating x_batch
        # xmin_xy = xmin[:2]
        # xmax_xy = xmax[:2]
        # x_batch_xy = x_batch[:, :2]
        # x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
        # y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
        # xy_center = np.array([[x_center, y_center]])
        # # Compute the center of the rectangle
        # side_lengths = xmax_xy - xmin_xy
        # radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
        #
        # # Filter out points that fall within the circle
        # distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        # mask = distances > radius  # Points outside the circle
        # x_filtered = x_batch[mask.all(axis=1)]

        # return jnp.array(x_filtered)

        return jnp.array(x_batch)
    def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
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
        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.0
        def generate_circular_boundary_points(center, r, num_points):
            cx = center[0]
            cy = center[1]
            theta = np.linspace(0, 2 * np.pi, num_points)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            circle_points = np.column_stack((x, y))
            return circle_points.tolist()
        pboundary = generate_circular_boundary_points([x_center, y_center], radius, batch_shape[0] * batch_shape[1])

        #       pdb.set_trace()
        def foo(input_list,time_start, time_end, num_time_samples):
            expanded_list = []
            hw = np.linspace(time_start, time_end, num_time_samples)
            for sublist in input_list:
                for i in hw:
                    expanded_list.append(sublist + [i])
            return expanded_list

        x_filtered = foo(pboundary,xmin[2],xmax[2],batch_shape[2])
        #      pdb.set_trace()
        return jnp.array(x_filtered)


    # def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
    #     # generating x_batch
    #     xmin_xy = xmin[:2]
    #     xmax_xy = xmax[:2]
    #     x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
    #     y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
    #     side_lengths = xmax_xy - xmin_xy
    #     radius = np.min(side_lengths) / 4.0
    #     def generate_circular_points(center, r, num_points):
    #         theta = np.random.uniform(0, 2 * np.pi, num_points)
    #         radius_samples = r * np.sqrt(np.random.uniform(0, 1, num_points))  # 使用均匀分布的半径样本
    #         x = center[0] + radius_samples * np.cos(theta)
    #         y = center[1] + radius_samples * np.sin(theta)
    #         circle_points = np.column_stack((x, y))
    #         return circle_points.tolist()
    #     pboundary = generate_circular_points([x_center, y_center], radius, batch_shape[0] * batch_shape[1])
    #
    #     #       pdb.set_trace()
    #     def foo(input_list,time_start, time_end, num_time_samples):
    #         expanded_list = []
    #         hw = np.linspace(time_start, time_end, num_time_samples)
    #         for sublist in input_list:
    #             for i in hw:
    #                 expanded_list.append(sublist + [i])
    #         return expanded_list
    #
    #     x_filtered = foo(pboundary,xmin[2],xmax[2],batch_shape[2])
    #     #      pdb.set_trace()
    #     return jnp.array(x_filtered)

    # def _rectangle_sampler2NDDD_circle(key, sampler, xmin, xmax, batch_shape):
    #     assert xmin.shape == xmax.shape
    #     assert xmin.ndim == 1
    #     xd = len(xmin)
    #     assert len(batch_shape) == xd
    #
    #     if not sampler in ["grid", "uniform", "sobol", "halton"]:
    #         raise ValueError("ERROR: unexpected sampler")
    #
    #     # 空间采样点的数量
    #     num_space_samples = batch_shape[0] * batch_shape[1]
    #     # 时间范围和采样点的数量
    #     time_start = xmin[2]
    #     time_end = xmax[2]
    #     num_time_samples = batch_shape[2]
    #     #get x、y_center radius
    #     xmin = xmin[:2]
    #     xmax = xmax[:2]
    #     x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
    #     y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
    #     xy_center = np.array([[x_center, y_center]])
    #     # Compute the center of the rectangle
    #     side_lengths = xmax - xmin
    #     radius = np.min(side_lengths) / 4.0  # Use the shorter side's forth as radius
    #
    #     if sampler == "grid":
    #
    #         # 生成等间距的角度值，范围从0到2π
    #         angles = np.linspace(0, 2 * np.pi, num_space_samples)
    #         # 生成等间距的时间值
    #         time_samples = np.linspace(time_start, time_end, num_time_samples)
    #         # 初始化列表以存储所有采样点
    #         all_samples = []
    #         # 对每个时间点进行空间采样
    #         for t in time_samples:
    #             # 计算圆周上的 (x, y) 坐标
    #             x_samples = x_center + radius * np.cos(angles)
    #             y_samples = y_center + radius * np.sin(angles)
    #
    #             # 将时间坐标添加到每个空间采样点
    #             samples_at_t = np.column_stack((x_samples, y_samples, np.full(num_space_samples, t)))
    #             # 添加到所有采样点的列表
    #             all_samples.append(samples_at_t)
    #         # 将所有采样点组合成一个数组
    #         x_batch = np.vstack(all_samples)
    #     else:
    #         if sampler == "halton":
    #             # use scipy as not implemented in jax (!)
    #             r = scipy.stats.qmc.Halton(xd)
    #             s = r.random(np.prod(batch_shape))
    #         elif sampler == "sobol":
    #             r = scipy.stats.qmc.Sobol(xd)
    #             s = r.random(np.prod(batch_shape))
    #         elif sampler == "uniform":
    #             s = jax.random.uniform(key, (np.prod(batch_shape), xd))
    #
    #         xmin, xmax = xmin.reshape((1, -1)), xmax.reshape((1, -1))
    #         x_batch = xmin + (xmax - xmin) * s
    #
    #     return jnp.array(x_batch)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    key = jax.random.PRNGKey(0)


    domain = RectangularDomainND
    sampler = "halton"


    # 1D

    xmin, xmax = jnp.array([-1,]), jnp.array([2,])
    batch_shape = (10,)
    batch_shapes = ((3,),(4,))

    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    plt.figure()
    plt.scatter(x_batch, jnp.zeros_like(x_batch))
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch, jnp.zeros_like(x_batch))
    plt.show()


    # 2D

    xmin, xmax = jnp.array([0,1]), jnp.array([1,2])
    batch_shape = (10,20)
    batch_shapes = ((3,),(4,),(5,),(6,))

    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static":{"domain":ps_[0]}, "trainable":{"domain":ps_[1]}}
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    plt.figure()
    plt.scatter(x_batch[:,0], x_batch[:,1])
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch[:,0], x_batch[:,1])
    plt.show()

    # 3D
    xmin, xmax = jnp.array([0, 1, 2]), jnp.array([1, 2, 3])  # 三维空间的最小和最大边界
    batch_shape = (10, 20, 30)
    batch_shapes = ((3, 4), (5, 6), (7, 8))  # 三维空间中每个维度的批次形状

    # 生成示例参数
    ps_ = domain.init_params(xmin, xmax)
    all_params = {"static": {"domain": ps_[0]}, "trainable": {"domain": ps_[1]}}

    # 采样内部点集
    x_batch = domain.sample_interior(all_params, key, sampler, batch_shape)
    # 采样边界点集
    x_batches = domain.sample_boundaries(all_params, key, sampler, batch_shapes)

    # 绘图
    plt.figure()
    plt.scatter(x_batch[:, 0], x_batch[:, 1], x_batch[:, 2])
    for x_batch in x_batches:
        print(x_batch.shape)
        plt.scatter(x_batch[:, 0], x_batch[:, 1], x_batch[:, 2])
    plt.show()





