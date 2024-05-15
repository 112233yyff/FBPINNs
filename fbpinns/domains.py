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
    def sample_start1d(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary1d(all_params, key, sampler, batch_shape, loc):

        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc)
    @staticmethod
    def sample_start2d(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDD(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary2d(all_params, key, sampler, batch_shape, dim, loc):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, dim, loc)

    @staticmethod
    def sample_boundaries(all_params, key, sampler, batch_shapes):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        xd = all_params["static"]["domain"]["xd"]

        assert len(batch_shapes) == 2*xd# total number of boundaries
        x_batches = []
        for i in range(xd):
            ic = jnp.array(list(range(i))+list(range(i+1,xd)), dtype=int)
            for j,v in enumerate([xmin[i], xmax[i]]):
                batch_shape = batch_shapes[2*i+j]
                if len(ic):
                    xmin_, xmax_ = xmin[ic], xmax[ic]
                    key, subkey = jax.random.split(key)
                    x_batch_ = RectangularDomainND._rectangle_samplerND(subkey, sampler, xmin_, xmax_, batch_shape)# (n, xd-1)
                    x_batch = v*jnp.ones((jnp.prod(jnp.array(batch_shape)),xd), dtype=float)
                    x_batch = x_batch.at[:,ic].set(x_batch_)
                else:
                    assert len(batch_shape) == 1
                    x_batch = v*jnp.ones(batch_shape+(1,), dtype=float)
                x_batches.append(x_batch)
        return x_batches

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

    @staticmethod
    def _rectangle_sampler1NDD(key, sampler, xmin, xmax, batch_shape):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            xs = [jnp.linspace(xmin[i], xmax[i], b) if i != 1 else jnp.array([xmin[i]]) for i, b in
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

    @staticmethod
    def _rectangle_sampler1NDDD(key, sampler, xmin, xmax, batch_shape, loc):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
            xs = [
                jnp.array([loc]) if i == 0 else  # 对于第一个维度，在loc处取值
                jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
                for i, b in enumerate(batch_shape)
            ]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
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

    @staticmethod
    @staticmethod
    def _rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape, dim, loc):
        "Get flattened samples of x in a rectangle, either on mesh or random"

        assert xmin.shape == xmax.shape
        assert xmin.ndim == 1
        xd = len(xmin)
        assert len(batch_shape) == xd

        if not sampler in ["grid", "uniform", "sobol", "halton"]:
            raise ValueError("ERROR: unexpected sampler")

        if sampler == "grid":
            assert xmin[0] <= loc <= xmax[0], "loc must be within the range defined by xmin[0] and xmax[0]"
            xs = [
                jnp.array([loc]) if i == dim else  # 对于第一个维度，在loc处取值
                jnp.linspace(xmin[i], xmax[i], b)  # 对于其他维度（包括第二个维度），按均匀间隔取样
                for i, b in enumerate(batch_shape)
            ]
            xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)
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

    # def _rectangle_sampler2NDDD(key, sampler, xmin, xmax, batch_shape):
    #     "Get flattened samples of x in a rectangle, either on mesh or random"
    #
    #     assert xmin.shape == xmax.shape
    #     assert xmin.ndim == 1
    #     xd = len(xmin)
    #
    #     if not sampler in ["grid", "uniform", "sobol", "halton"]:
    #         raise ValueError("ERROR: unexpected sampler")
    #
    #     if sampler == "grid":
    #         xs = [jnp.linspace(xmin, xmax, b) for xmin, xmax, b in zip(xmin, xmax, batch_shape)]
    #         xx = jnp.stack(jnp.meshgrid(*xs, indexing="ij"), -1)  # (batch_shape, xd)
    #         x_batch = xx.reshape((-1, xd))
    #         x_batch = x_batch[(x_batch[:, 0] == xmin[0]) | (x_batch[:, 0] == xmax[0]) |
    #                                   (x_batch[:, 1] == xmin[1]) | (x_batch[:, 1] == xmax[1])]
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
#     def sample_interior_cycle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_samplerND_cycle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_start2d_cycle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDD_cycle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def sample_boundary2d_cycle(all_params, key, sampler, batch_shape):
#         xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
#         return RectangularDomainND._rectangle_sampler2NDDD_cycle(key, sampler, xmin, xmax, batch_shape)
#
#     @staticmethod
#     def _rectangle_samplerND_cycle(key, sampler, xmin, xmax, batch_shape):
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
#         xmin = xmin[:2]
#         xmax = xmax[:2]
#         x_batch_xy = x_batch[:, :2]
#         x_center = xmin[0] + (3 / 4) * (xmax[0] - xmin[0])
#         y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
#         xy_center = np.array([[x_center, y_center]])
#         # Compute the center of the rectangle
#         side_lengths = xmax - xmin
#         radius = np.min(side_lengths) / 10  # Use the shorter side's fifth as radius
#
#         # Filter out points that fall within the circle
#         distances = cdist(x_batch_xy, xy_center, metric='euclidean')
#         mask = distances > radius  # Points outside the circle
#         x_filtered = x_batch[mask.all(axis=1)]
#
#         return jnp.array(x_filtered)
#     @staticmethod
#     def _rectangle_sampler2NDD_cycle(key, sampler, xmin, xmax, batch_shape):
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
#         xmin = xmin[:2]
#         xmax = xmax[:2]
#         x_batch_xy = x_batch[:, :2]
#         x_center = xmin[0] + (3 / 4) * (xmax[0] - xmin[0])
#         y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
#         xy_center = np.array([[x_center, y_center]])
#         # Compute the center of the rectangle
#         side_lengths = xmax - xmin
#         radius = np.min(side_lengths) / 10  # Use the shorter side's fifth as radius
#
#         # Filter out points that fall within the circle
#         distances = cdist(x_batch_xy, xy_center, metric='euclidean')
#         mask = distances > radius  # Points outside the circle
#         x_filtered = x_batch[mask.all(axis=1)]
#
#         return jnp.array(x_filtered)
#
#     def _rectangle_sampler2NDDD_cycle(key, sampler, xmin, xmax, batch_shape):
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
#         xmin = xmin[:2]
#         xmax = xmax[:2]
#         x_batch_xy = x_batch[:, :2]
#         x_center = xmin[0] + (3 / 4) * (xmax[0] - xmin[0])
#         y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
#         xy_center = np.array([[x_center, y_center]])
#         # Compute the center of the rectangle
#         side_lengths = xmax - xmin
#         radius = np.min(side_lengths) / 10  # Use the shorter side's fifth as radius
#
#         # Filter out points that fall within the circle
#         distances = cdist(x_batch_xy, xy_center, metric='euclidean')
#         mask = jnp.abs(distances - radius) <= 0.001  # Points outside the circle
#         x_filtered = x_batch[mask.all(axis=1)]
#
#         return jnp.array(x_filtered)

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





