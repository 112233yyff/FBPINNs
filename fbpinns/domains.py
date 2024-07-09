"""
Defines problem domains

Each domain class must inherit from the Domain base class.
Each domain class must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import jax, pdb
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

    def sample_interior_decircle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_interior_decircle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start_decircle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_start_decircle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary_circle(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_boundary_circle(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def _rectangle_interior_decircle(key, sampler, xmin, xmax, batch_shape):
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

        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius

        # Filter out points that fall within the circle
        distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        mask = distances > radius  # Points outside the circle
        x_filtered = x_batch[mask.all(axis=1)]

        return jnp.array(x_filtered)
    @staticmethod
    def _rectangle_start_decircle(key, sampler, xmin, xmax, batch_shape):
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

        # After generating x_batch
        xmin = xmin[:2]
        xmax = xmax[:2]
        x_batch_xy = x_batch[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax - xmin
        radius = np.min(side_lengths) / 4.  # Use the shorter side's fifth as radius

        # Filter out points that fall within the circle
        distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        mask = distances > radius  # Points outside the circle

        x_filtered = x_batch[mask.all(axis=1)]
        return jnp.array(x_filtered)

    def _rectangle_boundary_circle(key, sampler, xmin, xmax, batch_shape):
        # generating x_batch
        xmin_xy = xmin[:2]
        xmax_xy = xmax[:2]
        x_center = xmin_xy[0] + (1 / 2) * (xmax_xy[0] - xmin_xy[0])
        y_center = xmin_xy[1] + (1 / 4) * (xmax_xy[1] - xmin_xy[1])
        side_lengths = xmax_xy - xmin_xy
        radius = np.min(side_lengths) / 4.0
        def generate_circular_points(center, r, num_points):
            theta = np.random.uniform(0, 2 * np.pi, num_points)
            radius_samples = r * np.sqrt(np.random.uniform(0, 1, num_points))  # 使用均匀分布的半径样本
            x = center[0] + radius_samples * np.cos(theta)
            y = center[1] + radius_samples * np.sin(theta)
            circle_points = np.column_stack((x, y))
            return circle_points.tolist()
        pboundary = generate_circular_points([x_center, y_center], radius, batch_shape[0] * batch_shape[1])

        def foo(input_list,time_start, time_end, num_time_samples):
            expanded_list = []
            hw = np.linspace(time_start, time_end, num_time_samples)
            for sublist in input_list:
                for i in hw:
                    expanded_list.append(sublist + [i])
            return expanded_list

        x_filtered = foo(pboundary,xmin[2],xmax[2],batch_shape[2])

        return jnp.array(x_filtered)

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




