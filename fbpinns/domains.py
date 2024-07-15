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
from shapely.geometry import Point, Polygon
from shapely.geometry import Point, Polygon
from concurrent.futures import ThreadPoolExecutor

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
    def sample_interior_depec(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_interior_depec(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_start_depec(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_start_depec(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def sample_boundary_pec(all_params, key, sampler, batch_shape):
        xmin, xmax = all_params["static"]["domain"]["xmin"], all_params["static"]["domain"]["xmax"]
        return RectangularDomainND._rectangle_boundary_pec(key, sampler, xmin, xmax, batch_shape)

    @staticmethod
    def foo(input_list, time_start, time_end, num_time_samples):
        expanded_list = []
        hw = np.linspace(time_start, time_end, num_time_samples)
        for sublist in input_list:
            sublist = tuple(sublist)  # Ensure sublist is a tuple
            for i in hw:
                expanded_list.append(sublist + (i,))
        return expanded_list
    @staticmethod
    def boundary_circle(key, sampler, xmin, xmax, batch_shape):
        def generate_circular_boundary_points(center, r, num_points):
            theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            return np.column_stack((x, y))
        # Generate boundary points on the circle
        num_boundary_points = batch_shape[0] * batch_shape[1]
        pboundary = generate_circular_boundary_points([-0.7, 0.5], 0.25, num_boundary_points)
        x_filtered = RectangularDomainND.foo(pboundary.tolist(), xmin[2], xmax[2], batch_shape[2])

        return jnp.array(x_filtered)

    @staticmethod
    def boundary_rectangle(key, sampler, xmin, xmax, batch_shape):
        def generate_square_boundary_points(center, side_length, num_points_per_side=10):
            half_side_length = side_length / 2
            vertices = [
                (center[0] - half_side_length, center[1] - half_side_length),  # Bottom left
                (center[0] + half_side_length, center[1] - half_side_length),  # Bottom right
                (center[0] + half_side_length, center[1] + half_side_length),  # Top right
                (center[0] - half_side_length, center[1] + half_side_length)  # Top left
            ]

            # Generate points along each side of the rectangle
            boundary_points = []
            for i in range(4):
                start_point = vertices[i]
                end_point = vertices[(i + 1) % 4]
                side_points = np.linspace(start_point, end_point, num_points_per_side + 1)
                if i != 0:  # Exclude the first point of subsequent sides to avoid duplicates
                    side_points = side_points[1:]
                boundary_points.extend(side_points[:-1].tolist())  # Exclude the last point to avoid duplicates

            # Add the first vertex again to close the polygon
            boundary_points.append(vertices[0])

            return boundary_points

        pboundary = generate_square_boundary_points([-0.7, -0.5], 0.4, batch_shape[0] * batch_shape[1])
        x_filtered = RectangularDomainND.foo(pboundary, xmin[2], xmax[2], batch_shape[2])

        return jnp.array(x_filtered)

    @staticmethod
    def boundary_triangle(key, sampler, xmin, xmax, batch_shape):
        def generate_triangle_boundary_points(center, side_length, num_points_per_side):
            half_side_length = side_length / 2
            height = np.sqrt(side_length ** 2 - half_side_length ** 2)

            # 定义三角形的三个顶点
            vertices = [
                (center[0] - half_side_length, center[1] - height / 3),  # 左下
                (center[0] + half_side_length, center[1] - height / 3),  # 右下
                (center[0], center[1] + 2 * height / 3)  # 顶部
            ]
            # 生成三角形每条边的点
            boundary_points = []
            for i in range(3):
                start_point = vertices[i]
                end_point = vertices[(i + 1) % 3]
                side_points = np.linspace(start_point, end_point, num_points_per_side + 1)
                boundary_points.extend(side_points[:-1].tolist())  # 排除最后一个点以避免重复

            return boundary_points
        # 根据批次形状生成边界点
        pboundary = generate_triangle_boundary_points([0, -0.5], 0.5,
                                                      batch_shape[0] * batch_shape[1])

        x_filtered = RectangularDomainND.foo(pboundary, xmin[2], xmax[2], batch_shape[2])

        return jnp.array(x_filtered)

    @staticmethod
    def boundary_heart(key, sampler, xmin, xmax, batch_shape):
        scale = 0.1  # 缩放比例
        def generate_heart_boundary_points(center, scale, num_points):
            t = np.linspace(0, 2 * np.pi, num_points)
            x = scale * 16 * np.sin(t) ** 3
            y = scale * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
            x += center[0]
            y += center[1]
            return np.column_stack((x, y))

        # 根据批次形状生成边界点
        num_boundary_points = batch_shape[0] * batch_shape[1]
        pboundary = generate_heart_boundary_points([0.5, -0.5], scale, num_boundary_points)
        x_filtered = RectangularDomainND.foo(pboundary.tolist(), xmin[2], xmax[2], batch_shape[2])

        return jnp.array(x_filtered)
    @staticmethod
    def all_boundary(key, sampler, xmin, xmax, batch_shape):
        circle_points = RectangularDomainND.boundary_circle(key, sampler, xmin, xmax, batch_shape)
        rectangle_points = RectangularDomainND.boundary_rectangle(key, sampler, xmin, xmax, batch_shape)
        triangle_points = RectangularDomainND.boundary_triangle(key, sampler, xmin, xmax, batch_shape)
        heart_points = RectangularDomainND.boundary_heart(key, sampler, xmin, xmax, batch_shape)
        # all_boundary_points = np.concatenate([circle_points, rectangle_points, triangle_points, heart_points], axis=0)
        all_boundary_points = np.concatenate([circle_points, rectangle_points, triangle_points], axis=0)
        # # 绘制边界点
        # plt.figure(figsize=(8, 8))
        # for boundary_type, color in zip(['rectangle', 'circle', 'triangle', 'heart'],
        #                                 ['red', 'blue', 'green', 'purple']):
        #     boundary_func = getattr(RectangularDomainND, f'boundary_{boundary_type}')
        #     points = boundary_func(None, None, xmin, xmax, batch_shape)
        #     plt.scatter(points[:, 0], points[:, 1], color=color, label=boundary_type)
        #
        # plt.legend()
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Boundary Points')
        # plt.axis('equal')
        # plt.show()
        return all_boundary_points

    # @staticmethod
    # def is_inside(point, boundary_polygon):
    #     shapely_point = Point(point)
    #     return shapely_point.within(boundary_polygon) or shapely_point.touches(boundary_polygon)
    #
    # @staticmethod
    # def filter_points(points, boundary_points):
    #     boundary_polygon = Polygon(boundary_points)
    #
    #     def filter_point(point):
    #         return not RectangularDomainND.is_inside(point[:2], boundary_polygon)
    #
    #     results = [filter_point(point) for point in points]
    #     filtered_points = [point for point, keep in zip(points, results) if keep]
    #
    #     return np.array(filtered_points)
    @staticmethod
    def is_inside(point, boundary_polygon):
        shapely_point = Point(point)
        return shapely_point.within(boundary_polygon) or shapely_point.touches(boundary_polygon)

    @staticmethod
    def filter_points(points, boundary_points, max_workers=4):
        boundary_polygon = Polygon(boundary_points)

        def filter_point(point):
            return not RectangularDomainND.is_inside(point[:2], boundary_polygon)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(filter_point, points))

        filtered_points = [point for point, keep in zip(points, results) if keep]
        return np.array(filtered_points)
    @staticmethod
    def _rectangle_interior_depec(key, sampler, xmin, xmax, batch_shape):
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

        circle_points = RectangularDomainND.boundary_circle(key, sampler, xmin, xmax, batch_shape)
        rectangle_points = RectangularDomainND.boundary_rectangle(key, sampler, xmin, xmax, batch_shape)
        triangle_points = RectangularDomainND.boundary_triangle(key, sampler, xmin, xmax, batch_shape)

        filtered_circle_points = RectangularDomainND.filter_points(x_batch, circle_points)
        filtered_rectangle_points = RectangularDomainND.filter_points(filtered_circle_points, rectangle_points)
        filtered_triangle_points = RectangularDomainND.filter_points(filtered_rectangle_points, triangle_points)
        # # 绘制这些点
        # plt.figure(figsize=(8, 6))
        # plt.scatter(filtered_points[:, 0], filtered_points[:, 1])
        # plt.title('Filtered Points Visualization')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.grid(True)
        # plt.show()
        return filtered_triangle_points

    @staticmethod
    def _rectangle_start_depec(key, sampler, xmin, xmax, batch_shape):
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

        circle_points = RectangularDomainND.boundary_circle(key, sampler, xmin, xmax, batch_shape)
        rectangle_points = RectangularDomainND.boundary_rectangle(key, sampler, xmin, xmax, batch_shape)
        triangle_points = RectangularDomainND.boundary_triangle(key, sampler, xmin, xmax, batch_shape)

        filtered_circle_points = RectangularDomainND.filter_points(x_batch, circle_points)
        filtered_rectangle_points = RectangularDomainND.filter_points(filtered_circle_points, rectangle_points)
        filtered_triangle_points = RectangularDomainND.filter_points(filtered_rectangle_points, triangle_points)
        # # 绘制这些点
        # plt.figure(figsize=(8, 6))
        # plt.scatter(filtered_points[:, 0], filtered_points[:, 1])
        # plt.title('Filtered Points Visualization')
        # plt.xlabel('X Coordinate')
        # plt.ylabel('Y Coordinate')
        # plt.grid(True)
        # plt.show()
        return filtered_triangle_points

    @staticmethod
    def _rectangle_boundary_pec(key, sampler, xmin, xmax, batch_shape):
        boundary_points = RectangularDomainND.all_boundary(key, sampler, xmin, xmax, batch_shape)
        return boundary_points

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





