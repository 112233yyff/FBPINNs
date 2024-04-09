"""
Defines active schedulers

Active schedulers are iterables which allow us to define which FBPINN subdomains
are active/fixed at each training step.

Each scheduler must inherit from the ActiveScheduler base class.
Each scheduler must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""
# 定义激活调度器
#
# 激活调度器是可迭代对象，允许我们定义在每个训练步骤中哪些FBPINN子域是激活的/固定的。
#
# 每个调度器都必须继承自ActiveScheduler基类。
# 每个调度器都必须定义NotImplemented方法。
#
# 此模块被constants.py（进而被trainers.py）使用。

import numpy as np


class ActiveScheduler:
    """Base scheduler class to be inherited by different schedulers"""

    def __init__(self, all_params, n_steps):
        self.n_steps = n_steps
        self.m = all_params["static"]["decomposition"]["m"]
        self.xd = all_params["static"]["decomposition"]["xd"]
    # __init__方法：这是类的构造函数，用于初始化对象。它接收两个参数：all_params 和 n_steps。
    # all_params 是一个包含所有相关参数的字典，特别是关于子域分解的信息（如子域的数量m和维度xd）。
    # n_steps表示训练过程中的总步数。

    def __len__(self):
        return self.n_steps
    # 返回训练步数n_steps，即对象代表的训练过程中总的迭代步数。

    def __iter__(self):
        """
        Returns None if active array not to be changed, otherwise active array.
        active is an array of length m, where each value corresponds
        to the state of each model (i.e. subdomain), which can be one of:

        0 = inactive (but still trained if it overlaps with active models)
        1 = active
        2 = fixed
        """
        # 如果活动数组不需要更改，则返回None，否则返回活动数组。
        # 活动数组的长度为m，其中每个值对应于每个模型（即子域）的状态，可以是以下之一：
        #
        # 0 = 未激活（但如果与活动模型重叠，则仍会受到训练）
        # 1 = 激活
        # 2 = 固定
        # 这个方法使得对象成为一个可迭代对象。这意味着你可以在这个类的实例上使用for 循环。
        # 此方法的实现应该返回一个迭代器，该迭代器在每一步训练中定义哪些子域是活跃的、固定的或不活跃的。
        #
        # 如果某一步骤中活动数组（定义每个子域状态的数组）不需要更改，则该迭代器应该返回None。
        # 否则，它应返回一个长度为m的数组，其中每个元素表示相应子域的状态：
        # 0表示子域不活跃但如果与活动模型重叠则仍会接受训练，
        # 1表示子域是活跃的，
        # 2表示子域是固定的。
        raise NotImplementedError


class AllActiveSchedulerND(ActiveScheduler):
    "All models are active and training all of the time"
    #“所有模型始终处于激活状态并参与训练”。

    def __iter__(self):
        for i in range(self.n_steps):
            if i == 0:
                yield np.ones(self.m, dtype=int)
            #如果是第一步（i == 0），则生成（yield）一个长度为 self.m 的数组，数组中的每个元素都是 1，表示所有的模型在第一步训练时都是激活状态。
            else:
                yield None
            #对于第一步之后的每一步，生成（yield）None，表示在这些步骤中不改变模型的激活状态。


class _SubspacePointSchedulerRectangularND(ActiveScheduler):
    "Slowly expands radially outwards from a point in a subspace of a rectangular domain (in x units)"
    #从矩形域的子空间中的一个点开始，慢慢地向外径向扩展（以x单位计）。
    #段代码定义了一个用于生成激活/固定子域的调度器。它会从一个在矩形域的子空间中的点，逐渐向外扩展半径，根据不同的步骤生成激活的子域。
    def __init__(self, all_params, n_steps, point, iaxes):
        #这个类继承自ActiveScheduler，其目的是控制在一个矩形域的子空间中，从一个给定点开始，慢慢向外扩展的过程。
        #具体来说，这个过程可以用于确定哪些子域（或模型）在训练过程中应当被激活、固定或保持不活跃。
        super().__init__(all_params, n_steps)

        point = np.array(point)# point in constrained axes
        iaxes = list(iaxes)# unconstrained axes

        # validation
        if point.ndim != 1: raise Exception("ERROR: point.ndim != 1")
        if len(point) > self.xd: raise Exception("ERROR: len(point) > self.xd")
        if len(iaxes) + len(point) != self.xd: raise Exception("ERROR: len(iaxes) + len(point) != self.xd")

        # set set attributes
        self.point = point# (cd)
        self.iaxes = iaxes# (ucd)

        # get xmins, xmaxs
        self.xmins0 = all_params["static"]["decomposition"]["xmins0"].copy()# (m, xd)
        self.xmaxs0 = all_params["static"]["decomposition"]["xmaxs0"].copy()# (m, xd)
    #总结来说，这个构造函数初始化了一个在矩形域的子空间中的点周围逐步扩展的调度器。
    # 它通过验证输入的点和轴的维度和长度，然后将这些输入及子域的边界存储为类属性，为之后的迭代操作做准备。

    def _get_radii(self, point, xmins, xmaxs):
        "Get the shortest distance from a point to a hypperectangle"

        # make sure subspace dimensions match with point
        assert xmins.shape[1] == xmaxs.shape[1] == point.shape[0]

        # broadcast point
        point = np.expand_dims(point, axis=0)# (1, cd)

        # whether point is inside model
        c_inside = (point >= xmins) & (point <= xmaxs)# (m, cd) point is broadcast
        c_inside = np.product(c_inside, axis=1).astype(bool)# (m) must be true across all dims

        # get closest point on rectangle to point
        pmin = np.clip(point, xmins, xmaxs)# (m, cd) point is broadcast

        # get furthest point on rectangle to point
        dmin, dmax = point-xmins, point-xmaxs# (m, cd) point is broadcast
        ds = np.stack([dmin, dmax], axis=0)# (2, m, cd)
        i = np.argmax(np.abs(ds), axis=0, keepdims=True)# (1, m, cd)
        pmax = point-np.take_along_axis(ds, i, axis=0)[0]# (m, cd) point is broadcast

        # get radii
        rmin = np.sqrt(np.sum((pmin-point)**2, axis=1))# (m) point is broadcast
        rmax = np.sqrt(np.sum((pmax-point)**2, axis=1))# (m) point is broadcast

        # set rmin=0 where point is inside model
        rmin[c_inside] = 0.

        return rmin, rmax
    #它的目的是计算一个点到一个或多个高维矩形（hyperrectangle）的最短和最长距离。
    # 这个方法在数学和空间分析中特别有用，尤其是在确定点相对于某个空间区域（如模型的边界）的位置时。
    #_get_radii方法能够为给定点与一组矩形边界之间的空间关系提供详细的量化描述

    def __iter__(self):

        # slice constrained axes
        ic = [i for i in range(self.xd) if i not in self.iaxes]
        xmins, xmaxs = self.xmins0[:,ic], self.xmaxs0[:,ic]# (m, cd)

        # get subspace radii
        rmin, rmax = self._get_radii(self.point, xmins, xmaxs)
        r_min, r_max = rmin.min(), rmax.max()

        # initialise active array, start scheduling
        active = np.zeros(self.m, dtype=int)# (m)
        for i in range(self.n_steps):

            # advance radius
            rt = r_min + (r_max-r_min)*(i/(self.n_steps))

            # get filters
            c_inactive = (active == 0)
            c_active   = (active == 1)# (m) active filter
            c_radius = (rt >= rmin) & (rt < rmax)# (m) circle inside box
            c_to_active = c_inactive & c_radius# c_radius is broadcast
            c_to_fixed = c_active & (~c_radius)# c_radius is broadcast

            # set values
            if c_to_active.any() or c_to_fixed.any():
                active[c_to_active] = 1
                active[c_to_fixed] = 2
                yield active
            else:
                yield None

    #这个方法允许逐步探索一个矩形领域，通过逐渐扩大从一个中心点向外的搜索半径，动态地调整点的活跃状态。
    #这种方法特别适用于需要基于距离或相对位置逐步调整参数或元素状态的场景，如优化问题、空间分析或逐步探索空间。

class PointSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a point in the domain (in x units)"

    def __init__(self, all_params, n_steps, point):
        xd = all_params["static"]["decomposition"]["xd"]
        if len(point) != xd: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=[])
    # PointSchedulerRectangularND的目的是在一个多维矩形域内，从一个指定点开始，缓慢向外扩展（在x单位中度量）。
    # 这个类特别适用于需要从某个中心点开始，逐渐探索周围空间的情况，比如在参数空间中寻找最优点或者在物理空间中模拟扩散过程。

##########################################
#_SubspacePointSchedulerRectangularND 更加专注于在矩形域内的特定子空间中进行扩展，
#而PointSchedulerRectangularND 则是在整个矩形域内自由地进行扩展，不受任何限制。







class LineSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a line in the domain (in x units)"
    #从矩形域中的一条线开始，逐步向外扩展。

    def __init__(self, all_params, n_steps, point, iaxis):
        xd = all_params["static"]["decomposition"]["xd"]
        if xd < 2: raise Exception("ERROR: requires nd >=2")
        if len(point) != xd-1: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=[iaxis])
    #def init(self, all_params, n_steps, point, iaxis):：
    # 这是类的初始化方法，它接受四个参数：all_params、n_steps、point和iaxis。
    # all_params是一个包含所有参数的字典，用于初始化优化器等。
    # n_steps是扩展过程的总步数。
    # point是线的起始点坐标。
    # iaxis是一个整数，指定在哪个维度上的线进行扩展。

class PlaneSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a plane in the domain (in x units)"
    #从域中的一个平面慢慢向外扩展（以 x 单位计）

    def __init__(self, all_params, n_steps, point, iaxes):
        xd = all_params["static"]["decomposition"]["xd"]
        if xd < 3: raise Exception("ERROR: requires nd >=3")
        if len(point) != xd-2: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=iaxes)
    #在类的 __init__ 方法中：
    # 首先，从 all_params 中获取域的维度信息 xd。
    # 然后，检查域的维度是否大于等于 3，如果不是，则引发异常。
    # 接着，检查输入的点 point 的形状是否正确，如果不是，则引发异常。
    # 最后，调用父类的 __init__ 方法来初始化该类。
    # 总之，这个类的作用是定义了一个可以在域中从一个平面开始慢慢向外扩展的调度器。



if __name__ == "__main__":

    from fbpinns.decompositions import RectangularDecompositionND

    x = np.array([-6,-4,-2,0,2,4,6])

    subdomain_xs1 = [x]
    d1 = RectangularDecompositionND
    ps_ = d1.init_params(subdomain_xs1, [3*np.ones_like(x) for x in subdomain_xs1], (0,1))
    all_params1 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs1)}

    subdomain_xs2 = [x, x]
    d2 = RectangularDecompositionND
    ps_ = d2.init_params(subdomain_xs2, [3*np.ones_like(x) for x in subdomain_xs2], (0,1))
    all_params2 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs2)}

    subdomain_xs3 = [x, x, x]
    d3 = RectangularDecompositionND
    ps_ = d3.init_params(subdomain_xs3, [3*np.ones_like(x) for x in subdomain_xs3], (0,1))
    all_params3 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs3)}

    # test point
    for all_params in [all_params1, all_params2, all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Point")
        point = np.array([0]*xd)
        A = PointSchedulerRectangularND(all_params, 100, point)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()

    # test line
    for all_params in [all_params2, all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Line")
        point = np.array([0]*(xd-1))
        A = LineSchedulerRectangularND(all_params, 100, point, 0)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()

    # test plane
    for all_params in [all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Plane")
        point = np.array([0]*(xd-2))
        A = PlaneSchedulerRectangularND(all_params, 100, point, [0,1])
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()

