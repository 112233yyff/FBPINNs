"""
Defines trainer classes for FBPINNs and PINNs.

This is the main entry point for training FBPINNs and PINNs.

To train a FBPINN / PINN, use a Constants object to set up the problem and define its hyperparameters, and pass it
to one of the trainer classes defined here
"""

import time, pdb
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad, jvp
from jax import random
import optax
import numpy as np

from fbpinns.trainers_base import _Trainer
from fbpinns import networks, plot_trainer
from fbpinns.util.logger import logger
from fbpinns.util.jax_util import tree_index, total_size, str_tensor, partition, combine
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# LABELLING CONVENTIONS

# xd = dimensionality of point
# ud = dimensionality of solution
# dims = (ud, xd)
# n = number of points
# m = number of models (i.e. subdomains)
# c = number of constraints

# x = single coordinate (xd)
# x_batch = batch of coordinates (n, xd)
# uj = solution and gradient component list

# j = index in uj
# im = index of model
# ip = index of point
# ic = index of constraint
# i = generic index

# nm = shape of rectangular DDs
# ii = for grid index in nm


# STATIC FUNCTIONS

def tree_map_dicts(f, *trees):
    "Apply function to top-level dicts in tree(s)"

    is_dict = lambda x: isinstance(x, dict)

    def apply(leaf, *leaves):
        if is_dict(leaf):  # if top-level dict
            return f(leaf, *leaves)
        else:
            return leaf  # if leaf (i.e. at bottom of tree), return first tree's leaf only (!)

    return jax.tree_util.tree_map(apply, *trees, is_leaf=is_dict)  # stop traverse on top-level dicts


def get_jmaps(required_ujs):
    "Generate tree for computing chained jacobians"

    logger.debug("get_jmaps")

    # build tree of required gradients
    tree = {}
    for iu, ixs in required_ujs:
        t = tree
        for ix in ixs:
            if ix not in t:
                t[ix] = {}
            t = t[ix]

    # parse tree nodes
    def get_nodes(t, n=(), ks=()):
        ni = len(n) - 1 + 1  # index of parent node (including u at start)
        for k in t:
            ks_ = ks + (k,)
            if t[k]:
                n += (((ni, k), ks_, 0),)  # node
                n = get_nodes(t[k], n, ks_)
            else:
                n += (((ni, k), ks_, 1),)  # leaf
        return n

    # list of chained grad functions
    nodes = get_nodes(tree)
    logger.debug(nodes)

    # list of grad functions to evaluate
    leaves = tuple((i + 1, node[1]) for i, node in enumerate(nodes) if node[2])
    if not leaves:
        leaves = ((0, ()),)  # special case where only solution required. tree/nodes are empty in this case
    logger.debug(leaves)

    # get map between required_ujs and list of chained gradients
    jac_is = ()  # il (leaf index), io (order index), iu (u index)
    for iu, ixs in required_ujs:
        io = len(ixs)
        il = [leaf[1][:io] for leaf in leaves].index(ixs)  # also works for 0,()
        jac_is += ((il, io, iu),)
    logger.debug(jac_is)

    return nodes, leaves, jac_is


# JITTED FUNCTIONS

# def FBPINN_model_inner(params, x, mask,norm_fn, network_fn, unnorm_fn, window_fn):
#     x_norm = norm_fn(params, x)# normalise
#     u_raw = network_fn(params, x_norm, mask)# network
#     u = unnorm_fn(params, u_raw)# unnormalise
#     w = window_fn(params, x)# window
#     #window_fn 函数的作用是基于给定的参数 params 和输入 x
#     #返回一个加权和的结果，其中权重由 params[4] 控制。
#     return u*w, w, u_raw
def FBPINN_model_inner(params, x, norm_fn, network_fn, unnorm_fn, window_fn):
    x_norm = norm_fn(params, x)  # normalise
    u_raw = network_fn(params, x_norm)  # network
    u = unnorm_fn(params, u_raw)  # unnormalise
    w = window_fn(params, x)  # window
    return u * w, w, u_raw


def PINN_model_inner(all_params, x, norm_fn, network_fn, unnorm_fn):
    x_norm = norm_fn(all_params, x)  # normalise
    u_raw = network_fn(all_params, x_norm)  # network
    u = unnorm_fn(u_raw)  # unnormalise
    return u, u_raw


# def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
#     "Defines FBPINN model"
#
#     norm_fn, network_fn, unnorm_fn, window_fn, constraining_fn = model_fns
#     m_take, n_take, p_take, np_take, npou = takes
#
#     # take x_batch
#     x_take = x_batch[n_take]# (s, xd)
#     log_ = logger.info if verbose else logger.debug
#     log_("x_batch")
#     log_(str_tensor(x_batch))# (n, xd)
#     log_("x_take")
#     log_(str_tensor(x_take))
#
#     # take subdomain params
#     d = all_params
#     all_params_take = {t_k: {cl_k: {k: jax.tree_map(lambda p:p[m_take], d[t_k][cl_k][k]) if k=="subdomain" else d[t_k][cl_k][k]
#         for k in d[t_k][cl_k]}
#         for cl_k in d[t_k]}
#         for t_k in ["static", "trainable"]}
#     f = {t_k: {cl_k: {k: jax.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k=="subdomain" else jax.tree_map(lambda p: None, d[t_k][cl_k][k])
#         for k in d[t_k][cl_k]}
#         for cl_k in d[t_k]}
#         for t_k in ["static", "trainable"]}
#     logger.debug("all_params")
#     logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))
#     logger.debug("all_params_take")
#     logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params_take))
#     logger.debug("vmap f")
#     logger.debug(f)
#     # 创建一个与 m_take 相同长度的零数组
#     mask = jnp.zeros_like(m_take, dtype=int)
#     # 将 m_take 中值为2的位置设为1
#     mask = jnp.where(m_take == 3, 1, mask)
#
#     # jax.debug.print("ret {}", mask)
#     # us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f, 0, 0, None, None, None, None))(all_params_take, x_take, mask,
#     #                                                                                      norm_fn, network_fn, unnorm_fn, window_fn)  # (s, ud)
#     us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f, 0, None, None, None, None))(all_params_take, x_take, norm_fn, network_fn, unnorm_fn, window_fn)
#
#     #思路：先判别点是属于哪个子域的，进入到if-else语句中，然后再利用vmap函数
#     #unnormalise * window  window  network
#     # us = u*w（在求和之前，每个网络被一个平滑的、可微的窗口函数相乘，该窗口函数将其局部限制在它的子区域内
#     # ws = w（原始的x经过窗函数处理的值，更像是一个系数）
#     # us_raw = u_raw(经过标准化然后神经网络输出的值）
#     logger.debug("u")
#     logger.debug(str_tensor(us))
#     # apply POU and sum
#     u = jnp.concatenate([us, ws], axis=1)# (s, ud+1)
#     # unnormalise * window  window
#     # 其中 s 是样本数量，ud 是数据维度。
#     u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))# (_, ud+1)
#     #p_take = Array([0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9], dtype=int32)
#     #np_take = Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
#     # 这行代码应用了segment_sum操作，它根据p_take数组中的索引将u中的元素分组求和。
#     # p_take指定了每个元素属于哪个子域，num_segments=len(np_take)指定了总共有多少个不同子域。
#     wp = u[:,-1:]
#     #将数组 u 的最后一列提取出来,最后一列是window
#     u = u[:,:-1]/wp
#     #这一行代码是将数组 u 中除了最后一列之外的所有列除以 wp，也就是unnormalise * window / window
#     logger.debug(str_tensor(u))
#     u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))# (n, ud)
#     #np_take = Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
#     logger.debug(str_tensor(u))
#     u = u/npou
#     ##npou：1
#     #npou 计算了所有不同处理单元索引的数量，因此它代表了总的处理单元数量。
#     logger.debug(str_tensor(u))
#
#     # then apply constraining operator
#     u = constraining_fn(all_params, x_batch, u)# (n, ud)
#     # logger.debug(str_tensor(u))
#
#     return u, wp, us, ws, us_raw
def FBPINN_model(all_params, x_batch, takes, model_fns, verbose=True):
    "Defines FBPINN model"

    norm_fn, network_fn, unnorm_fn, window_fn, constraining_fn = model_fns
    m_take, n_take, p_take, np_take, npou = takes

    # take x_batch
    x_take = x_batch[n_take]  # (s, xd)
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))  # (n, xd)
    log_("x_take")
    log_(str_tensor(x_take))

    # take subdomain params
    d = all_params
    all_params_take = {
        t_k: {cl_k: {k: jax.tree_map(lambda p: p[m_take], d[t_k][cl_k][k]) if k == "subdomain" else d[t_k][cl_k][k]
                     for k in d[t_k][cl_k]}
              for cl_k in d[t_k]}
        for t_k in ["static", "trainable"]}
    f = {t_k: {cl_k: {
        k: jax.tree_map(lambda p: 0, d[t_k][cl_k][k]) if k == "subdomain" else jax.tree_map(lambda p: None,
                                                                                            d[t_k][cl_k][k])
        for k in d[t_k][cl_k]}
               for cl_k in d[t_k]}
         for t_k in ["static", "trainable"]}
    logger.debug("all_params")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))
    logger.debug("all_params_take")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params_take))
    logger.debug("vmap f")
    logger.debug(f)

    # batch over parameters and points
    us, ws, us_raw = vmap(FBPINN_model_inner, in_axes=(f, 0, None, None, None, None))(all_params_take, x_take, norm_fn,
                                                                                      network_fn, unnorm_fn,
                                                                                      window_fn)  # (s, ud)
    # us是神经网络输出的N*unnorm*窗函数w
    # ws是窗函数
    # us_raw是神经网络原始输出的N
    logger.debug("u")
    logger.debug(str_tensor(us))
    # apply POU and sum
    u = jnp.concatenate([us, ws], axis=1)  # (s, ud+1)
    u = jax.ops.segment_sum(u, p_take, indices_are_sorted=False, num_segments=len(np_take))  # (_, ud+1)
    # p_take：点与POU组合的唯一索引。点在原批次中的索引。
    # np_take：点的次序索引。
    #     #p_take = Array([0, 1, 2, 3, 4, 4, 5, 5, 6, 7, 8, 9], dtype=int32)
    #     #np_take = Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)
    #     # 这行代码应用了segment_sum操作，它根据p_take数组中的索引将u中的元素分组求和。
    wp = u[:, -1:]
    u = u[:, :-1] / wp
    logger.debug(str_tensor(u))
    u = jax.ops.segment_sum(u, np_take, indices_are_sorted=False, num_segments=len(x_batch))  # (n, ud)
    logger.debug(str_tensor(u))
    u = u / npou
    logger.debug(str_tensor(u))

    # then apply constraining operator
    # u = constraining_fn(all_params, x_batch, u)# (n, ud)
    logger.debug(str_tensor(u))

    return u, wp, us, ws, us_raw


def PINN_model(all_params, x_batch, model_fns, verbose=True):
    "Defines PINN model"

    norm_fn, network_fn, unnorm_fn, constraining_fn = model_fns
    log_ = logger.info if verbose else logger.debug
    log_("x_batch")
    log_(str_tensor(x_batch))  # (n, xd)

    # batch over parameters and points
    u, u_raw = vmap(PINN_model_inner, in_axes=(None, 0, None, None, None))(all_params, x_batch, norm_fn, network_fn,
                                                                           unnorm_fn)  # (n, ud)
    logger.debug("u")
    logger.debug(str_tensor(u))

    # then apply constraining operator
    u = constraining_fn(all_params, x_batch, u)

    return u, u_raw


def FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps):
    "Computes gradients of FBPINN model"

    # isolate model function
    def u(x_batch):
        return FBPINN_model(all_params, x_batch, takes, model_fns)[0], ()

    return _get_ujs(x_batch, jmaps, u)


def PINN_forward(all_params, x_batch, model_fns, jmaps):
    "Computes gradients of PINN model"

    # isolate model function
    def u(x_batch):
        return PINN_model(all_params, x_batch, model_fns)[0], ()

    return _get_ujs(x_batch, jmaps, u)


def _get_ujs(x_batch, jmaps, u):
    nodes, leaves, jac_is = jmaps
    vs = jnp.tile(jnp.eye(x_batch.shape[1]), (x_batch.shape[0], 1, 1))

    # chain required jacobian functions
    fs = [u]
    for (ni, ix), _, _ in nodes:
        fs.append(jacfwd(fs[ni], vs[:, ix]))

    # evaluate required jacobian functions
    jacs = []
    for ie, _ in leaves:
        fin, jac = fs[ie](x_batch)
        jacs.append(jac + (fin,))

    # index required jacobians
    ujs = [jacs[il][io][:, iu:iu + 1] for il, io, iu in jac_is]

    logger.debug("fs")
    logger.debug(fs)
    logger.debug("jacs")
    for jac in jacs: logger.debug([j.shape for j in jac])  # (n, ud)
    logger.debug("ujs")
    for uj in ujs: logger.debug(str_tensor(uj))

    return ujs


def jacfwd(f, v):
    "Computes jacobian for single x, for all y, fully chained"

    def jacfun(x):
        y, j, aux = jvp(f, (x,), (v,), has_aux=True)
        aux = aux + (y,)
        return j, aux

    return jacfun


def FBPINN_loss(active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn):
    # add fixed params to active, recombine all_params
    d, da = active_params, fixed_params
    trainable_params = {cl_k: {
        k: jax.tree_map(lambda p1, p2: jnp.concatenate([p1, p2], 0), d[cl_k][k], da[cl_k][k]) if k == "subdomain" else
        d[cl_k][k]
        for k in d[cl_k]}
                        for cl_k in d}
    all_params = {"static": static_params, "trainable": trainable_params}

    # run FBPINN for each constraint, with shared params
    constraints_ = []
    for takes, jmaps, constraint in zip(takess, jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = FBPINN_forward(all_params, x_batch, takes, model_fns, jmaps)
        constraints_.append(constraint + ujs)
    return loss_fn(all_params, constraints_)


def PINN_loss(active_params, static_params, constraints, model_fns, jmapss, loss_fn):
    # recombine all_params
    all_params = {"static": static_params, "trainable": active_params}

    # run PINN for each constraint, with shared params
    constraints_ = []
    for jmaps, constraint in zip(jmapss, constraints):
        logger.debug("constraint")
        for c_ in constraint:
            logger.debug(str_tensor(c_))
        x_batch = constraint[0]
        ujs = PINN_forward(all_params, x_batch, model_fns, jmaps)
        constraints_.append(constraint + ujs)
    return loss_fn(all_params, constraints_)


@partial(jit, static_argnums=(0, 5, 8, 9, 10))
def FBPINN_update(optimiser_fn, active_opt_states,
                  active_params, fixed_params, static_params_dynamic, static_params_static,
                  takess, constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(FBPINN_loss, argnums=0)(
        active_params, fixed_params, static_params, takess, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    #updates: 这是由优化算法根据损失函数相对于模型参数的梯度计算得出的更新值。换句话说，updates包含了如何改变active_params以减小损失函数的指导信息。这些更新通常与active_params的结构一一对应，即每个参数都有一个对应的更新值。
    return lossval, active_opt_states, active_params


@partial(jit, static_argnums=(0, 4, 6, 7, 8))
def PINN_update(optimiser_fn, active_opt_states,
                active_params, static_params_dynamic, static_params_static,
                constraints, model_fns, jmapss, loss_fn):
    # recombine static params
    static_params = combine(static_params_dynamic, static_params_static)
    # update step
    lossval, grads = value_and_grad(PINN_loss, argnums=0)(
        active_params, static_params, constraints, model_fns, jmapss, loss_fn)
    updates, active_opt_states = optimiser_fn(grads, active_opt_states, active_params)
    active_params = optax.apply_updates(active_params, updates)
    return lossval, active_opt_states, active_params


# For fast test inference only

@partial(jax.jit, static_argnums=(1, 4, 5))
def _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return FBPINN_model(all_params, x_batch, takes, model_fns, verbose)


def FBPINN_model_jit(all_params, x_batch, takes, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _FBPINN_model_jit(all_params_dynamic, all_params_static, x_batch, takes, model_fns, verbose)


@partial(jax.jit, static_argnums=(1, 3, 4))
def _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose):
    all_params = combine(all_params_dynamic, all_params_static)
    return PINN_model(all_params, x_batch, model_fns, verbose)


def PINN_model_jit(all_params, x_batch, model_fns, verbose=True):
    all_params_dynamic, all_params_static = partition(all_params)
    return _PINN_model_jit(all_params_dynamic, all_params_static, x_batch, model_fns, verbose)


def get_inputs(x_batch, active, all_params, decomposition):
    "Get the inputs to the FBPINN model based on x_batch and the active models"
    # 基于 `x_batch` 和活跃模型，获取 FB-PINN 模型的输入数据。

    # get the ims inside x_batch
    n_take, m_take, training_ims = decomposition.inside_points(all_params, x_batch)  # (nc, m)
    # 函数返回三个结果：
    # n_take：点在原批次中的索引（take[:,0]）
    # len(n_take)>len(x_batch)是因为无论是n_take还是m_take都是看每个点出现在子域的次数
    # 有的点可能会出现两次所以导致维度的扩张
    # m_take:对应的子域索引（take[:,1]），
    # training_ims:以及至少有一个点在其内的子域索引（inside_ims）。
    # 输入数据批次 x_batch 中每个点属于哪些子域

    # get active_ims and fixed_ims
    # scheduler should return
    # 0 = inactive (but still trained if it intersects with active models)
    # 1 = active
    # 2 = fixed
    # now modify active to
    # 0 = discard (not in current training points)
    # 1 = active
    # 2 = fixed
    active = jnp.array(active).copy()
    assert jnp.isin(active, jnp.array([0, 1, 2])).all()
    assert active.shape == (all_params["static"]["decomposition"]["m"],)

    active = active.at[active == 0].set(1)  # set inactive models to active
    mask = jnp.zeros_like(active)  # mask out models in training points
    mask = mask.at[training_ims].set(1)
    active = active * mask
    ims_ = jnp.arange(all_params["static"]["decomposition"]["m"])
    active_ims = ims_[active == 1]  # assume unsorted
    fixed_ims = ims_[active == 2]
    logger.debug("updated active")
    logger.debug(active)
    logger.debug("active_ims")
    logger.debug(active_ims)
    logger.debug("fixed_ims")
    logger.debug(fixed_ims)
    # 于模型初始状态和当前训练点信息，最终区分并获取了活跃模型和固定模型的索引

    # note, numbers in all_ims == numbers in training_ims == numbers in m_take
    # which also means we need all m_take points above
    all_ims = jnp.concatenate([active_ims, fixed_ims])

    # re-index m_take to all_ims index
    inv = jnp.zeros(all_params["static"]["decomposition"]["m"], dtype=int)
    inv = inv.at[all_ims].set(jnp.arange(len(all_ims)))  # assumes all_ims is unique
    m_take = inv[m_take]

    # (!) note: make sure n_take, pous (and therefore p_take / np_take) are sorted - makes segment_sum quicker
    logger.debug("takes")
    logger.debug(str_tensor(m_take))
    logger.debug(str_tensor(n_take))

    # get POUs
    pous = all_params["static"]["decomposition"]["subdomain"]["pou"][all_ims].astype(int)
    # 这一行从参数字典all_params中抽取了所有活跃和固定模型的POU信息，并将结果存储在pous数组中。
    # all_ims是一个包含了所有当前考虑的模型索引的数组，这里通过索引all_ims来选择出与之相关的POU信息。
    np = jnp.stack([n_take, pous[m_take, 0]], axis=-1).astype(int)  # points and pous
    # 每一行表示一个数据点的索引和其对应的处理单元索引。
    # `pous[m_take, 0]`这个表达式的含义是从名为`pous`的数组中选取特定行和列的元素。
    # - `m_take`是一个索引数组或列表，用于选取`pous`数组的行。- `0`表示选取列的索引，这里始终选择第一列。
    # 因此，`pous[m_take, 0]`的结果是从`pous`数组中选取了`m_take`指定的行，并且选取了这些行的第一列元素。
    # np是一个二维数组（12，2）。第一个坐标表示的是第几个点的索引，第二个坐标表示的是pou，也就是显示单元。
    logger.debug(str_tensor(np))
    npu, p_take = jnp.unique(np, axis=0, return_inverse=True)  # unique points and pous (sorted), point-pou takes
    np_take = npu[:, 0]
    # np_take 包含了所有点的次序的索引
    logger.debug(str_tensor(p_take))
    logger.debug(str_tensor(np_take))
    npou = len(jnp.unique(all_params["static"]["decomposition"]["subdomain"]["pou"].astype(int)))  # global npou
    # npou：1
    # npou 计算了所有不同处理单元索引的数量，因此它代表了总的处理单元数量。
    logger.debug(f"Total number of POUs: {npou}")

    takes = (m_take, n_take, p_take, np_take, npou)

    # m_take：重新索引后的点所属子域的索引。
    # n_take：点在原批次中的索引。
    # p_take：点与POU组合的唯一索引。
    # np_take：所有点的次序索引。
    # npou：总的POU数量。

    # cut active and fixed parameter trees
    def cut_active(d):
        "Cuts active_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p: p[active_ims], d[cl_k][k]) if k == "subdomain" else d[cl_k][k]
                       for k in d[cl_k]}
                for cl_k in d}

    def cut_fixed(d):
        "Cuts fixed_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p: p[fixed_ims], d[cl_k][k]) if k == "subdomain" else d[cl_k][k]
                       for k in d[cl_k]}
                for cl_k in d}

    def cut_all(d):
        "Cuts all_ims from param dict"
        return {cl_k: {k: jax.tree_map(lambda p: p[all_ims], d[cl_k][k]) if k == "subdomain" else d[cl_k][k]
                       for k in d[cl_k]}
                for cl_k in d}

    def merge_active(da, d):
        "Merges active_ims from param dict da to d"
        for cl_k in d:
            for k in d[cl_k]:
                if k == "subdomain":
                    d[cl_k][k] = jax.tree_map(lambda pa, p: p.copy().at[active_ims].set(pa), da[cl_k][k], d[cl_k][k])
                else:
                    d[cl_k][k] = da[cl_k][k]
        return d

    return takes, all_ims, (active, cut_active, cut_fixed, cut_all, merge_active)


def _common_train_initialisation(c, key, all_params, problem, domain):
    # print stats
    logger.info("Total number of trainable parameters:")
    for k in all_params["trainable"]:
        logger.info(f'\t{k}: {total_size(all_params["trainable"][k]):,}')
    # logger.info: 记录日志信息。
    # all_params["trainable"]: 可训练参数的字典。
    # total_size: 计算参数的总大小。

    # initialise optimiser
    optimiser = optax.adam(**c.optimiser_kwargs)
    # 创建一个 Adam 优化器实例，使用 c.optimiser_kwargs 中的参数。
    all_opt_states = optimiser.init(all_params["trainable"])
    # 初始化优化器状态，基于所有可训练参数。
    logger.debug("all_opt_states")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), all_opt_states))
    optimiser_fn, loss_fn = optimiser.update, problem.loss_fn
    # 优化器的更新函数，用于在每次迭代中更新参数。
    # 损失函数，用于计算模型的误差。

    # 优化器是机器学习和深度学习中的一个重要组成部分。
    # 它负责调整模型的可训练参数，以最小化损失函数，从而提高模型的性能。
    # 在神经网络训练过程中，优化器通过使用损失函数的梯度信息来更新模型参数，使模型的预测结果更加准确。

    # get global constraints (training points)
    key, subkey = random.split(key)
    constraints_global = problem.sample_constraints(all_params=all_params, domain=domain, key=subkey, sampler=c.sampler,
                                                    batch_shapes=c.ns, start_batch_shapes=c.n_start,
                                                    boundary_batch_shapes=c.n_boundary)
    # 调用 problem.sample_constraints 方法，生成全局约束。
    # 返回的 constraints_global 是一个列表，每个元素是一个约束。
    for constraint_ in constraints_global:  # 遍历 constraints_global 中的每个约束。constraint_ 是当前遍历的约束。
        for c_ in constraint_[
                  :-1]:  # 遍历 constraint_ 的所有部分，但不包括最后一个部分（通过 [:-1] 实现切片操作）。例如，如果 constraint_ 有三个部分 [a, b, c]，那么 constraint_[:-1] 只包含 [a, b]。
            assert c_.shape[0] == constraint_[0].shape[0]
            # 确认当前部分 c_ 的第一个维度（采样点批次大小）与 constraint_ 的第一个部分（即 constraint_[0]）的第一个维度相同。
    # parse global constraints
    x_batch_global = jnp.concatenate([constraint_[0] for constraint_ in constraints_global])  # (n, xd)ma
    # constraint_[0] 表示每个约束的第一个部分，通常是输入数据 x。
    # jnp.concatenate 将所有约束的输入数据沿第一个维度连接起来，形成一个大的输入批量 x_batch_global。
    # 结果是一个形状为 (n, xd) 的张量
    # 其中 n 是所有约束中样本数量的总和，xd 是输入数据的特征维度。
    constraint_offsets_global = jnp.array([0] + [constraint_[0].shape[0] for constraint_ in constraints_global[:-1]],
                                          dtype=int).cumsum()  # (c,) offset index of each constraint
    # 每个约束列表的偏移量，用于在 x_batch_global 中定位。
    constraint_fs_global = jnp.zeros((x_batch_global.shape[0], len(constraints_global)), dtype=bool)  # (n, c)
    # constraint_fs_global是一个布尔矩阵，表示每个样本属于哪些约束。
    # 矩阵的形状为(n, c)，其中n是样本的总数，c是约束的数量。
    # 初始值全为False，后续代码会填充这个矩阵，将特定位置的值设置为True。
    # 这个矩阵在训练过程中用于指示每个样本和约束的关系，有助于模型根据约束信息进行训练。

    for ic in range(
            len(constraints_global)):  # fill in constraint filters 布尔矩阵用于指示每个样本属于哪些约束，通过遍历 constraints_global 填充矩阵。
        constraint_fs_global = constraint_fs_global.at[
                               constraint_offsets_global[ic]:constraint_offsets_global[ic] +
                                                             constraints_global[ic][0].shape[0], ic].set(True)
    required_ujss = [constraint_[-1] for constraint_ in constraints_global]
    # 从 constraints_global 列表中提取每个约束列表相应的导数约束条件。
    constraints_global = [constraint_[:-1] for constraint_ in constraints_global]
    # 从 constraints_global 列表中提取每个约束条件的除最后一个元素外的部分，并将其组成一个新的列表 constraints_global
    logger.info(f"Total number of constraints: {len(constraints_global)}")
    logger.debug("constraints_global")
    logger.debug(jax.tree_map(lambda x: str_tensor(x), constraints_global))
    logger.debug(constraint_offsets_global)
    logger.debug(str_tensor(constraint_fs_global))
    logger.debug(required_ujss)
    logger.debug("x_batch_global")
    logger.debug(str_tensor(x_batch_global))

    # get jac maps
    jmapss = tuple(get_jmaps(required_ujs) for required_ujs in required_ujss)
    # 根据给定的约束条件required_ujss，使用函数get_jmaps()计算得到对应的Jacobian映射，
    # 并将结果存储在元组jmapss中。每个元素都是一个Jacobian映射。

    # get test points - for now, just use global interior points
    # 只是简单地使用全局内部点作为测试点
    x_batch_test = domain.sample_interior(all_params=all_params, key=None, sampler="grid", batch_shape=c.n_test)
    logger.debug("x_batch_test")
    logger.debug(str_tensor(x_batch_test))

    # get exact solution if it exists
    logger.info("Computing exact solution..")
    u_exact = problem.exact_solution(all_params=all_params, x_batch=x_batch_test, batch_shape=c.n_test)
    logger.info("Computing done")
    logger.debug("u_exact")
    logger.debug(str_tensor(u_exact))

    return (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
            constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
            x_batch_test, u_exact)


class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"

    def _get_x_batch(self, i, active, all_params, x_batch_global, constraints_global, constraint_fs_global,
                     constraint_offsets_global, decomposition):
        "Get the x_batch points from x_batch_global which are inside active models"

        # cut active points out of x_batch_global
        ims = jnp.arange(all_params["static"]["decomposition"]["m"])[active == 1]
        training_ips, _d = decomposition.inside_models(all_params, x_batch_global, ims)  # (n, mc)
        # 该函数 inside_models_batch 的目的是确定输入数据批次 x_batch 中的哪些点位于给定的多个模型（通过 ims 索引表示）内部，
        # 并返回这些点的索引以及每个模型内点的数量。
        x_batch = x_batch_global[training_ips]

        # report
        logger.info(f"[i: {i}/{self.c.n_steps}] Average number of points/dimension in active subdomains: {_d:.2f}")
        logger.debug("x_batch")
        logger.debug(str_tensor(x_batch))

        # cut same points out of constraints_global
        constraint_fs = constraint_fs_global[training_ips]  # for each training point, whether in constraint
        ix_ = jnp.arange(x_batch.shape[0])
        constraint_ips = [ix_[f] for f in constraint_fs.T]  # indices of training points in each constraint
        # 将属于活跃模型的训练点切割为三个部分，因为有三个约束条件
        constraints = [[c_[training_ips[constraint_ips[ic]] - constraint_offsets_global[ic]]
                        for c_ in constraints_global[ic]]
                       for ic in range(len(constraints_global))]  # cut constraints
        logger.debug("constraints")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), constraints))
        logger.debug(jax.tree_map(lambda x: str_tensor(x), constraint_ips))
        logger.debug(str_tensor(constraint_fs))

        return x_batch, constraints, constraint_fs, constraint_ips

    def _get_update_inputs(self, i, active, all_params, all_opt_states, x_batch_global, constraints_global,
                           constraint_fs_global, constraint_offsets_global, decomposition, problem):
        "Get inputs to the FBPINN update step based on active models"

        start0 = time.time()
        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs..")

        # check active
        logger.debug(active)
        active = jnp.array(active).copy()
        assert jnp.isin(active, jnp.array([0, 1, 2])).all()
        assert active.shape == (all_params["static"]["decomposition"]["m"],)

        # get x_batch / constraints inputs from x_batch_global based on active
        x_batch, constraints, constraint_fs, constraint_ips = self._get_x_batch(i, active, all_params, x_batch_global,
                                                                                constraints_global,
                                                                                constraint_fs_global,
                                                                                constraint_offsets_global,
                                                                                decomposition)

        # get model takes / scheduler cuts given x_batch and active
        takes, _, (active, cut_active, cut_fixed, cut_all, merge_active) = get_inputs(x_batch, active, all_params,
                                                                                      decomposition)

        # cut params / opt states (schedule)
        active_params = cut_active(all_params["trainable"])
        fixed_params = cut_fixed(all_params["trainable"])
        static_params = cut_all(all_params["static"])
        active_opt_states = tree_map_dicts(cut_active,
                                           all_opt_states)  # because all_opt_states has more complex structure
        logger.debug("active_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), active_params))
        logger.debug("fixed_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), fixed_params))
        logger.debug("static_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), static_params))
        logger.debug("active_opt_states")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), active_opt_states))

        # split takes into many using constraint_ips (multiple constraints)
        # choice 1: use serial constraints - to stop too many gradients
        # choice 2: combine constraints into joint params/x_batch - 1) for single inside computation 2) needed for joint loss (shared params)
        (m_take, n_take, p_take, np_take, npou) = takes
        takess = []
        iu_ = jnp.arange(np_take.shape[0])
        for f, ips in zip(constraint_fs.T, constraint_ips):
            f1 = f  # cut x_batch
            f2 = f[np_take]  # cut unique point-pou

            # ips = ix_[f1] cut x_batch index (already done above)
            ius = iu_[f2]  # cut unique point-pou index

            f3 = f1[n_take]  # cut n_take
            f4 = f2[p_take]  # cut p_take

            # re-index takes based on cuts
            inv = jnp.zeros(x_batch.shape[0], dtype=int)
            inv = inv.at[ips].set(jnp.arange(len(ips)))

            inv2 = jnp.zeros(np_take.shape[0], dtype=int)
            inv2 = inv2.at[ius].set(jnp.arange(len(ius)))

            m_take_ic = m_take[f3]  # not necessary to re-index as params are not cut per constraint
            n_take_ic = inv[n_take[f3]]

            p_take_ic = inv2[p_take[f4]]
            np_take_ic = inv[np_take[f2]]

            takess.append((m_take_ic, n_take_ic, p_take_ic, np_take_ic, npou))

        logger.info(f"[i: {i}/{self.c.n_steps}] Updating active inputs done ({time.time() - start0:.2f} s)")

        return active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch

    def train(self):
        "Train model"

        c, writer = self.c, self.writer
        # 行代码从类实例（假设是某个类的方法内部）中获取两个属性c和writer，
        # 并将它们赋给同名的局部变量。

        # generate root key
        key = random.PRNGKey(c.seed)
        # 使用来自配置c中的随机种子c.seed初始化一个伪随机数生成器（PRNG）的密钥。
        # 这在随机初始化模型参数时非常关键，确保实验的可重复性。
        np.random.seed(c.seed)
        # 同样地，设置NumPy的随机种子为c.seed，确保NumPy相关的随机操作也是确定性的。

        # define all_params
        all_params = {"static": {}, "trainable": {}}
        # 创建一个字典all_params，它有两个键："static" 和 "trainable"，
        # 分别用来存放不可训练（静态）参数和可训练参数。

        # initialise domain, problem and decomposition params
        domain, problem, decomposition = c.domain, c.problem, c.decomposition
        for tag, cl, kwargs in zip(["domain", "problem", "decomposition"], [domain, problem, decomposition],
                                   [c.domain_init_kwargs, c.problem_init_kwargs, c.decomposition_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        # 对于每一个组件，使用特定的初始化参数调用其初始化方法，得到一组参数，其中可能包含静态和可训练参数。
        # 接下来，根据ps_中返回的参数类型（静态或可训练），将其分别添加到all_params的相应部分。
        assert (all_params["static"]["domain"]["xd"] == \
                all_params["static"]["problem"]["dims"][1] == \
                all_params["static"]["decomposition"]["xd"])
        # 确保子域、问题和分解的维度一致性，即它们都具有相同的空间维度xd。
        # 这是保证模型正确配置的重要一步。
        logger.info(f'Total number of subdomains: {all_params["static"]["decomposition"]["m"]}')
        # initialise subdomain network params
        network = c.network
        key, *subkeys = random.split(key, all_params["static"]["decomposition"]["m"] + 1)
        ps_ = vmap(network.init_params, in_axes=(0, None))(jnp.array(subkeys), *c.network_init_kwargs.values())
        # 使用JAX的vmap函数，这允许我们并行地对每个子域应用network.init_params函数，使用对应的subkeys作为随机密钥，并传入网络初始化的关键字参数值
        if ps_[0]: all_params["static"]["network"] = tree_index(ps_[0], 0)  # grab first set of static params only
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}  # add subdomain key
        logger.debug("all_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))
        model_fns = (decomposition.norm_fn, network.network_fn, decomposition.unnorm_fn, decomposition.window_fn,
                     problem.constraining_fn)
        # 定义一个包含多个模型函数的元组 model_fns。
        # 这些函数包括标准化函数、网络函数、去标准化函数、窗口函数和约束函数，
        # 分别来自 decomposition、network 和 problem。

        # initialise scheduler
        scheduler = c.scheduler(all_params=all_params, n_steps=c.n_steps, **c.scheduler_kwargs)
        # 初始化调度器 scheduler，用于控制训练过程中的调度策略。
        # 调用配置中的 scheduler，传入所有参数 all_params、步数 n_steps 以及其他调度器的关键字参数 scheduler_kwargs。

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
         constraints_global, x_batch_global, constraint_offsets_global, constraint_fs_global, jmapss,
         x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)
        # optimiser: 优化器对象，用于优化模型参数。
        # all_opt_states: 优化器的状态。
        # optimiser_fn: 优化器函数，用于执行参数更新。
        # loss_fn: 损失函数。
        # key: 随机数生成器的 key。
        # constraints_global: 所有约束条件的列表。去除约束条件中的导数信息
        # x_batch_global: 全局数据点。
        # constraint_offsets_global: 每个约束条件的偏移量索引。个数是约束列表的约束条件个数
        # constraint_fs_global: 约束矩阵，表明所有采样点所属的约束列表的信息（n,c)
        # jmapss: Jacobian映射。
        # x_batch_test: 测试点。
        # u_exact: 实际解

        # fix test data inputs
        logger.info("Getting test data inputs..")
        active_test_ = jnp.ones(all_params["static"]["decomposition"]["m"], dtype=int)
        # 创建一个全为 1 的数组，长度为 all_params["static"]["decomposition"]["m"]。这个数组用于表示哪些子域是活跃的。
        takes_, all_ims_, (_, _, _, cut_all_, _) = get_inputs(x_batch_test, active_test_, all_params, decomposition)
        test_inputs = (takes_, all_ims_, cut_all_)

        # train loop
        pstep, fstep, u_test_losses = 0, 0, []
        u_test_lossess = []
        start0, start1, report_time = time.time(), time.time(), 0.
        merge_active, active_params, active_opt_states, fixed_params = None, None, None, None
        lossval = None
        for i, active_ in enumerate(scheduler):
            # update active
            if active_ is not None:
                active = active_
                # first merge latest all_params / all_opt_states
                if i != 0:
                    all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                    all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)
                # then get new inputs to update step
                active, merge_active, active_opt_states, active_params, fixed_params, static_params, takess, constraints, x_batch = \
                    self._get_update_inputs(i, active, all_params, all_opt_states, x_batch_global, constraints_global,
                                            constraint_fs_global, constraint_offsets_global, decomposition, problem)
                # AOT compile update function
                startc = time.time()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling update step..")
                static_params_dynamic, static_params_static = partition(static_params)
                update = FBPINN_update.lower(optimiser_fn, active_opt_states,
                                             active_params, fixed_params, static_params_dynamic, static_params_static,
                                             takess, constraints, model_fns, jmapss, loss_fn).compile()
                logger.info(f"[i: {i}/{self.c.n_steps}] Compiling done ({time.time() - startc:.2f} s)")
                cost_ = update.cost_analysis()
                p, f = total_size(active_params["network"]), cost_[0]["flops"] if (cost_ and "flops" in cost_[0]) else 0
                logger.debug("p, f")
                logger.debug((p, f))

            # report initial model
            if i == 0:
                u_test_losses, start1, report_time = \
                    self._report(i, pstep, fstep, u_test_losses, u_test_lossess, start0, start1, report_time,
                                 u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem,
                                 decomposition,
                                 active, merge_active, active_opt_states, active_params, x_batch,
                                 lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                                               active_params, fixed_params, static_params_dynamic,
                                                               takess,
                                                               constraints)  # note compiled function only accepts dynamic arguments
            pstep, fstep = pstep + p, fstep + f

            # report
            u_test_losses, start1, report_time = \
                self._report(i + 1, pstep, fstep, u_test_losses, u_test_lossess, start0, start1, report_time,
                             u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem,
                             decomposition,
                             active, merge_active, active_opt_states, active_params, x_batch,
                             lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i + 1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = merge_active(active_params, all_params["trainable"])
        all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

        return all_params

    def _report(self, i, pstep, fstep, u_test_losses, u_test_lossess, start0, start1, report_time,
                u_exact, x_batch_test, test_inputs, all_params, all_opt_states, model_fns, problem, decomposition,
                active, merge_active, active_opt_states, active_params, x_batch,
                lossval):
        "Report results"

        c = self.c
        summary_, test_, model_save_ = [(i % f == 0) for f in
                                        [c.summary_freq, c.test_freq, c.model_save_freq]]
        if i != 0:
            rate = c.summary_freq / (time.time() - start1 - report_time)
            self._print_summary(i, lossval.item(), rate, start0, summary_)
            start1, report_time = time.time(), 0.

        if test_ or model_save_:

            if test_ or model_save_:

                start2 = time.time()

                # merge latest all_params / all_opt_states
                all_params["trainable"] = merge_active(active_params, all_params["trainable"])
                all_opt_states = tree_map_dicts(merge_active, active_opt_states, all_opt_states)

                # take test step
                if test_:
                    u_test_losses = self._test(
                        x_batch_test, u_exact, u_test_losses, u_test_lossess, x_batch, test_inputs, i, pstep, fstep,
                        start0, active, all_params, model_fns, problem, decomposition)

                # save model
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, active, jnp.array(u_test_losses)))

                report_time += time.time() - start2

        return u_test_losses, start1, report_time

    def _test(self, x_batch_test, u_exact, u_test_losses, u_test_lossess, x_batch, test_inputs, i, pstep, fstep, start0,
              active, all_params, model_fns, problem, decomposition):
        "Test step"
        c, writer = self.c, self.writer
        n_test = c.n_test
        num = c.test_freq

        # get FBPINN solution using test data
        takes, all_ims, cut_all = test_inputs
        all_params_cut = {"static": cut_all(all_params["static"]),
                          "trainable": cut_all(all_params["trainable"])}
        u_test, wp_test_, us_test_, ws_test_, us_raw_test_ = FBPINN_model_jit(all_params_cut, x_batch_test, takes,
                                                                              model_fns, verbose=False)
        if all_params["static"]["problem"]["dims"][1] == 1:  # 1D plots require full lines, not just hist stats

            m, ud, n = all_params["static"]["decomposition"]["m"], all_params["static"]["problem"]["dims"][0], \
            x_batch_test.shape[0]

            us_test = jnp.full((m, n, ud), jnp.nan)
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].set(us_test_)

            ws_test = jnp.full((m, n, 1), jnp.nan)
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].set(ws_test_)

            us_raw_test = jnp.full((m, n, ud), jnp.nan)
            us_raw_test = us_raw_test.at[all_ims[takes[0]], takes[1], :].set(us_raw_test_)

            # apply POU
            us_test = us_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]]) / takes[4]
            ws_test = ws_test.at[all_ims[takes[0]], takes[1], :].divide(wp_test_[takes[2]]) / takes[4]

            # apply constraining operator
            us_test = vmap(model_fns[-1], in_axes=(None, None, 0))(all_params, x_batch_test, us_test)

        else:
            us_test, ws_test, us_raw_test = us_test_, ws_test_, us_raw_test_

        xmin = [-1.0, -1.0]
        xmax = [1.0, 1.0]

        x_batch_xy = x_batch_test[:, :2]
        x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        xy_center = np.array([[x_center, y_center]])
        # Compute the center of the rectangle
        side_lengths = xmax[0] - xmin[0]
        radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
        distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        mask = distances <= radius  # Points inside the circle
        #        pdb.set_trace()
        u_test = u_test.at[np.squeeze(mask)].set(0.0)
        # get losses over test data
        l1 = jnp.mean(jnp.abs(u_exact - u_test)).item()
        l1n = l1 / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time() - start0, l1, l1n])
        u_test_lossess.append([l1])
        writer.add_scalar("loss/test/l1_istep", l1, i)

        # create figures
        if i % (c.test_freq * 5) == 0:
            fs = plot_trainer.plot("FBPINN", all_params["static"]["problem"]["dims"],
                                   x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i,
                                   active, decomposition, n_test)
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses


class PINNTrainer(_Trainer):
    "PINN model trainer class"

    def train(self):
        "Train model"

        c, writer = self.c, self.writer

        # generate root key
        key = random.PRNGKey(c.seed)
        np.random.seed(c.seed)

        # define all_params
        all_params = {"static": {}, "trainable": {}}

        # initialise domain, problem and decomposition params
        domain, problem = c.domain, c.problem
        for tag, cl, kwargs in zip(["domain", "problem"], [domain, problem],
                                   [c.domain_init_kwargs, c.problem_init_kwargs]):
            ps_ = cl.init_params(**kwargs)
            if ps_[0]: all_params["static"][tag] = ps_[0]
            if ps_[1]: all_params["trainable"][tag] = ps_[1]
        assert (all_params["static"]["domain"]["xd"] == \
                all_params["static"]["problem"]["dims"][1])

        # initialise network params
        network = c.network
        key, subkey = random.split(key)
        ps_ = network.init_params(key=subkey, **c.network_init_kwargs)
        if ps_[0]: all_params["static"]["network"] = ps_[0]
        if ps_[1]: all_params["trainable"]["network"] = {"subdomain": ps_[1]}  # add subdomain key
        logger.debug("all_params")
        logger.debug(jax.tree_map(lambda x: str_tensor(x), all_params))

        # define unnorm function
        mu_, sd_ = c.decomposition_init_kwargs["unnorm"]
        unnorm_fn = lambda u: networks.unnorm(mu_, sd_, u)
        model_fns = (domain.norm_fn, network.network_fn, unnorm_fn, problem.constraining_fn)

        # common initialisation
        (optimiser, all_opt_states, optimiser_fn, loss_fn, key,
         constraints_global, x_batch_global, _, _, jmapss,
         x_batch_test, u_exact) = _common_train_initialisation(c, key, all_params, problem, domain)

        # get implicit jitted update function
        active_params = all_params["trainable"]
        static_params = all_params["static"]
        active_opt_states = all_opt_states
        x_batch = x_batch_global
        constraints = constraints_global

        # AOT compile update function
        startc = time.time()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling update step..")
        static_params_dynamic, static_params_static = partition(static_params)
        update = PINN_update.lower(optimiser_fn, active_opt_states,
                                   active_params, static_params_dynamic, static_params_static,
                                   constraints, model_fns, jmapss, loss_fn).compile()
        logger.info(f"[i: {0}/{self.c.n_steps}] Compiling done ({time.time() - startc:.2f} s)")
        cost_ = update.cost_analysis()
        p, f = total_size(active_params["network"]), cost_[0]["flops"] if (cost_ and "flops" in cost_[0]) else 0
        logger.debug("p, f")
        logger.debug((p, f))

        # train loop
        pstep, fstep, u_test_losses = 0, 0, []
        start0, start1, report_time = time.time(), time.time(), 0.
        lossval = None
        for i in range(c.n_steps):

            if i == 0:
                # report initial model
                u_test_losses, start1, report_time = \
                    self._report(i, pstep, fstep, u_test_losses, start0, start1, report_time,
                                 u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                                 active_opt_states, active_params,
                                 x_batch,
                                 lossval)

            # take a training step
            lossval, active_opt_states, active_params = update(active_opt_states,
                                                               active_params, static_params_dynamic,
                                                               constraints)  # note compiled function only accepts dynamic arguments
            pstep, fstep = pstep + p, fstep + f
            # report
            u_test_losses, start1, report_time = \
                self._report(i + 1, pstep, fstep, u_test_losses, start0, start1, report_time,
                             u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                             active_opt_states, active_params,
                             x_batch,
                             lossval)

        # cleanup
        writer.close()
        logger.info(f"[i: {i + 1}/{self.c.n_steps}] Training complete")

        # return trained parameters
        all_params["trainable"] = active_params
        all_opt_states = active_opt_states

        return all_params

    def _report(self, i, pstep, fstep, u_test_losses, start0, start1, report_time,
                u_exact, x_batch_test, all_params, all_opt_states, model_fns, problem,
                active_opt_states, active_params,
                x_batch,
                lossval):
        "Report results"

        c = self.c
        summary_, test_, model_save_ = [(i % f == 0) for f in
                                        [c.summary_freq, c.test_freq, c.model_save_freq]]
        if i != 0:
            rate = c.summary_freq / (time.time() - start1 - report_time)
            self._print_summary(i, lossval.item(), rate, start0, summary_)
            start1, report_time = time.time(), 0.

        if test_ or model_save_:

            if test_ or model_save_:

                start2 = time.time()

                # merge latest params
                all_params["trainable"] = active_params
                all_opt_states = active_opt_states

                # take test step
                if test_:
                    u_test_losses = self._test(
                        x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns,
                        problem)

                # save model
                if model_save_:
                    self._save_model(i, (i, all_params, all_opt_states, jnp.array(u_test_losses)))

                report_time += time.time() - start2

        return u_test_losses, start1, report_time

    def _test(self, x_batch_test, u_exact, u_test_losses, x_batch, i, pstep, fstep, start0, all_params, model_fns,
              problem):
        "Test step"
        c, writer = self.c, self.writer
        n_test = c.n_test
        # get PINN solution using test data
        u_test, u_raw_test = PINN_model_jit(all_params, x_batch_test, model_fns, verbose=False)
        # xmin = [-1.0, -1.0]
        # xmax = [1.0, 1.0]
        # x_batch_xy = x_batch_test[:, :2]
        # x_center = xmin[0] + (1 / 2) * (xmax[0] - xmin[0])
        # y_center = xmin[1] + (1 / 4) * (xmax[1] - xmin[1])
        # xy_center = np.array([[x_center, y_center]])
        # # Compute the center of the rectangle
        # side_lengths = xmax[0] - xmin[0]
        # radius = np.min(side_lengths) / 4.0  # Use the shorter side's fifth as radius
        # distances = cdist(x_batch_xy, xy_center, metric='euclidean')
        # mask = distances <= radius  # Points inside the circle
        #        pdb.set_trace()
        # u_test = u_test.at[np.squeeze(mask)].set(0.0)
        # get losses over test data
        l1 = jnp.mean(jnp.abs(u_exact - u_test)).item()
        l1n = l1 / u_exact.std().item()
        u_test_losses.append([i, pstep, fstep, time.time() - start0, l1, l1n])
        writer.add_scalar("loss/test/l1_istep", l1, i)

        # create figures
        if i % (c.test_freq * 5) == 0:
            fs = plot_trainer.plot("PINN", all_params["static"]["problem"]["dims"],
                                   x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test)
            if fs is not None:
                self._save_figs(i, fs)

        return u_test_losses


if __name__ == "__main__":
    from fbpinns.constants import Constants, get_subdomain_ws
    from fbpinns.domains import RectangularDomainND
    from fbpinns.problems import FDTD1D, FDTD3D
    from fbpinns.decompositions import RectangularDecompositionND
    from fbpinns.networks import FCN
    from fbpinns.schedulers import LineSchedulerRectangularND
    from fbpinns.trainers import FBPINNTrainer, PINNTrainer

    # subdomain_xs = [np.linspace(-2, 2, 3), np.linspace(0, 1, 4)]#定义了x和t的范围，以及在x和t上划分的区间个数
    # subdomain_ws = get_subdomain_ws(subdomain_xs, 1.1)
    # c = Constants(
    #     run="test",
    #     domain=RectangularDomainND,
    #     domain_init_kwargs=dict(
    #         xmin=np.array([-0.5, 0]),
    #         xmax=np.array([0.5, 1]),
    #     ),
    #     problem=FDTD1D,
    #     problem_init_kwargs=dict(
    #         c=1, sd=0.1,
    #     ),
    #     decomposition=RectangularDecompositionND,
    #     decomposition_init_kwargs=dict(
    #         subdomain_xs=subdomain_xs,
    #         subdomain_ws=subdomain_ws,
    #         unnorm=(0., 1.),
    #     ),
    #     network=FCN,
    #     network_init_kwargs=dict(
    #         layer_sizes=[2,32,32,32,32, 2],
    #     ),
    #     ns=((50, 40),),#计算物理损失的点
    #     n_start=((100000, 1),),  # 表示在t==0时，x取200000个点
    #     n_boundary=((1, 100000),),
    #     n_test=(50, 40),
    #     n_steps=170000,
    #     optimiser_kwargs=dict(learning_rate=1e-3),
    #     summary_freq=2000,
    #     test_freq=2000,
    #     show_figures=False,
    #     clear_output=True,
    #     save_figures=True,
    # )
    # # run = FBPINNTrainer(c)
    # run = PINNTrainer(c)
    # run.train()

    # fdtd2d
    subdomain_xs = [np.linspace(-1, 1, 2), np.linspace(-1, 1, 2), np.linspace(0, 2, 3)]
    subdomain_ws = get_subdomain_ws(subdomain_xs, 1.9)

    c = Constants(
        run="test",
        domain=RectangularDomainND,
        domain_init_kwargs=dict(
            xmin=np.array([-1, -1, 0]),
            xmax=np.array([1, 1, 2]),
        ),
        problem=FDTD3D,
        problem_init_kwargs=dict(),
        decomposition=RectangularDecompositionND,
        decomposition_init_kwargs=dict(
            subdomain_xs=subdomain_xs,
            subdomain_ws=subdomain_ws,
            unnorm=(0., 1.),
        ),
        network=FCN,
        network_init_kwargs=dict(
            layer_sizes=[3, 16, 32, 32, 3],
        ),

        ns=((100, 100, 60),),
        n_start=((200, 200, 1),),
        n_boundary=((80, 80, 20),),
        # n_boundary = ((100, 1, 50),),
        n_test=(100, 100, 10),
        n_steps=100000,
        optimiser_kwargs=dict(learning_rate=1e-3),
        summary_freq=2000,
        test_freq=2000,
        show_figures=False,
        clear_output=True,
    )
    c["network_init_kwargs"] = dict(layer_sizes=[3, 32, 32, 32, 3])
    run = PINNTrainer(c)
    # run = FBPINNTrainer(c)
    run.train()
