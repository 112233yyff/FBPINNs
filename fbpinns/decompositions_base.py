# """
# Contains fast memory-limited functions for computing whether a set of points are inside a set of models
#
# We batch computation across the points dimension to limit memory usage
#
# Notes:
#     we want to avoid many dynamic shapes where possible, to avoid lots of implicit compilation
#     below the only dynamic shape is the size of the global n_take, m_take (and final reduction of inside_ims/inside_ips)
#     which is precomputed using _inside_sum_batch
#     we avoid using (dynamic) nonzero in the inner batch loop by instead using a gather operation on n_take, m_take
#     lax.scan and lax.map need static batch shapes, so masking is used for remainders
#     eventually this could be batched across model dimension too
#
# This module is used by decompositions.py
# """
# #这个模块包含用于计算一组点是否在一组模型内的快速、内存受限函数。
# # 我们通过对点维度进行批处理来限制内存使用。
# # 注意：
# # - 我们希望尽可能避免使用许多动态形状，以避免大量的隐式编译。
# # - 下面唯一的动态形状是全局 `n_take`、`m_take` 的大小（以及 `inside_ims/inside_ips` 的最终归约），它们是使用 `_inside_sum_batch` 预先计算的。
# # - 我们通过在内部批处理循环中使用 gather 操作，而不是使用（动态的）nonzero 来避免使用（动态的）nonzero。
# # - `lax.scan` 和 `lax.map` 需要静态的批处理形状，所以使用掩码来处理剩余部分。
# # - 最终，这可以在模型维度上进行批处理。
# #
# # 这个模块由 `decompositions.py` 使用。
#
# from functools import partial
#
# import jax
# import jax.numpy as jnp
#
#
# @partial(jax.jit, static_argnums=(3,4))
# def _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn):
#     # 这段代码的作用是批处理计算一批点是否在一组模型内。现在来解释返回值中的
#     # s：s是一个整数，表示在给定批次中，所有点在所有模型中的总数。它是一个标量值，代表着批次中所有点在所有模型中的总和。
#     # 这个值的计算步骤如下：首先，对所有模型进行批处理计算。s1是一个长度为n的布尔值数组，表示每个点是否在任何模型中。s2是一个长度为m的数组，表示每个模型内的点的总数。这些值由batch_step函数计算得出。
#     # 然后，通过对s2执行sum操作，得到了所有模型内的点的总数。这个值即为s，也就是批次中所有点在所有模型中的总数。
#     # 总之，s表示了批次中所有点在所有模型中的总数。
#
#     def batch_step(x):
#         i, mask = x
#         x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
#         inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
#         s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
#         return (s1, s2)# (n), (m)
#
#     # get fully-populated batches by shifting last value of irange
#     r = x_batch.shape[0]%batch_size
#     shift = batch_size-r if r else 0
#     irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
#     mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
#     irange = irange.at[-1].add(-shift)
#     mask = mask.at[-1,:shift].set(False)
#     s1, s2 = jax.lax.map(batch_step, (irange, mask))
#
#     # parse ims and ips
#     inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
#     inside_ims = s2.sum(0)# (m)
#     d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
#     s = inside_ims.sum()
#     inside_ims = inside_ims.astype(bool)
#     return (s, inside_ips, inside_ims, d), irange, mask
# #其中 s 表示总的内部点数，inside_ips 表示每个批次中内部点的索引，inside_ims 表示每个子域中的内部点数，d 表示每个模型平均包含的点数。
#
# @partial(jax.jit, static_argnums=(3,4,5))
# def _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask):
#
#     def batch_step(carry, x):
#         i, mask = x
#         x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
#         inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
#         inside_ = inside_.ravel()# (n*m)
#         itake = jnp.cumsum(inside_)-1# (n*m)
#         ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
#         take, s = carry
#         take = take.at[s+itake].add(ii_)# (s, 2)
#         return (take, s+itake[-1]+1), None
#
#     ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
#     #ix 是一个包含了从 0 到 batch_size - 1 的一维数组，表示批次中每个点在批次中的索引。
#     #iy 是一个包含了从 0 到 ims.shape[0] - 1 的一维数组，表示子域在子域数组中的索引。
#     ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
#     take = jnp.zeros((s,2), dtype=int)# (s, 2)
#     (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
#     return take
# #从一组点中选择那些位于模型内部的点，并返回这些点在原始数据中的索引。
#
# def inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn):
#     assert batch_size <= x_batch.shape[0]
#     (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
#     inside_ims = jnp.arange(ims.shape[0])[inside_ims]
#     s = s.item()
#     take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
#     return take[:,0], take[:,1], inside_ims
# #这段代码的函数 `inside_points_batch` 用于批量处理一组点，并确定这些点是否位于给定模型的内部。函数返回三个数组：
# # 1. 第一个数组包含了选定的内部点在原始数据中的行索引。
# # 2. 第二个数组包含了选定的内部点在原始数据中的列索引。
# # 3. 第三个数组包含了所有模型中内部点的模型索引。
# # 具体解释如下：
# # 1. 首先调用 `_inside_sum_batch` 函数计算给定点集中所有点的内部信息。返回的元组中包含了内部点的数量 `s`、内部点在原始数据中的行索引 `inside_ips`、
# # 内部点在模型中的列索引 `inside_ims` 和每个模型中平均点的数量 `d`。
# #
# # 2. 将模型索引数组 `inside_ims` 转换为数组形式，其中数组的值表示对应模型的索引。
# #
# # 3. 使用 `_inside_take_batch` 函数获取选定的内部点在原始数据中的索引。这个函数返回一个二维数组 `take`，其中每一行表示一个选定的内部点的全局索引，第一列是行索引，第二列是列索引。
# #
# # 4. 返回三个数组：选定的内部点在原始数据中的行索引、列索引以及所有模型中内部点的模型索引。
# def inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn):
#     assert batch_size <= x_batch.shape[0]
#     (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
#     inside_ips = jnp.arange(x_batch.shape[0])[inside_ips]
#     return inside_ips, d
#
#
#
#
# if __name__ == "__main__":
#
#     import jax.random as random
#
#     def inside_fn(all_params, x_batch, ims):
#         "Code for assessing if point is in ND hyperrectangle"
#         x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
#         xmins = jnp.expand_dims(all_params[0][ims], 0)# (1,mc,xd)
#         xmaxs = jnp.expand_dims(all_params[1][ims], 0)# (1,mc,xd)
#         inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
#         inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
#         return inside
#
#     def inside(all_params, x_batch, ims, inside_fn):
#         "full batch code to compare to"
#         inside = inside_fn(all_params, x_batch, ims)# (n, m)
#         n_take, m_take = jnp.nonzero(inside)
#         inside_ims = jnp.nonzero(jnp.any(inside, axis=0))[0]
#         inside_ips = jnp.nonzero(jnp.any(inside, axis=1))[0]
#         return n_take, m_take, inside_ims, inside_ips
#
#     n,m = 10000, 1000
#     x_batch = random.uniform(random.PRNGKey(0), (n,2), minval=0, maxval=2)
#     c = random.uniform(random.PRNGKey(0), (m,2), minval=1, maxval=3)
#     xmin, xmax = c.copy(), c.copy()
#     xmin -= 0.1
#     xmax += 0.1
#     all_params = [xmin, xmax]
#     ims = jnp.arange(m)
#
#     n_take_true, m_take_true, inside_ims_true, inside_ips_true = inside(all_params, x_batch, ims, inside_fn)
#
#     for batch_size in [1, 9, 10, 128, n, n+1]:
#         print(batch_size)
#
#         n_take, m_take, inside_ims = inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn)
#         inside_ips, d = inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn)
#
#         assert (n_take_true==n_take).all()
#         assert (m_take_true==m_take).all()
#         assert (inside_ims_true==inside_ims).all()
#         assert (inside_ips_true==inside_ips).all()
#
#
#
#
#
#
"""
Contains fast memory-limited functions for computing whether a set of points are inside a set of models

We batch computation across the points dimension to limit memory usage

Notes:
    we want to avoid many dynamic shapes where possible, to avoid lots of implicit compilation
    below the only dynamic shape is the size of the global n_take, m_take (and final reduction of inside_ims/inside_ips)
    which is precomputed using _inside_sum_batch
    we avoid using (dynamic) nonzero in the inner batch loop by instead using a gather operation on n_take, m_take
    lax.scan and lax.map need static batch shapes, so masking is used for remainders
    eventually this could be batched across model dimension too

This module is used by decompositions.py
"""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3,4))
def _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn):

    def batch_step(x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        s1, s2 = jnp.any(inside_, axis=1), inside_.sum(0)
        return (s1, s2)# (n), (m)

    # get fully-populated batches by shifting last value of irange
    r = x_batch.shape[0]%batch_size
    shift = batch_size-r if r else 0
    irange = jnp.arange(0, x_batch.shape[0], batch_size)# (k)
    mask = jnp.ones((len(irange), batch_size), dtype=bool)# (k, n)
    irange = irange.at[-1].add(-shift)
    mask = mask.at[-1,:shift].set(False)
    s1, s2 = jax.lax.map(batch_step, (irange, mask))

    # parse ims and ips
    inside_ips = jnp.concatenate([s1[:-1].ravel(), s1[-1][shift:]], axis=0)# (n)
    inside_ims = s2.sum(0)# (m)
    d = (inside_ims.mean()**(1/x_batch.shape[1]))# average number of points per model
    s = inside_ims.sum()
    inside_ims = inside_ims.astype(bool)
    return (s, inside_ips, inside_ims, d), irange, mask

@partial(jax.jit, static_argnums=(3,4,5))
def _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask):

    def batch_step(carry, x):
        i, mask = x
        x_batch_ = jax.lax.dynamic_slice(x_batch, (i,0), (batch_size, x_batch.shape[1]))# (n, xd)
        inside_ = jnp.expand_dims(mask,1)*inside_fn(all_params, x_batch_, ims)# (n, m)
        inside_ = inside_.ravel()# (n*m)
        itake = jnp.cumsum(inside_)-1# (n*m)
        ii_ = jnp.expand_dims(inside_,1)*ii.at[:,0].add(i)# (n*m, 2)
        take, s = carry
        take = take.at[s+itake].add(ii_)# (s, 2)
        return (take, s+itake[-1]+1), None

    ix,iy = jnp.meshgrid(jnp.arange(batch_size), jnp.arange(ims.shape[0]), indexing="ij")# (n, m)
    ii = jnp.stack([ix.ravel(), iy.ravel()], axis=1)# (n*m, 2)
    take = jnp.zeros((s,2), dtype=int)# (s, 2)
    (take, _), _ = jax.lax.scan(batch_step, (take, 0), (irange, mask))
    return take

def inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn):
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ims = jnp.arange(ims.shape[0])[inside_ims]
    s = s.item()
    take = _inside_take_batch(all_params, x_batch, ims, batch_size, inside_fn, s, irange, mask)
    return take[:,0], take[:,1], inside_ims

def inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn):
    assert batch_size <= x_batch.shape[0]
    (s, inside_ips, inside_ims, d), irange, mask = _inside_sum_batch(all_params, x_batch, ims, batch_size, inside_fn)
    inside_ips = jnp.arange(x_batch.shape[0])[inside_ips]
    return inside_ips, d




if __name__ == "__main__":

    import jax.random as random

    def inside_fn(all_params, x_batch, ims):
        "Code for assessing if point is in ND hyperrectangle"
        x_batch = jnp.expand_dims(x_batch, 1)# (n,1,xd)
        xmins = jnp.expand_dims(all_params[0][ims], 0)# (1,mc,xd)
        xmaxs = jnp.expand_dims(all_params[1][ims], 0)# (1,mc,xd)
        inside = (x_batch >= xmins) & (x_batch <= xmaxs)# (n,mc,xd)
        inside = jnp.all(inside, -1)# (n,mc) keep as bool to reduce memory
        return inside

    def inside(all_params, x_batch, ims, inside_fn):
        "full batch code to compare to"
        inside = inside_fn(all_params, x_batch, ims)# (n, m)
        n_take, m_take = jnp.nonzero(inside)
        inside_ims = jnp.nonzero(jnp.any(inside, axis=0))[0]
        inside_ips = jnp.nonzero(jnp.any(inside, axis=1))[0]
        return n_take, m_take, inside_ims, inside_ips

    n,m = 10000, 1000
    x_batch = random.uniform(random.PRNGKey(0), (n,2), minval=0, maxval=2)
    c = random.uniform(random.PRNGKey(0), (m,2), minval=1, maxval=3)
    xmin, xmax = c.copy(), c.copy()
    xmin -= 0.1
    xmax += 0.1
    all_params = [xmin, xmax]
    ims = jnp.arange(m)

    n_take_true, m_take_true, inside_ims_true, inside_ips_true = inside(all_params, x_batch, ims, inside_fn)

    for batch_size in [1, 9, 10, 128, n, n+1]:
        print(batch_size)

        n_take, m_take, inside_ims = inside_points_batch(all_params, x_batch, ims, batch_size, inside_fn)
        inside_ips, d = inside_models_batch(all_params, x_batch, ims, batch_size, inside_fn)

        assert (n_take_true==n_take).all()
        assert (m_take_true==m_take).all()
        assert (inside_ims_true==inside_ims).all()
        assert (inside_ips_true==inside_ips).all()