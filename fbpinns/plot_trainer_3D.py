"""
Defines plotting functions for 3D FBPINN / PINN problems

This module is used by plot_trainer.py (and subsequently trainers.py)
"""

import matplotlib.pyplot as plt
import numpy as np
from fbpinns.plot_trainer_1D import _plot_setup, _to_numpy
from fbpinns.plot_trainer_2D import _plot_test_im

@_to_numpy
def plot_3D_FBPINN(x_batch_test, u_exact, u_test, us_test, ws_test, us_raw_test, x_batch, all_params, i, active, decomposition, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch_test.min(0), x_batch_test.max(0)

    nt = n_test[-1]# slice across last dimension
    shape = (1+nt+1, 3)# nrows, ncols
    f = plt.figure(figsize=(8,8*shape[0]/3))

    # plot domain + x_batch
    for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
        plt.subplot2grid(shape,(0,iplot))
        plt.title(f"[{i}] Domain decomposition")
        plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
        decomposition.plot(all_params, active=active, create_fig=False, iaxes=[a,b])
        plt.xlim(xlim[0][a], xlim[1][a])
        plt.ylim(xlim[0][b], xlim[1][b])
        plt.gca().set_aspect("equal")

    # plot full solutions
    for it in range(nt):
        plt.subplot2grid(shape,(1+it,0))
        plt.title(f"[{i}] Full solution")
        _plot_test_im(u_test[:,2].reshape(-1, 1), xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,1))
        plt.title(f"[{i}] Ground truth")
        _plot_test_im(u_exact, xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,2))
        plt.title(f"[{i}] Difference")
        _plot_test_im(u_exact - u_test[:,2].reshape(-1, 1), xlim0, ulim, n_test, it=it)



    # plot raw hist
    plt.subplot2grid(shape,(1+nt,0))
    plt.title(f"[{i}] Raw solutions")
    plt.hist(us_raw_test.flatten(), bins=100, label=f"{us_raw_test.min():.1f}, {us_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)
@_to_numpy
def plot_3D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):

    xlim, ulim = _plot_setup(x_batch_test, u_exact)
    xlim0 = x_batch.min(0), x_batch.max(0)

    nt = n_test[-1]# slice across last dimension
    shape = (1+nt+1, 3)# nrows, ncols
    f = plt.figure(figsize=(8,8*shape[0]/3))

    # plot x_batch
    for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
        plt.subplot2grid(shape,(0,iplot))
        plt.title(f"[{i}] Training points")
        plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
        plt.xlim(xlim[0][a], xlim[1][a])
        plt.ylim(xlim[0][b], xlim[1][b])
        plt.gca().set_aspect("equal")

    # plot full solution
    for it in range(nt):
        plt.subplot2grid(shape,(1+it,0))
        plt.title(f"[{i}] Full solution")
        _plot_test_im(u_test, xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,1))
        plt.title(f"[{i}] Ground truth")
        _plot_test_im(u_exact, xlim0, ulim, n_test, it=it)

        plt.subplot2grid(shape,(1+it,2))
        plt.title(f"[{i}] Difference")
        _plot_test_im(u_exact - u_test, xlim0, ulim, n_test, it=it)

    # plot raw hist
    plt.subplot2grid(shape,(1+nt,0))
    plt.title(f"[{i}] Raw solution")
    plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
    plt.legend(loc=1)
    plt.xlim(-5,5)

    plt.tight_layout()

    return (("test",f),)
# @_to_numpy
# def plot_3D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):
#
#     xlim, ulim = _plot_setup(x_batch_test, u_exact)#测试点的最大值最小值；实际解的最大值和最小值
#     xlim0 = x_batch.min(0), x_batch.max(0)#采样点的横坐标x的最大值和最小值
#
#     nt = n_test[-1]# slice across last dimension
#     shape = (1+nt+1, 3)# nrows, ncols
#     f = plt.figure(figsize=(8,8*shape[0]/3))
#
#     # plot x_batch
#     for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
#         plt.subplot2grid(shape,(0,iplot))
#         plt.title(f"[{i}] Training points")
#         plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
#         plt.xlim(xlim[0][a], xlim[1][a])
#         plt.ylim(xlim[0][b], xlim[1][b])
#         plt.gca().set_aspect("equal")
#
#     # plot full solution
#     for it in range(nt):
#
#         plt.subplot2grid(shape,(1+it,0))
#         plt.title(f"[{i}] Pinn solution")
#         _plot_test_im(u_test[:,2].reshape(-1, 1), xlim0, ulim, n_test, it=it)
#         # _plot_test_im(u_test[:, 2], xlim0, ulim, n_test, it=it)
#
#         plt.subplot2grid(shape,(1+it,1))
#         plt.title(f" FDTD truth")
#         _plot_test_im(u_exact, xlim0, ulim, n_test, it=it)
#
#         plt.subplot2grid(shape,(1+it,2))
#         plt.title(f"[{i}] Difference")
#         # _plot_test_im(np.abs(u_exact - u_test[:,2].reshape(-1, 1)), xlim0, differencelim, n_test, it=it)
#         _plot_test_im(u_exact - u_test[:, 2].reshape(-1, 1), xlim0, ulim, n_test, it=it)
#
#     # plot raw hist
#     plt.subplot2grid(shape,(1+nt,0))
#     plt.title(f"[{i}] Raw solution")
#     plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
#     plt.legend(loc=1)
#     plt.xlim(-5,5)
#
#     plt.tight_layout()
#
#     return (("test",f),)

# #自适应每一行的最大值和最小值
# def plot_3D_PINN(x_batch_test, u_exact, u_test, u_raw_test, x_batch, all_params, i, n_test):
#
#     xlim, ulim1 = _plot_setup(x_batch_test, u_exact)
#     xlim0 = x_batch.min(0), x_batch.max(0)
#
#     num = n_test[0] * n_test[1] * n_test[2]
#     nt = n_test[-1] # slice across last dimension
#     batch = num // nt
#     shape = (1 + nt + 1, 3) # nrows, ncols
#     f = plt.figure(figsize=(8, 8 * shape[0] / 3))
#
#     # Compute common limits for the first two columns
#     common_xlim = [
#         min(np.min(x_batch_test[:, 0]), np.min(x_batch[:, 0])),
#         max(np.max(x_batch_test[:, 0]), np.max(x_batch[:, 0]))
#     ]
#     common_ylim = [
#         min(np.min(x_batch_test[:, 1]), np.min(x_batch[:, 1])),
#         max(np.max(x_batch_test[:, 1]), np.max(x_batch[:, 1]))
#     ]
#
#     # plot x_batch
#     for iplot, (a,b) in enumerate([[0,1],[0,2],[1,2]]):
#         plt.subplot2grid(shape,(0,iplot))
#         plt.title(f"[{i}] Training points")
#         plt.scatter(x_batch[:,a], x_batch[:,b], alpha=0.5, color="k", s=1)
#         plt.xlim(xlim[0][a], xlim[1][a])
#         plt.ylim(xlim[0][b], xlim[1][b])
#         plt.gca().set_aspect("equal")
#
#         # plot full solution
#     for it in range(nt):
#         it_start = int(it * batch)
#         it_end = int((it + 1) * batch)
#         current_ulim = [u_exact[it_start:it_end].min(),
#                         u_exact[it_start:it_end].max()]
#
#         plt.subplot2grid(shape, (1 + it, 0))
#         plt.title(f"[{i}] Pinn solution")
#         _plot_test_im(u_test[:, 2].reshape(-1, 1), xlim0, current_ulim, n_test, it=it)
#         plt.xlim(common_xlim)
#         plt.ylim(common_ylim)
#
#         plt.subplot2grid(shape, (1 + it, 1))
#         plt.title(f"FDTD truth")
#         _plot_test_im(u_exact, xlim0, current_ulim, n_test, it=it)
#         plt.xlim(common_xlim)
#         plt.ylim(common_ylim)
#
#         plt.subplot2grid(shape, (1 + it, 2))
#         plt.title(f"[{i}] Difference")
#         difference_lim = [0, max(abs((u_exact - u_test[:, 2].reshape(-1, 1))).max(), 0.1)]
#         _plot_test_im(u_exact - u_test[:, 2].reshape(-1, 1), xlim0, difference_lim, n_test, it=it)
#         plt.xlim(common_xlim)
#         plt.ylim(common_ylim)
#
#     # plot raw hist
#     plt.subplot2grid(shape, (1 + nt, 0))
#     plt.title(f"[{i}] Raw solution")
#     plt.hist(u_raw_test.flatten(), bins=100, label=f"{u_raw_test.min():.1f}, {u_raw_test.max():.1f}")
#     plt.legend(loc=1)
#     plt.xlim(-5, 5)
#
#     plt.tight_layout()
#
#     return (("test", f),)
