import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show


# 圆形PEC
def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, sd, ):
    f0 = 1
    Lf = 1
    xdim, ydim, time_tot = NX, NY, NSTEPS
    deltax, deltay, deltat = DELTAX, DELTAY, DELTAT
    Ez_out = np.zeros((xdim, ydim, time_tot))
    Hx_out = np.zeros((xdim, ydim, time_tot))
    Hy_out = np.zeros((xdim, ydim, time_tot))
    # Initialize magnetic and electric fields
    Hx = np.zeros((xdim, ydim))
    Hy = np.zeros((xdim, ydim))
    Ez = np.zeros((xdim, ydim))
    # Courant stability factor
    S = 1 / (2 ** 0.5)

#####方形PEC
    # Perfect Electric Conductor (PEC) setup
    x_center, y_center = xmin + (1 / 2) * (xmax - xmin), ymin + (1 / 4) * (ymax - ymin)
    rect_width, rect_height = 1, 1
    rect_xmin = x_center - rect_width / 2
    rect_xmax = x_center + rect_width / 2
    rect_ymin = y_center - rect_height / 2
    rect_ymax = y_center + rect_height / 2
    pec_pt = []

    # 使用 numpy.arange 来生成浮点数范围
    x_values = np.arange(xmin, xmax + deltax, deltax)
    y_values = np.arange(ymin, ymax + deltay, deltay)

    for i in range(0, NX):
        for j in range(0, NY):
            # 判断该点是否在矩形内
            if ~((x_values[i] < rect_xmin) | (x_values[i] > rect_xmax) | (y_values[j] < rect_ymin) | (
                y_values[j] > rect_ymax)):
                pec_pt.append((i, j))
        # Perfect Electric Conductor (PEC) setup
    pec_cx, pec_cy = xmin + (1 / 4) * (xmax - xmin), ymax - (1 / 4) * (ymax - ymin)
    pec_rad = (xmax - xmin) / 4.0

    for i in range(0, NX):
        for j in range(0, NY):
            # 计算点 (i, j) 到中心点 (pec_cx, pec_cy) 的距离
            distance = np.sqrt((x_values[i] - pec_cx) ** 2 + (y_values[j] - pec_cy) ** 2)
            # 判断该点是否在圆内
            if distance < pec_rad:
                pec_pt.append((i, j))
    # pec_pt 现在包含了所有符合条件的点

    # Permittivity of vacuum [farad/meter]
    e0 = 1  # 8.854e-12
    # Permeability of vacuum [henry/meter]
    u0 = 1  # 4 * np.pi * 10**-7
    # Speed of light [meter/second]
    c0 = 1  # 1 / np.sqrt(e0 * u0)
    L0 = c0 / f0
    t0 = 1 / f0
    epsilon = e0 * np.ones((xdim, ydim))
    mu = u0 * np.ones((xdim, ydim))

    # PML setup
    pml_width_dim = 25
    gradingorder = 6
    refl_coeff = 1e-8
    sigmamax_x = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltax)
    sigmamax_y = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltay)

    boundfact1 = ((epsilon[xdim // 2, pml_width_dim] / e0) * sigmamax_y) / (
                (pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact2 = ((epsilon[xdim // 2, ydim - pml_width_dim] / e0) * sigmamax_y) / (
                (pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact3 = ((epsilon[pml_width_dim, ydim // 2] / e0) * sigmamax_x) / (
                (pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact4 = ((epsilon[xdim - pml_width_dim, ydim // 2] / e0) * sigmamax_x) / (
                (pml_width_dim ** gradingorder) * (gradingorder + 1))

    # Initializing electric conductivity matrices in x and y directions
    sigmax = np.zeros((xdim, ydim))
    sigmay = np.zeros((xdim, ydim))

    x = np.arange(pml_width_dim + 1)
    for i in range(xdim):
        sigmax[i, pml_width_dim::-1] = boundfact1 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
                    x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
        sigmax[i, ydim - pml_width_dim - 1:] = boundfact2 * (
                    (x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
                        x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))

    for i in range(ydim):
        sigmay[pml_width_dim::-1, i] = boundfact3 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
                    x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
        sigmay[xdim - pml_width_dim - 1:, i] = boundfact4 * (
                    (x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
                        x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))

    # Magnetic conductivity matrix obtained by Perfectly Matched Layer condition
    sigma_starx = (sigmax * mu) / epsilon
    sigma_stary = (sigmay * mu) / epsilon

    # H Field Coefficients

    udy, udx = deltat / ((mu + 0.5 * deltat * sigma_starx) * deltay), deltat / (
                (mu + 0.5 * deltat * sigma_stary) * deltax)
    Gx = (mu - 0.5 * deltat * sigma_starx) / (mu + 0.5 * deltat * sigma_starx)
    Ay = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)
    # E Field Coefficients
    edx, edy = deltat / ((epsilon + 0.5 * deltat * sigmay) * deltax), deltat / (
                (epsilon + 0.5 * deltat * sigmax) * deltay)
    Cx = np.ones((xdim, ydim))
    for i in range(xdim):
        Cx[i, :] = (epsilon[i, :] - 0.5 * deltat * sigmax[i, :]) / (epsilon[i, :] + 0.5 * deltat * sigmax[i, :])
    for j in range(pml_width_dim, ydim - pml_width_dim):
        Cx[:, j] = (epsilon[:, j] - 0.5 * deltat * sigmay[:, j]) / (epsilon[:, j] + 0.5 * deltat * sigmay[:, j])

    # Create initial Gaussian profile for Ez
    xg = np.linspace(xmin, xmax, xdim)
    yg = np.linspace(ymin, ymax, ydim)
    xv, yv = np.meshgrid(xg, yg)
    zz = np.exp(-0.5 * ((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / sd ** 2)

    Ez = zz
    # Hx = -((yv - 0.5) / sd ** 2) * np.exp(-0.5 * ((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / sd ** 2)
    # Hy = -((xv - 0.5) / sd ** 2) * np.exp(-0.5 * ((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / sd ** 2)
    # Simulation loop
    for t in range(1, time_tot + 1):
        # Magnetic field update
        Hx[:-1, :-1] = Gx[:-1, :-1] * Hx[:-1, :-1] - udy[:-1, :-1] * np.diff(Ez[:-1, :], axis=1)
        Hy[:-1, :-1] = Ay[:-1, :-1] * Hy[:-1, :-1] + udx[:-1, :-1] * np.diff(Ez[:, :-1], axis=0)  #

        # Electric field update
        Ez[1:xdim - 1, 1:ydim - 1] = Cx[1:xdim - 1, 1:ydim - 1] * Ez[1:xdim - 1, 1:ydim - 1] + edx[1:xdim - 1,
                                                                                               1:ydim - 1] * np.diff(
            Hy[:xdim - 1, 1:ydim - 1], axis=0) - edy[1:xdim - 1, 1:ydim - 1] * np.diff(Hx[1:xdim - 1, :ydim - 1],
                                                                                       axis=1)

        # Enforce PEC condition
        for (px, py) in pec_pt:
            Ez[px, py] = 0

        Ez_out[:, :, t - 1] = Ez
        Hx_out[:, :, t - 1] = Hx
        Hy_out[:, :, t - 1] = Hy
    return Ez_out
    # return Hx_out
    # return Hy_out


# #c_fn
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import draw, show
#
# def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, sd, velocity):
#     f0 = 1
#     Lf = 1
#     xdim, ydim, time_tot = NX, NY, NSTEPS
#     deltax, deltay, deltat = DELTAX, DELTAY, DELTAT
#     Ez_out = np.zeros((xdim, ydim, time_tot))
#     Hx_out = np.zeros((xdim, ydim, time_tot))
#     Hy_out = np.zeros((xdim, ydim, time_tot))
#     # Initialize magnetic and electric fields
#     Hx = np.zeros((xdim, ydim))
#     Hy = np.zeros((xdim, ydim))
#     # Courant stability factor
#     S = 1 / (2 ** 0.5)
#
#     # Perfect Electric Conductor (PEC) setup
#     pec_cx, pec_cy = xmin + (1 / 2) * (xmax - xmin), ymin + (1 / 4) * (ymax - ymin)
#     pec_rad = (xmax - xmin) / 4.0
#     pec_pt = []
#
#     # 使用 numpy.arange 来生成浮点数范围
#     x_values = np.arange(xmin, xmax + deltax, deltax)
#     y_values = np.arange(ymin, ymax + deltay, deltay)
#
#     for i in range(0, NX):
#         for j in range(0, NY):
#             # 计算点 (i, j) 到中心点 (pec_cx, pec_cy) 的距离
#             distance = np.sqrt((x_values[i] - pec_cx) ** 2 + (y_values[j] - pec_cy) ** 2)
#             # 判断该点是否在圆内
#             if distance < pec_rad:
#                 pec_pt.append((i, j))
#
#     # pec_pt 现在包含了所有符合条件的点
#
#     # Permittivity of vacuum [farad/meter]
#     e0 = 1  # 8.854e-12
#     # Permeability of vacuum [henry/meter]
#     u0 = 1  # 4 * np.pi * 10**-7
#     # Speed of light [meter/second]
#     c0 = 1  # 1 / np.sqrt(e0 * u0)
#     L0 = c0 / f0
#     t0 = 1 / f0
#     epsilon = velocity
#     mu = u0 * np.ones((xdim, ydim))
#
#     # PML setup
#     pml_width_dim = 25
#     gradingorder = 6
#     refl_coeff = 1e-8
#     sigmamax_x = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltax)
#     sigmamax_y = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltay)
#
#     boundfact1 = ((epsilon[xdim // 2, pml_width_dim] / e0) * sigmamax_y) / (
#                 (pml_width_dim ** gradingorder) * (gradingorder + 1))
#     boundfact2 = ((epsilon[xdim // 2, ydim - pml_width_dim] / e0) * sigmamax_y) / (
#                 (pml_width_dim ** gradingorder) * (gradingorder + 1))
#     boundfact3 = ((epsilon[pml_width_dim, ydim // 2] / e0) * sigmamax_x) / (
#                 (pml_width_dim ** gradingorder) * (gradingorder + 1))
#     boundfact4 = ((epsilon[xdim - pml_width_dim, ydim // 2] / e0) * sigmamax_x) / (
#                 (pml_width_dim ** gradingorder) * (gradingorder + 1))
#
#     # Initializing electric conductivity matrices in x and y directions
#     sigmax = np.zeros((xdim, ydim))
#     sigmay = np.zeros((xdim, ydim))
#
#     x = np.arange(pml_width_dim + 1)
#     for i in range(xdim):
#         sigmax[i, pml_width_dim::-1] = boundfact1 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
#                     x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
#         sigmax[i, ydim - pml_width_dim - 1:] = boundfact2 * (
#                     (x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
#                         x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
#
#     for i in range(ydim):
#         sigmay[pml_width_dim::-1, i] = boundfact3 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
#                     x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
#         sigmay[xdim - pml_width_dim - 1:, i] = boundfact4 * (
#                     (x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (
#                         x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
#
#     # Magnetic conductivity matrix obtained by Perfectly Matched Layer condition
#     sigma_starx = (sigmax * mu) / epsilon
#     sigma_stary = (sigmay * mu) / epsilon
#
#     # H Field Coefficients
#
#     udy, udx = deltat / ((mu + 0.5 * deltat * sigma_starx) * deltay), deltat / (
#                 (mu + 0.5 * deltat * sigma_stary) * deltax)
#     Gx = (mu - 0.5 * deltat * sigma_starx) / (mu + 0.5 * deltat * sigma_starx)
#     Ay = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)
#     # E Field Coefficients
#     edx, edy = deltat / ((epsilon + 0.5 * deltat * sigmay) * deltax), deltat / (
#                 (epsilon + 0.5 * deltat * sigmax) * deltay)
#     Cx = np.ones((xdim, ydim))
#     for i in range(xdim):
#         Cx[i, :] = (epsilon[i, :] - 0.5 * deltat * sigmax[i, :]) / (epsilon[i, :] + 0.5 * deltat * sigmax[i, :])
#     for j in range(pml_width_dim, ydim - pml_width_dim):
#         Cx[:, j] = (epsilon[:, j] - 0.5 * deltat * sigmay[:, j]) / (epsilon[:, j] + 0.5 * deltat * sigmay[:, j])
#
#     # Create initial Gaussian profile for Ez
#     xg = np.linspace(xmin, xmax, xdim)
#     yg = np.linspace(ymin, ymax, ydim)
#     xv, yv = np.meshgrid(xg, yg)
#     zz = np.exp(-0.5 * ((xv - 0.5) ** 2 + (yv - 0.5) ** 2) / sd ** 2)
#
#     Ez = zz
#
#     # Simulation loop
#     for t in range(1, time_tot + 1):
#         # Magnetic field update
#         Hx[:-1, :-1] = Gx[:-1, :-1] * Hx[:-1, :-1] - udy[:-1, :-1] * np.diff(Ez[:-1, :], axis=1)
#         Hy[:-1, :-1] = Ay[:-1, :-1] * Hy[:-1, :-1] + udx[:-1, :-1] * np.diff(Ez[:, :-1], axis=0)  #
#
#         # Electric field update
#         Ez[1:xdim - 1, 1:ydim - 1] = Cx[1:xdim - 1, 1:ydim - 1] * Ez[1:xdim - 1, 1:ydim - 1] + edx[1:xdim - 1,
#                                                                                                1:ydim - 1] * np.diff(
#             Hy[:xdim - 1, 1:ydim - 1], axis=0) - edy[1:xdim - 1, 1:ydim - 1] * np.diff(Hx[1:xdim - 1, :ydim - 1],
#                                                                                        axis=1)
#
#         # # Enforce PEC condition
#         # for (px, py) in pec_pt:
#         #     Ez[px, py] = 0
#
#         Ez_out[:, :, t - 1] = Ez
#         Hx_out[:, :, t - 1] = Hx
#         Hy_out[:, :, t - 1] = Hy
#
#
#     # return Ez_out
#     # return Hx_out
#     return Hy_out