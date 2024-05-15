import numpy as np
import matplotlib.pyplot as plt

# def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, xdim, ydim, time_tot, deltax, deltay, deltat, sd):
#     # Define Simulation Based off Source and Wavelength
#     # f0 = 1e6
#     f0 = 1
#     # Lf = 10
#     Lf = 1
#
#     # Spatial and Temporal System
#     # e0 = 8.854e-12
#     e0 = 1
#     # u0 = 4 * np.pi * 1e-7
#     u0 = 1
#     # c0 = 1 / (e0 * u0) ** 0.5
#     c0 = 1
#     L0 = c0 / f0
#     t0 = 1 / f0
#     Nx, Ny, nt = xdim, ydim, time_tot # Points in x,y
#     dx, dy, dt = deltax, deltay, deltat # x,y,z increment
#
#     # Initialize vectors
#     Hx, Hy, Ez = np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny))  # Magnetic and Electric Fields
#     Ez_out = np.zeros((Nx, Ny, nt))
#     udy, udx = dt / (u0 * dy), dt / (u0 * dx)  # H Field Coefficients
#     edx, edy = dt / (e0 * dx), dt / (e0 * dy)  # E Field Coefficients
#
#     # # Start loop
#     # pec_cx, pec_cy, pec_rad = 50, 50, 20
#     # pec_pt = []
#     # for i in range(1, 101):
#     #     for j in range(1, 101):
#     #         if np.sqrt((i - pec_cx) ** 2 + (j - pec_cy) ** 2) < pec_rad:
#     #             pec_pt.append([i, j])
#     #
#     # Npec, du = np.shape(pec_pt)
#
#     xg = np.linspace(xmin, xmax, Nx)
#     yg = np.linspace(ymin, ymax, Ny)
#     xv, yv = np.meshgrid(xg, yg)
#     zz = np.exp(-0.5 * ((xv - 0.) ** 2 + (yv + 0.) ** 2) / sd ** 2)
#
#     Ez[:,:] = zz
#
#     for t in range(1, nt + 1):
#         # Magnetic Field Update
#         for i in range(Nx - 1):
#             for j in range(Ny - 1):
#                 Hx[i, j] = Hx[i, j] - udy * (Ez[i, j + 1] - Ez[i, j])
#                 Hy[i, j] = Hy[i, j] + udx * (Ez[i + 1, j] - Ez[i, j])
#
#         # Electric Field Update
#         for i in range(1, Nx - 1):
#             for j in range(1, Ny - 1):
#                 Ez[i, j] = Ez[i, j] + edx * (Hy[i, j] - Hy[i - 1, j]) - edy * (Hx[i, j] - Hx[i, j - 1])
#
#         # # Point Source
#         # for m in range(Npec):
#         #     Ez[pec_pt[m][0], pec_pt[m][1]] = 0
#
#         Ez_out[:, :, t - 1] = Ez
#
#     return Ez_out


# import numpy as np
# import matplotlib.pyplot as plt
# def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, xdim, ydim, time_tot, deltax, deltay, deltat, sd):
#     f0 = 1e6
#     e0 = 8.854e-12
#     u0 = 4 * np.pi * 1e-7
#     c0 = 1 / (e0 * u0) ** 0.5
#     Nx, Ny, Nt = xdim, ydim, time_tot # Points in x,y
#     dx, dy, dt = deltax, deltay, deltat # x,y,z increment
#
#     # Initialize vectors
#     Hx, Hy, Ez = np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny))  # Magnetic and Electric Fields
#     Ez_out = np.zeros((Nx, Ny, Nt))
#
#     xg = np.linspace(xmin, xmax, Nx)
#     yg = np.linspace(ymin, ymax, Ny)
#     xv, yv = np.meshgrid(xg, yg)
#     zz = np.exp(-0.5 * ((xv - 0.) ** 2 + (yv + 0.) ** 2) / sd ** 2)
#
#     Ez[:,:] = zz
#
#     # Initialization of permittivity and permeability matrices
#     epsilon = e0 * np.ones((Nx, Ny))
#     mu = u0 * np.ones((Nx, Ny))
#
#     # Initializing electric conductivity matrices in x and y directions
#     sigmax = np.zeros((Nx, Ny))
#     sigmay = np.zeros((Nx, Ny))
#
#     # Perfectly matched layer boundary design
#     bound_width = 25
#     gradingorder = 6
#     refl_coeff = 1e-6
#
#     sigmamax = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * bound_width * deltax)
#     boundfact1 = ((epsilon[xdim // 2, bound_width] / e0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact2 = 0 * ((epsilon[xdim // 2, ydim - bound_width] / e0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact3 = ((epsilon[bound_width, ydim // 2] / e0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact4 = ((epsilon[xdim - bound_width, ydim // 2] / e0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#
#     x = np.arange(bound_width + 1)
#     for i in range(Nx):
#         sigmax[i, bound_width::-1] = boundfact1 * ((x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                     x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#         sigmax[i, ydim - bound_width - 1:] = boundfact2 * (
#                     (x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                         x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#
#     for i in range(Ny):
#         sigmay[bound_width::-1, i] = boundfact3 * ((x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                     x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#         sigmay[xdim - bound_width - 1:, i] = boundfact4 * (
#                     (x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                         x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#
#     # Magnetic conductivity matrix obtained by Perfectly Matched Layer condition
#     sigma_starx = (sigmax * mu) / epsilon
#     sigma_stary = (sigmay * mu) / epsilon
#
#     # Multiplication factor matrices for H matrix update
#     G = (mu - 0.5 * deltat * sigma_starx) / (mu + 0.5 * deltat * sigma_starx)
#     H = (deltat / deltax) / (mu + 0.5 * deltat * sigma_starx)
#     A = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)
#     B = (deltat / deltax) / (mu + 0.5 * deltat * sigma_stary)
#
#     # Multiplication factor matrices for E matrix update
#     C = (epsilon - 0.5 * deltat * sigmax) / (epsilon + 0.5 * deltat * sigmax)
#     D = (deltat / deltax) / (epsilon + 0.5 * deltat * sigmax)
#     E = (epsilon - 0.5 * deltat * sigmay) / (epsilon + 0.5 * deltat * sigmay)
#     F = (deltat / deltax) / (epsilon + 0.5 * deltat * sigmay)
#
#     # Update loop
#     # Update loop
#     for t in range(1, Nt + 1):
#         # Magnetic Field Update
#         Hx[0:Nx - 1, 0:Ny - 1] = A[0:Nx - 1, 0:Ny - 1] * Hx[0:Nx - 1, 0:Ny - 1] + B[0:Nx - 1, 0:Ny - 1] * np.diff(Ez[0:Nx - 1, :], axis=1)
#         Hy[0:Nx - 1, 0:Ny - 1] = G[0:Nx - 1, 0:Ny - 1] * Hx[0:Nx - 1, 0:Ny - 1] + H[0:Nx - 1, 0:Ny - 1] *  np.diff(Ez[:, 0:Ny - 1], axis=0)
#
#         # Electric Field Update
#         Ez[1:Nx - 1, 1:Ny - 1] = C[1:Nx - 1, 1:Ny - 1] * Ez[1:Nx - 1, 1:Ny - 1] + D[1:Nx - 1, 1:Ny - 1] * np.diff(Hy[0:Nx - 1, 1:Ny - 1], axis=0) + E[1:Nx - 1, 1:Ny - 1] * Ez[1:Nx - 1, 1:Ny - 1] +  F[1:Nx - 1, 1:Ny - 1] * np.diff(Hx[1:Nx - 1, 0:Ny - 1],
#                                                                                                 axis=1)
#         Ez_out[:, :, t - 1] = Ez
#
#     return Ez_out


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import draw, show


# # Define PEC area
# for i in range(1, 101):
#     for j in range(1, 101):
#         if np.sqrt((i - pec_cx)**2 + (j - pec_cy)**2) < pec_rad:
#             pec_pt.append((i, j))

# def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, sd):
#
#     xdim, ydim, time_tot = NX, NY, NSTEPS
#     deltax, deltay, deltat = DELTAX, DELTAY, DELTAT
#     Ez_out = np.zeros((xdim, ydim, time_tot))
#     # Initialize magnetic and electric fields
#     Hx = np.zeros((xdim, ydim))
#     Hy = np.zeros((xdim, ydim))
#     Ez = np.zeros((xdim, ydim))
#     # Courant stability factor
#     S = 1 / (2 ** 0.5)
#
#     # Permittivity of vacuum [farad/meter]
#     e0 = 8.854e-12
#     # Permeability of vacuum [henry/meter]
#     u0 = 4 * np.pi * 10 ** -7
#     # Speed of light [meter/second]
#     c0 = 1 / np.sqrt(e0 * u0)
#
#
#     # PML setup
#     pml_width = 15
#     sigma_max = (np.log(1.0 / 1e-4) * (S + 1) / (2 * pml_width * deltax)) * e0 * c0
#     # Initialize sigma arrays for PML
#     sigmax = np.zeros((xdim, ydim))
#     sigmay = np.zeros((xdim, ydim))
#
#     # 定义sigma值
#     for i in range(xdim):
#         if i < pml_width:
#             sigmax[i, :] = sigma_max * ((pml_width - i) / pml_width) ** 2
#         elif i >= xdim - pml_width:
#             sigmax[i, :] = sigma_max * ((i - (xdim - pml_width)) / pml_width) ** 2
#
#     for j in range(ydim):
#         if j < pml_width:
#             sigmay[:, j] = sigma_max * ((pml_width - j) / pml_width) ** 2
#         elif j >= ydim - pml_width:
#             sigmay[:, j] = sigma_max * ((j - (ydim - pml_width)) / pml_width) ** 2
#
#     # Perfect Electric Conductor (PEC) setup
#     pec_cx, pec_cy = 50, 50
#     pec_rad = 5
#     pec_pt = []
#
#     # H Field Coefficients
#     udx, udy = deltat / (u0 * deltax), deltat / (u0 * deltay)
#     # E Field Coefficients
#     edx, edy = deltat / (e0 * deltax), deltat / (e0 * deltay)
#
#     # Number of PEC points
#     Npec = len(pec_pt)
#
#     # Create initial Gaussian profile for Ez
#     xg = np.linspace(xmin, xmax, xdim)
#     yg = np.linspace(ymin, ymax, ydim)
#     xv, yv = np.meshgrid(xg, yg)
#     zz = np.exp(-0.5 * ((xv - 0.) ** 2 + (yv + 0.) ** 2) / sd ** 2)
#
#     Ez = zz
#
#     # Simulation loop
#     for t in range(1, time_tot + 1):
#         # Magnetic field update
#         Hx[:-1, :-1] -= udy * np.diff(Ez[:-1, :], axis=1) / (1 + sigmax[:-1, :-1] * deltat)
#         Hy[:-1, :-1] += udx * np.diff(Ez[:, :-1], axis=0) / (1 + sigmay[:-1, :-1] * deltat)
#
#         # Electric field update
#         Ez[1:xdim - 1, 1:ydim - 1] += (edx / (1 + sigmax[1:-1, 1:-1] * deltat)) * np.diff(Hy[:xdim - 1, 1:ydim - 1],
#                                                                                           axis=0) - (
#                                                   edy / (1 + sigmay[1:-1, 1:-1] * deltat)) * np.diff(
#             Hx[1:xdim - 1, :ydim - 1], axis=1)
#         Ez *= (1 - sigmax * deltat / e0) / (1 + sigmax * deltat / e0)  # PML damping for Ez
#         Ez *= (1 - sigmay * deltat / e0) / (1 + sigmay * deltat / e0)
#         Ez_out[:, :, t - 1] = Ez
#
#     return Ez_out

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show

def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, sd):
    f0 = 1
    Lf = 1
    xdim, ydim, time_tot = NX, NY, NSTEPS
    deltax, deltay, deltat = DELTAX, DELTAY, DELTAT
    Ez_out = np.zeros((xdim, ydim, time_tot))
    # Initialize magnetic and electric fields
    Hx = np.zeros((xdim, ydim))
    Hy = np.zeros((xdim, ydim))
    Ez = np.zeros((xdim, ydim))
    # Courant stability factor
    S = 1 / (2 ** 0.5)

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
    refl_coeff = 1e-4
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
    zz = np.exp(-0.5 * ((xv - 0.) ** 2 + (yv + 0.) ** 2) / sd ** 2)

    Ez = zz

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
        # for (px, py) in pec_pt:
        #     Ez[px, py] = 0

        Ez_out[:, :, t - 1] = Ez
    #     # Plotting
    #     plt.clf()
    #     plt.imshow(Ez.T,
    #                extent=(deltax * 1e+6 * -xdim, deltax * 1e+6 * xdim, deltay * 1e+6 * -ydim, deltay * 1e+6 * ydim),
    #                cmap='RdBu')  # , vmin=-1, vmax=1)
    #
    #     plt.imshow(Ez.T, vmin=-0.1, vmax=0.1)
    #     plt.colorbar()
    #     plt.title(
    #         f'Colour-scaled image plot of Ez in a spatial domain with PML boundary and at time = {round(t * deltat * 1e+15)} fs')
    #     plt.xlabel('x (in um)')
    #     plt.ylabel('y (in um)')
    #     # plt.show()
    #     plt.draw()
    #     plt.pause(0.01)
    #
    # show()
    return Ez_out

#
# Ez_out = FDTD2D(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 298, 298, 669, 0.006734006734006735, 0.006734006734006735,
#                 0.004714045207910317, 0.1)
