# import numpy as np
# import matplotlib.pyplot as plt
#
# def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, xdim, ydim, time_tot, delta, deltat, sd):
#     # Position of PEC along the line y=0
#     pec_pt = []
#
#     for i in range(xdim):
#         for j in range(ydim):
#             if j == 0:  # Check if the point is on the line y=0
#                 pec_pt.append([i, j])
#     Ez_out = np.zeros((xdim, ydim, time_tot))
#     # Position of the source (center of the domain)
#     xsource =(int) (xdim / 2)
#     ysource = (int) (ydim / 2)
#
#     # Parameters of free space
#     epsilon0 = 1  # (1 / (36 * np.pi)) * 1e-9
#     mu0 = 1  # 4 * np.pi * 1e-7
#     c = 1
#
#     # Initialization of field matrices
#     Ez = np.zeros((xdim, ydim))
#     Ezx = np.zeros((xdim, ydim))
#     Ezy = np.zeros((xdim, ydim))
#     Hy = np.zeros((xdim, ydim))
#     Hx = np.zeros((xdim, ydim))
#
#     # Initialization of permittivity and permeability matrices
#     epsilon = epsilon0 * np.ones((xdim, ydim))
#     mu = mu0 * np.ones((xdim, ydim))
#
#     # Choice of nature of source
#     gaussian = 1;
#     sine = 0;
#     # The user can give a frequency of his choice for sinusoidal (if sine=1 above) waves in Hz
#     frequency = 1.5  # 1.5e+13;
#     impulse = 0;
#
#     # Initializing electric conductivity matrices in x and y directions
#     sigmax = np.zeros((xdim, ydim))
#     sigmay = np.zeros((xdim, ydim))
#
#     # Perfectly matched layer boundary design
#     bound_width = 25
#     gradingorder = 6
#     refl_coeff = 1e-6
#
#     sigmamax = (-np.log10(refl_coeff) * (gradingorder + 1) * epsilon0 * c) / (2 * bound_width * delta)
#     boundfact1 = ((epsilon[xdim // 2, bound_width] / epsilon0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact2 = 0 * ((epsilon[xdim // 2, ydim - bound_width] / epsilon0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact3 = ((epsilon[bound_width, ydim // 2] / epsilon0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#     boundfact4 = ((epsilon[xdim - bound_width, ydim // 2] / epsilon0) * sigmamax) / (
#                 (bound_width ** gradingorder) * (gradingorder + 1))
#
#     x = np.arange(bound_width + 1)
#     for i in range(xdim):
#         sigmax[i, bound_width::-1] = boundfact1 * ((x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                     x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#         sigmax[i, ydim - bound_width - 1:] = boundfact2 * (
#                     (x + 0.5 * np.ones(bound_width + 1)) ** (gradingorder + 1) - (
#                         x - 0.5 * np.concatenate(([0], np.ones(bound_width)))) ** (gradingorder + 1))
#
#     for i in range(ydim):
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
#     H = (deltat / delta) / (mu + 0.5 * deltat * sigma_starx)
#     A = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)
#     B = (deltat / delta) / (mu + 0.5 * deltat * sigma_stary)
#
#     # Multiplication factor matrices for E matrix update
#     C = (epsilon - 0.5 * deltat * sigmax) / (epsilon + 0.5 * deltat * sigmax)
#     D = (deltat / delta) / (epsilon + 0.5 * deltat * sigmax)
#     E = (epsilon - 0.5 * deltat * sigmay) / (epsilon + 0.5 * deltat * sigmay)
#     F = (deltat / delta) / (epsilon + 0.5 * deltat * sigmay)
#
#     # #start
#     # x_cood = np.linspace(xmin, xmax, xdim)
#     # y_cood = np.linspace(ymin, ymax, ydim)
#     # X, Y = np.meshgrid(x_cood, y_cood)
#     # gaussian_value = np.exp(-(((X ** 2 + Y ** 2) / (sd ** 2)) / 2))
#     # Ez_out[:, :, 0] = gaussian_value.reshape(xdim, ydim)
#     # Ezx[:, 0:1] = np.exp(-(((x_cood / sd) ** 2) / 2)).reshape(-1, 1)
#     # Ezy[:, 0:1] = np.exp(-(((y_cood / sd) ** 2) / 2)).reshape(-1, 1)
#     # Ez_out[:, :, 0:1] = np.exp(-(((x_cood **2 + y_cood **2) / sd **2)/ 2)).reshape(-1, 1)
#
#     # Update loop
#     for n in range(1, time_tot + 1):
#         if n == 1:
#             Ez = np.exp(-((xsource ** 2 + ysource **2)/ sd **2) / 2) .reshape(-1, 1)
#         n1 = xsource - n - 1 if n < xsource - 2 else 1
#         n2 = xsource + n if n < xdim - 1 - xsource else xdim - 1
#         n11 = ysource - n - 1 if n < ysource - 2 else 1
#         n21 = ysource + n if n < ydim - 1 - ysource else ydim - 1
#
#         Hy[n1:n2, n11:n21] = A[n1:n2, n11:n21] * Hy[n1:n2, n11:n21] + B[n1:n2, n11:n21] * (
#                 Ezx[n1 + 1:n2 + 1, n11:n21] - Ezx[n1:n2, n11:n21] + Ezy[n1 + 1:n2 + 1, n11:n21] - Ezy[n1:n2, n11:n21])
#         Hx[n1:n2, n11:n21] = G[n1:n2, n11:n21] * Hx[n1:n2, n11:n21] - H[n1:n2, n11:n21] * (
#                 Ezx[n1:n2, n11 + 1:n21 + 1] - Ezx[n1:n2, n11:n21] + Ezy[n1:n2, n11 + 1:n21 + 1] - Ezy[n1:n2, n11:n21])
#
#         Ezx[n1 + 1:n2 + 1, n11 + 1:n21 + 1] = C[n1 + 1:n2 + 1, n11 + 1:n21 + 1] * Ezx[n1 + 1:n2 + 1,
#                                                                                   n11 + 1:n21 + 1] + D[
#                                                                                                      n1 + 1:n2 + 1,
#                                                                                                      n11 + 1:n21 + 1] * (
#                                                       -Hx[n1 + 1:n2 + 1, n11 + 1:n21 + 1] + Hx[n1 + 1:n2 + 1, n11:n21])
#         Ezy[n1 + 1:n2 + 1, n11 + 1:n21 + 1] = E[n1 + 1:n2 + 1, n11 + 1:n21 + 1] * Ezy[n1 + 1:n2 + 1,
#                                                                                   n11 + 1:n21 + 1] + F[
#                                                                                                      n1 + 1:n2 + 1,
#                                                                                                      n11 + 1:n21 + 1] * (
#                                                       Hy[n1 + 1:n2 + 1, n11 + 1:n21 + 1] - Hy[n1:n2, n11 + 1:n21 + 1])
#
#         # PEC
#         for px, py in pec_pt:
#             Ezx[px, py] = 0.
#             Ezy[px, py] = 0.
#         if n == 1:
#             Ez = Ez
#         else:
#             Ez = Ezx + Ezy
#         Ez_out[:, :, n - 1] = Ez
#
#     return Ez_out


import numpy as np
import matplotlib.pyplot as plt

def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, xdim, ydim, time_tot, deltax, deltay, deltat, sd):
    # Define Simulation Based off Source and Wavelength
    # f0 = 1e6
    f0 = 1
    # Lf = 10
    Lf = 1

    # Spatial and Temporal System
    # e0 = 8.854e-12
    e0 = 1
    # u0 = 4 * np.pi * 1e-7
    u0 = 1
    # c0 = 1 / (e0 * u0) ** 0.5
    c0 = 1
    L0 = c0 / f0
    t0 = 1 / f0
    Nx, Ny, nt = xdim, ydim, time_tot # Points in x,y
    dx, dy, dt = deltax, deltay, deltat # x,y,z increment

    # Initialize vectors
    Hx, Hy, Ez = np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), np.zeros((Nx, Ny))  # Magnetic and Electric Fields
    Ez_out = np.zeros((Nx, Ny, nt))
    udy, udx = dt / (u0 * dy), dt / (u0 * dx)  # H Field Coefficients
    edx, edy = dt / (e0 * dx), dt / (e0 * dy)  # E Field Coefficients

    # # Start loop
    # pec_cx, pec_cy, pec_rad = 50, 50, 20
    # pec_pt = []
    # for i in range(1, 101):
    #     for j in range(1, 101):
    #         if np.sqrt((i - pec_cx) ** 2 + (j - pec_cy) ** 2) < pec_rad:
    #             pec_pt.append([i, j])
    #
    # Npec, du = np.shape(pec_pt)

    xg = np.linspace(xmin, xmax, Nx)
    yg = np.linspace(ymin, ymax, Ny)
    xv, yv = np.meshgrid(xg, yg)
    zz = np.exp(-0.5 * ((xv - 0.) ** 2 + (yv + 0.) ** 2) / sd ** 2)

    Ez[:,:] = zz

    for t in range(1, nt + 1):
        # Magnetic Field Update
        for i in range(Nx - 1):
            for j in range(Ny - 1):
                Hx[i, j] -= udy * (Ez[i, j + 1] - Ez[i, j])
                Hy[i, j] += udx * (Ez[i + 1, j] - Ez[i, j])

        # Electric Field Update
        for i in range(1, Nx - 1):
            for j in range(1, Ny - 1):
                Ez[i, j] += edx * (Hy[i, j] - Hy[i - 1, j]) - edy * (Hx[i, j] - Hx[i, j - 1])

        # # Point Source
        # for m in range(Npec):
        #     Ez[pec_pt[m][0], pec_pt[m][1]] = 0

        Ez_out[:, :, t - 1] = Ez

    return Ez_out
    # # Plot
    # plt.imshow(Hx, extent=[x[0], x[-1], y[0], y[-1]])
    # plt.colorbar()
    # plt.show()
