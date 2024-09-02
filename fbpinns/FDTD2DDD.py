import numpy as np

def FDTD2D(xmin, xmax, ymin, ymax, tmin, tmax, NX, NY, NSTEPS, DELTAX, DELTAY, DELTAT, sd, c):
    f0 = 10
    Lf = 1
    xdim, ydim, time_tot = NX, NY, NSTEPS
    deltax, deltay, deltat = DELTAX, DELTAY, DELTAT

    # Define new domain boundaries
    new_xmin, new_xmax = 2*xmin, 2*xmax
    new_ymin, new_ymax = 2*ymin, 2*ymax
    new_xdim = 2 * NX
    new_ydim = 2 * NY

    Ez_out = np.zeros((xdim, ydim, time_tot))
    Hx_out = np.zeros((xdim, ydim, time_tot))
    Hy_out = np.zeros((xdim, ydim, time_tot))

    # Initialize magnetic and electric fields for the extended domain
    Hx = np.zeros((new_xdim, new_ydim))
    Hy = np.zeros((new_xdim, new_ydim))
    Ez = np.zeros((new_xdim, new_ydim))

    # Courant stability factor
    S = 1 / (2 ** 0.5)

    # ########## PEC
    # pec_pt = []
    # x_values = np.arange(new_xmin, new_xmax + deltax, deltax)
    # y_values = np.arange(new_ymin, new_ymax + deltay, deltay)
    #
    # ### Rectangle PEC
    # x_center, y_center = -0.7, -0.5
    # rect_width, rect_height = 0.4, 0.4
    # rect_xmin = x_center - rect_width / 2
    # rect_xmax = x_center + rect_width / 2
    # rect_ymin = y_center - rect_height / 2
    # rect_ymax = y_center + rect_height / 2
    #
    # for i in range(new_xdim):
    #     for j in range(new_ydim):
    #         if not ((x_values[i] < rect_xmin) or (x_values[i] > rect_xmax) or (y_values[j] < rect_ymin) or (y_values[j] > rect_ymax)):
    #             pec_pt.append((i, j))
    #
    # ### Circle PEC
    # pec_cx, pec_cy = -0.7, 0.5
    # pec_rad = 0.25
    #
    # for i in range(new_xdim):
    #     for j in range(new_ydim):
    #         distance = np.sqrt((x_values[i] - pec_cx) ** 2 + (y_values[j] - pec_cy) ** 2)
    #         if distance < pec_rad:
    #             pec_pt.append((i, j))
    #
    # ### Triangle PEC
    # x_center = 0
    # y_center = -0.5
    # side_length = 0.5
    # half_side_length = side_length / 2
    # height = np.sqrt(side_length ** 2 - half_side_length ** 2)
    # tri_x = np.array([x_center - half_side_length, x_center + half_side_length, x_center])
    # tri_y = np.array([y_center - height / 3, y_center - height / 3, y_center + 2 * height / 3])
    #
    # for i in range(new_xdim):
    #     for j in range(new_ydim):
    #         denominator = (tri_y[1] - tri_y[2]) * (tri_x[0] - tri_x[2]) + (tri_x[2] - tri_x[1]) * (tri_y[0] - tri_y[2])
    #         a = ((tri_y[1] - tri_y[2]) * (x_values[i] - tri_x[2]) + (tri_x[2] - tri_x[1]) * (y_values[j] - tri_y[2])) / denominator
    #         b = ((tri_y[2] - tri_y[0]) * (x_values[i] - tri_x[2]) + (tri_x[0] - tri_x[2]) * (y_values[j] - tri_y[2])) / denominator
    #         c = 1 - a - b
    #         if (a >= 0) and (b >= 0) and (c >= 0):
    #             pec_pt.append((i, j))

    # Permittivity of vacuum [farad/meter]
    e0 = 1
    # Permeability of vacuum [henry/meter]
    u0 = 1
    # Speed of light [meter/second]
    c0 = 1
    L0 = c0 / f0
    t0 = 1 / f0
    # epsilon = e0 * np.ones((new_xdim, new_ydim))
    epsilon = c
    mu = u0 * np.ones((new_xdim, new_ydim))

    # PML setup
    pml_width_dim = 25
    gradingorder = 6
    refl_coeff = 1e-8
    sigmamax_x = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltax)
    sigmamax_y = (-np.log10(refl_coeff) * (gradingorder + 1) * e0 * c0) / (2 * pml_width_dim * deltay)

    boundfact1 = ((epsilon[new_xdim // 2, pml_width_dim] / e0) * sigmamax_y) / ((pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact2 = ((epsilon[new_xdim // 2, new_ydim - pml_width_dim] / e0) * sigmamax_y) / ((pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact3 = ((epsilon[pml_width_dim, new_ydim // 2] / e0) * sigmamax_x) / ((pml_width_dim ** gradingorder) * (gradingorder + 1))
    boundfact4 = ((epsilon[new_xdim - pml_width_dim, new_ydim // 2] / e0) * sigmamax_x) / ((pml_width_dim ** gradingorder) * (gradingorder + 1))

    # Initializing electric conductivity matrices in x and y directions
    sigmax = np.zeros((new_xdim, new_ydim))
    sigmay = np.zeros((new_xdim, new_ydim))

    x = np.arange(pml_width_dim + 1)
    for i in range(new_xdim):
        sigmax[i, pml_width_dim::-1] = boundfact1 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
        sigmax[i, new_ydim - pml_width_dim - 1:] = boundfact2 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))

    for i in range(new_ydim):
        sigmay[pml_width_dim::-1, i] = boundfact3 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))
        sigmay[new_xdim - pml_width_dim - 1:, i] = boundfact4 * ((x + 0.5 * np.ones(pml_width_dim + 1)) ** (gradingorder + 1) - (x - 0.5 * np.concatenate(([0], np.ones(pml_width_dim)))) ** (gradingorder + 1))

    # Magnetic conductivity matrix obtained by Perfectly Matched Layer condition
    sigma_starx = (sigmax * mu) / epsilon
    sigma_stary = (sigmay * mu) / epsilon

    # H Field Coefficients
    udy, udx = deltat / ((mu + 0.5 * deltat * sigma_starx) * deltay), deltat / ((mu + 0.5 * deltat * sigma_stary) * deltax)
    Gx = (mu - 0.5 * deltat * sigma_starx) / (mu + 0.5 * deltat * sigma_starx)
    Ay = (mu - 0.5 * deltat * sigma_stary) / (mu + 0.5 * deltat * sigma_stary)

    # E Field Coefficients
    edx, edy = deltat / ((epsilon + 0.5 * deltat * sigmay) * deltax), deltat / ((epsilon + 0.5 * deltat * sigmax) * deltay)
    Cx = np.ones((new_xdim, new_ydim))
    for i in range(new_xdim):
        Cx[i, :] = (epsilon[i, :] - 0.5 * deltat * sigmax[i, :]) / (epsilon[i, :] + 0.5 * deltat * sigmax[i, :])
    for j in range(pml_width_dim, new_ydim - pml_width_dim):
        Cx[:, j] = (epsilon[:, j] - 0.5 * deltat * sigmay[:, j]) / (epsilon[:, j] + 0.5 * deltat * sigmay[:, j])

    # Create initial Gaussian profile for Ez
    for j in range(new_xdim):
        for i in range(new_ydim):
            x = new_xmin + (new_xmax - new_xmin) * j / (new_xdim - 1)
            y = new_ymin + (new_ymax - new_ymin) * i / (new_ydim - 1)
            Ez[j, i] = np.exp(-0.5 * ((x - 0.5) ** 2 + (y - 0.5) ** 2) / sd ** 2)

    # Simulation loop
    for t in range(1, time_tot + 1):
        # Magnetic field update
        Hx[:-1, :-1] = Gx[:-1, :-1] * Hx[:-1, :-1] - udy[:-1, :-1] * np.diff(Ez[:-1, :], axis=1)
        Hy[:-1, :-1] = Ay[:-1, :-1] * Hy[:-1, :-1] + udx[:-1, :-1] * np.diff(Ez[:, :-1], axis=0)

        # Electric field update
        Ez[1:new_xdim - 1, 1:new_ydim - 1] = Cx[1:new_xdim - 1, 1:new_ydim - 1] * Ez[1:new_xdim - 1, 1:new_ydim - 1] + edx[1:new_xdim - 1, 1:new_ydim - 1] * np.diff(Hy[:new_xdim - 1, 1:new_ydim - 1], axis=0) - edy[1:new_xdim - 1, 1:new_ydim - 1] * np.diff(Hx[1:new_xdim - 1, :new_ydim - 1], axis=1)

        # # Enforce PEC condition
        # for (px, py) in pec_pt:
        #     Ez[px, py] = 0

        # Extract the results for the original domain [-1, 1]
        ix_min, ix_max = int((xmin - new_xmin) / deltax), int((xmax - new_xmin) / deltax)
        iy_min, iy_max = int((ymin - new_ymin) / deltay), int((ymax - new_ymin) / deltay)

        Ez_out[:, :, t - 1] = Ez[ix_min:ix_max + 1, iy_min:iy_max + 1]
        Hx_out[:, :, t - 1] = Hx[ix_min:ix_max + 1, iy_min:iy_max + 1]
        Hy_out[:, :, t - 1] = Hy[ix_min:ix_max + 1, iy_min:iy_max + 1]

    return Ez_out
