#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:43:14 2018

@author: bmoseley
"""

# This module defines a Constants object which defines and stores a problem setup and all of its hyperparameters,
# for both FBPINN and PINN runs
# The instantiated constants object should be passed to the appropriate trainer classes defined in main.py

# This module is used by main.py and the paper_main_ND.py scripts

# 该模块定义了一个 Constants 对象，用于定义和存储问题设置及其所有超参数，适用于 FBPINN 和 PINN 运行。
# 实例化的 constants 对象应传递给 main.py 中定义的适当的训练器类。

# 该模块被 main.py 和 paper_main_ND.py 脚本使用。
import socket

import numpy as np

import models
import problems
import active_schedulers
import torch
from constantsBase import ConstantsBase



# helper functions

def get_subdomain_xs(ds, scales):
    xs = []
    for d,scale in zip(ds, scales):
        x = np.cumsum(np.pad(d, (1,0)))
        x = 2*(x-x.min())/(x.max()-x.min())-1# normalise to [-1, 1]
        xs.append(scale*x)
    return xs

def get_subdomain_ws(subdomain_xs, width):
    return [width*np.min(np.diff(x))*np.ones_like(x) for x in subdomain_xs]


class Constants(ConstantsBase):

    def __init__(self, **kwargs):

        # Define run
        self.RUN = "test"
        # Define problem
        self.P = problems.WaveEquation3D(c="gaussian", source_sd=0.3)
        # Define domain
        self.SUBDOMAIN_XS = [np.array([-10, -3.33, 3.33, 10]), np.array([-10, -3.33, 3.33, 10]),
                        np.array([0, 2.5, 5, 7.5, 10])]
        # Define normalisation parameters
        self.BOUNDARY_N = (0.3,)  # sd
        # self.Y_N = (0,1/self.P.w)# mu, sd
        self.Y_N =  (0, 1)  # mu, sd
        # # Define scheduler
        # self.ACTIVE_SCHEDULER = active_schedulers.PointActiveSchedulerND
        # self.ACTIVE_SCHEDULER_ARGS = (np.array([0, ]),)
        # GPU parameters
        self.DEVICE = torch.device("cuda:0" )
        # self.DEVICE = [2,3]  # cuda device
        # Model parameters
        self.MODEL = models.FCN
        self.N_HIDDEN = 64
        self.N_LAYERS = 4

        # Optimisation parameters
        self.BATCH_SIZE = (58, 58, 58)
        self.RANDOM = True
        self.LRATE = 1e-3

        self.N_STEPS = 75000

        # seed
        self.SEED = 123

        # other
        self.BATCH_SIZE_TEST = (50, 50, 10)
        self.PLOT_LIMS = (1, False)

        ### summary output frequencies
        self.SUMMARY_FREQ = 250
        self.TEST_FREQ = 15000
        self.MODEL_SAVE_FREQ = 150000
        self.SHOW_FIGURES =False  # whether to show figures
        self.SAVE_FIGURES = True  # whether to save figures
        self.CLEAR_OUTPUT = False  # whether to clear output periodically
        ##########

        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]  # invokes __setitem__ in ConstantsBase

        # other calculated variables
        self.SUMMARY_OUT_DIR = "results/summaries/%s/" % (self.RUN)
        self.MODEL_OUT_DIR = "results/models/%s/" % (self.RUN)
        self.HOSTNAME = socket.gethostname().lower()

        # "Define default parameters"
        #
        # ######################################
        # ##### GLOBAL CONSTANTS FOR MODEL
        # ######################################
        #
        #
        # # Define run
        # self.RUN = "test"
        #
        # # Define problem
        # #self.P = problems.Cos1D_1(w=w, A=0)
        # self.P = problems.WaveEquation3D(c="gaussian", source_sd=0.3)
        #
        # # Define domain
        # self.SUBDOMAIN_XS = [np.array([-10, -3.33, 3.33, 10]), np.array([-10, -3.33, 3.33, 10]), np.array([0, 2.5, 5, 7.5, 10])]
        # self.SUBDOMAIN_WS = get_subdomain_ws(self.SUBDOMAIN_XS, 0.9)
        #
        # # Define normalisation parameters
        # self.BOUNDARY_N = (0.3,)# sd
        # #self.Y_N = (0,1/self.P.w)# mu, sd
        # self.Y_N = (0,1)# mu, sd
        #
        # # Define scheduler
        # self.ACTIVE_SCHEDULER = active_schedulers.PlaneActiveSchedulerND
        # self.ACTIVE_SCHEDULER_ARGS = (np.array([0, ]), [0, 1])
        #
        # # GPU parameters
        # self.DEVICE = 0# cuda device
        #
        # # Model parameters
        # self.MODEL = models.FCN
        # self.N_HIDDEN = 64
        # self.N_LAYERS = 4
        #
        # # Optimisation parameters
        # self.BATCH_SIZE = (58,58,58)
        # self.RANDOM = False
        # self.LRATE = 1e-3
        #
        # self.N_STEPS = 150000
        #
        # # seed
        # self.SEED = 123
        #
        # # other
        # self.BATCH_SIZE_TEST = (100,100,10)
        # self.PLOT_LIMS = (1, False)
        #
        # ### summary output frequencies
        # self.SUMMARY_FREQ    = 250
        # self.TEST_FREQ       = 5000
        # self.MODEL_SAVE_FREQ = 10000
        # self.SHOW_FIGURES = True# whether to show figures
        # self.SAVE_FIGURES = False# whether to save figures
        # self.CLEAR_OUTPUT = False# whether to clear output periodically
        # ##########
        #
        #
        #
        # # overwrite with input arguments
        # for key in kwargs.keys(): self[key] = kwargs[key]# invokes __setitem__ in ConstantsBase
        #
        # # other calculated variables
        # self.SUMMARY_OUT_DIR = "results/summaries/%s/"%(self.RUN)
        # self.MODEL_OUT_DIR = "results/models/%s/"%(self.RUN)
        # self.HOSTNAME = socket.gethostname().lower()


        # # Define run
        # self.RUN = "test"
        #
        # # Define problem
        # w = 1e-10
        # # self.P = problems.Cos1D_1(w=w, A=0)
        # self.P = problems.Sin1D_2(w=w, A=0, B=-1 / w)
        #
        # # Define domain
        # self.SUBDOMAIN_XS = get_subdomain_xs([np.array([2, 3, 2, 4, 3])], [2 * np.pi / self.P.w])
        # self.SUBDOMAIN_WS = get_subdomain_ws(self.SUBDOMAIN_XS, 0.7)
        #
        # # Define normalisation parameters
        # self.BOUNDARY_N = (1 / self.P.w,)  # sd
        # # self.Y_N = (0,1/self.P.w)# mu, sd
        # self.Y_N = (0, 1 / self.P.w ** 2)  # mu, sd
        #
        # # Define scheduler
        # self.ACTIVE_SCHEDULER = active_schedulers.PointActiveSchedulerND
        # self.ACTIVE_SCHEDULER_ARGS = (np.array([0, ]),)
        #
        # # GPU parameters
        # self.DEVICE = 0  # cuda device
        #
        # # Model parameters
        # self.MODEL = models.FCN
        # self.N_HIDDEN = 16
        # self.N_LAYERS = 2
        #
        # # Optimisation parameters
        # self.BATCH_SIZE = (500,)
        # self.RANDOM = False
        # self.LRATE = 1e-3
        #
        # self.N_STEPS = 50000
        #
        # # seed
        # self.SEED = 123
        #
        # # other
        # self.BATCH_SIZE_TEST = (5000,)
        # self.PLOT_LIMS = (1, False)
        #
        # ### summary output frequencies
        # self.SUMMARY_FREQ = 250
        # self.TEST_FREQ = 5000
        # self.MODEL_SAVE_FREQ = 10000
        # self.SHOW_FIGURES = True  # whether to show figures
        # self.SAVE_FIGURES = False  # whether to save figures
        # self.CLEAR_OUTPUT = False  # whether to clear output periodically
        # ##########
        #
        # # overwrite with input arguments
        # for key in kwargs.keys(): self[key] = kwargs[key]  # invokes __setitem__ in ConstantsBase
        #
        # # other calculated variables
        # self.SUMMARY_OUT_DIR = "results/summaries/%s/" % (self.RUN)
        # self.MODEL_OUT_DIR = "results/models/%s/" % (self.RUN)
        # self.HOSTNAME = socket.gethostname().lower()



    



