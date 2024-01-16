# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 19:48:59 2023

@author: shuw
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

PICKLE_FILE_hy = 'hy_1d.txt'
PICKLE_FILE_ez = 'ez_1d.txt'
PICKLE_FILE_hy = 'hy.txt'
PICKLE_FILE_ez = 'ez.txt'

# read_from_pickle(PICKLE_FILE)

with open(PICKLE_FILE_hy, 'rb') as file:
  hy = np.loadtxt(file)

with open(PICKLE_FILE_ez, 'rb') as file:
  ez = np.loadtxt(file)

hy = hy.reshape(240,120)
ez = ez.reshape(240,120)
x = np.linspace(-2,2,240)
for time in range(120):
  if time % 20 == 0:
    plt.clf()
    plt.title("Ex after t=%i" % time)
    plt.plot(x, ez[:, time], x, hy[:, time])
    plt.legend(['ez', 'hy'])
    # plt.ylim([-8, 8])
    plt.show()
    plt.clf()
    plt.close()

