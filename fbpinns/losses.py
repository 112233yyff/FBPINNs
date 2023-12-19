#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:16:38 2021

@author: bmoseley
"""

# This module defines various loss functions

# This module is used by main.py and problems.py
#Sure, here's the translation:

# 该模块定义了各种损失函数。

# 该模块被 main.py 和 problems.py 使用。

import torch


def l2_loss(a, b):
    "L2 loss function"
    
    return torch.mean((a-b)**2)


def l1_loss(a, b):
    "L1 loss function"
    
    return torch.mean(torch.abs(a-b))