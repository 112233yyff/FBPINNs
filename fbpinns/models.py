#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:32:55 2021

@author: bmoseley
"""

# This module defines standard pytorch NN models

# This module is used by constants.py when defining when defining FBPINN / PINN problems
## 该模块定义了标准的 PyTorch 神经网络模型。

# 当定义 FBPINN / PINN 问题时，constants.py 使用该模块。

import torch
import torch.nn as nn

total_params = lambda model: sum(p.numel() for p in model.parameters())

#这个类是一个继承自 PyTorch 的 nn.Module 类的自定义神经网络模型。


class FCN(nn.Module):
    "Fully connected network"

    # 在 __init__ 方法中，它定义了神经网络的结构，包括输入层、多个隐藏层和输出层。每个隐藏层都是由线性层和激活函数组成的序列。
    # forward 方法定义了数据在网络中的传播过程，即数据从输入层经过隐藏层最终到达输出层。
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        
        # define layers

        activation = nn.Tanh
        
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        
        # define helper attributes / methods for analysing computational complexity
        
        d1,d2,h,l = N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS
        self.size =           d1*h + h         + (l-1)*(  h*h + h)        +   h*d2 + d2
        self._single_flop = 2*d1*h + h + 5*h   + (l-1)*(2*h*h + h + 5*h)  + 2*h*d2 + d2# assumes Tanh uses 5 FLOPS
        self.flops = lambda BATCH_SIZE: BATCH_SIZE*self._single_flop
        assert self.size == total_params(self)
        
    def forward(self, x):
                
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        
        return x
    

if __name__ == "__main__":
    # 在__main__部分进行了简单的测试，包括创建一个输入数据张量x，构建一个FCN模型，并对输入数据进行前向传播得到输出y。
    # 打印了模型的结构、输出形状、参数数量和浮点运算次数。
    import numpy as np
    
    x = np.arange(100).reshape((25,4)).astype(np.float32)
    x = torch.from_numpy(x)
    print(x.shape)
    
    model = FCN(4, 2, 8, 3)
    print(model)
    
    y = model(x)
    print(y.shape)
    
    print("Number of parameters:", model.size)
    print("Number of FLOPS:", model.flops(x.shape[0]))
    
    