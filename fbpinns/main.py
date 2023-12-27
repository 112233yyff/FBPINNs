#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:53:56 2021

@author: bmoseley
"""

# This module defines trainer classes for FBPINNs and PINNs. It is the main entry point for training FBPINNs and PINNs
# To train a FBPINN / PINN, use a Constants object to setup the problem and define its hyperparameters, and pass that 
# to one of the trainer classes defined here

# This module is used by the paper_main_ND.py scripts

#这个模块定义了用于 FBPINN 和 PINN 的训练器类。它是训练 FBPINN 和 PINN 的主要入口点。
# 要训练 FBPINN / PINN，使用一个 Constants 对象来设置问题并定义其超参数，然后将其传递给这里定义的其中一个训练器类。
#
# 这个模块被 paper_main_ND.py 脚本所使用。

import time

import numpy as np
import torch
import torch.optim

import plot_main
import losses
from trainersBase import _Trainer
from constants import Constants
from domains import ActiveRectangularDomainND


## HELPER FUNCTIONS


def _x_random(subdomain_xs, batch_size, device):
    "Get flattened random samples of x"
    s = torch.tensor([[x.min(), x.max()] for x in subdomain_xs], dtype=torch.float32, device=device).T.unsqueeze(1)# (2, 1, nd)
    x_random = s[0]+(s[1]-s[0])*torch.rand((np.prod(batch_size),len(subdomain_xs)), device=device)# random samples in domain
    return x_random

def _x_mesh(subdomain_xs, batch_size, device):
    "Get flattened samples of x on a mesh"
    x_mesh = [torch.linspace(x.min(), x.max(), b, device=device) for x,b in zip(subdomain_xs, batch_size)]
    x_mesh = torch.stack(torch.meshgrid(*x_mesh), -1).view(-1, len(subdomain_xs))# nb torch.meshgrid uses np indexing="ij"
    return x_mesh

def full_model_FBPINN(x, models, c, D):
    """Get the full FBPINN prediction over all active and fixed models (forward inference only)"""
    
    def _single_model(im):# use separate function to ensure computational graph/memory is released
        
        x_ = x.detach().clone().requires_grad_(True)
        
        # normalise, run model, add window function
        mu, sd = D.n_torch[im]# (nd)
        y = models[im]((x_-mu)/sd)
        y_raw = y.detach().clone()
        y = y*c.Y_N[1] + c.Y_N[0]
        y = D.w[im](x_)*y
        
        # get gradients
        yj = c.P.get_gradients(x_, y)# problem-specific
        
        # detach from graph
        yj = [t.detach() for t in yj]
        
        # apply boundary conditions (for QC only)
        yj_bc = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
        
        return yj, yj_bc, y_raw
    
    # run all models
    yjs, yjs_bc, ys_raw = [], [], []
    for im in D.active_fixed_ims:
        yj, yj_bc, y_raw = _single_model(im)
        
        # add to model lists
        yjs.append(yj); yjs_bc.append(yj_bc); ys_raw.append(y_raw)
    
    # sum across models
    yj = [torch.sum(torch.stack(ts, -1), -1) for ts in zip(*yjs)]# note zip(*) transposes
    
    # apply boundary condition to summed solution
    yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
    
    return yj, yjs_bc, ys_raw

def full_model_PINN(x, model, c):
    """Get the full PINN prediction (forward inference only)"""
    #"获取完整的 PINN 预测（仅进行前向推断）"
    
    # get normalisation values
    xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=x.device).T
    mu = (xmin + xmax)/2; sd = (xmax - xmin)/2
    
    # get full model solution using test data
    x_ = x.detach().clone().requires_grad_(True)
    y = model((x_-mu)/sd)
    y_raw = y.detach().clone()
    y = y*c.Y_N[1] + c.Y_N[0]
    
    # get gradients
    yj = c.P.get_gradients(x_, y)# problem-specific
    
    # detach from graph
    yj = [t.detach() for t in yj]
    
    # apply boundary conditions
    yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
    
    return yj, y_raw


## MAIN TRAINER CLASSES


class FBPINNTrainer(_Trainer):
    "FBPINN model trainer class"
    
    def _train_step(self, models, optimizers, c, D, i):# use separate function to ensure computational graph/memory is released
        
        ## ZERO PARAMETER GRADIENTS, SET TO TRAIN
        for optimizer in optimizers: optimizer.zero_grad()
        for model in models: model.train()
        
        ## RANDOMLY SAMPLE ALL SEGMENTS
        x_segments = D.sample_segments()
        
        ## RUN MODELS (ACTIVE AND FIXED NEIGHBOURS)
        xs, yjs = [], []
        for im,_ in D.active_fixed_neighbours_ims:
            
            # sample segments in model
            x = [x_segments[iseg] for iseg in D.m[im]]
            x = torch.cat(x, dim=0).detach().clone().requires_grad_(True)#(N, nd)
            
            # normalise, run model, add window function
            mu, sd = D.n_torch[im]# (nd)
            y = models[im]((x-mu)/sd)
            y = y*c.Y_N[1] + c.Y_N[0]
            y = D.w[im](x)*y
            
            # get gradients
            yj = c.P.get_gradients(x, y)# problem-specific
            
            # add to model lists
            yjs.append(yj)
            xs.append(x)
            
            if (i % c.SUMMARY_FREQ) == 0: 
                print(*[t.shape for t in yj], x.shape)
        
        ## SUM OVERLAPPING MODELS, APPLY BCS (ACTIVE)
        yjs_sum = []
        for im,i1 in D.active_ims:
            
            # for each segment in model
            yjs_segs = []
            for iseg in D.m[im]:
                
                # for each model which contributes to that segment
                yjs_seg = []
                for im2,j1,j2,j3 in D.s[iseg]:
                    
                    # get model yj segment iseg
                    yj = yjs[j1]# j1 is the index of yj for model im2 in yjs above
                    if im2 == im: yj = [t[j2:j3]           for t in yj]# get appropriate segment
                    else:         yj = [t[j2:j3].detach()  for t in yj]
                    
                    # add to model list
                    yjs_seg.append(yj)
                
                # sum across models
                yj_seg = [torch.sum(torch.stack(ts, -1), -1) for ts in zip(*yjs_seg)]# note zip(*) transposes
                
                # add to segment list
                yjs_segs.append(yj_seg)
            
            # concatenate across segments
            yj = [torch.cat(ts) for ts in zip(*yjs_segs)]# note zip(*) transposes
            
            # apply boundary conditions
            x = xs[i1]
            yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
            
            # add to model list
            yjs_sum.append(yj)
            
            if (i % c.SUMMARY_FREQ) == 0: 
                print(*[t.shape for t in yj])# should be the same as above!
            
        ## BACKPROPAGATE LOSS (ACTIVE)
        for im,i1 in D.active_ims:
            x, yj = xs[i1], yjs_sum[i1]
            loss = c.P.physics_loss(x, *yj)# problem-specific
            loss.backward()
            optimizers[im].step()
        
        # return result
        return ([t.detach() for t in xs], 
                [[t.detach() for t in ts] for ts in yjs], 
                [[t.detach() for t in ts] for ts in yjs_sum], loss.item())
    
    def _test_step(self, x_test, yj_true,   xs, yjs, yjs_sum,   models, c, D, i, mstep, fstep, writer, yj_test_losses):# use separate function to ensure computational graph/memory is released

        # get full model solution using test data
        yj_full, yjs_full, ys_full_raw = full_model_FBPINN(x_test, models, c, D)
        print(x_test.shape, yj_true[0].shape, yj_full[0].shape)
        
        # get losses over test data
        yj_test_loss = [losses.l1_loss(a,b).item() for a,b in zip(yj_true, yj_full)]
        physics_loss = c.P.physics_loss(x_test, *yj_full).item()# problem-specific
        yj_test_losses.append([i + 1, mstep, fstep]+yj_test_loss+[physics_loss])
        for j,l in enumerate(yj_test_loss): 
            for step,tag in zip([i + 1, mstep, fstep], ["istep", "mstep", "zfstep"]):
                writer.add_scalar("loss_%s/yj%i/test"%(tag,j), l, step)
        writer.add_scalar("loss_istep/zphysics/test", physics_loss, i + 1)
        
        # PLOTS
        
        if (i + 1) % c.TEST_FREQ == 0:
            
            # save figures
            fs = plot_main.plot_FBPINN(x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw,   yj_test_losses,   c, D, i + 1)
            if fs is not None: self._save_figs(i, fs)
        
        del x_test, yj_true,   xs, yjs, yjs_sum,   yj_full, yjs_full, ys_full_raw# fixes weird over-allocation of GPU memory bug caused by plotting (?)
        
        return yj_test_losses
    
    def train(self):
        "Train model"
        
        c, device, writer = self.c, self.device, self.writer
        
        # define domain
        D = ActiveRectangularDomainND(c.SUBDOMAIN_XS, c.SUBDOMAIN_WS, device=device)
        D.update_sampler(c.BATCH_SIZE, c.RANDOM)
        A = c.ACTIVE_SCHEDULER(c.N_STEPS, D, *c.ACTIVE_SCHEDULER_ARGS)
        
        # create models
        models = [c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS) for _ in range(D.N_MODELS)]# problem-specific
        
        # create optimisers
        optimizers = [torch.optim.Adam(model.parameters(), lr=c.LRATE) for model in models]
        
        # put models on device
        for model in models: model.to(device)

        # get exact solution if it exists
        x_test = _x_mesh(c.SUBDOMAIN_XS, c.BATCH_SIZE_TEST, device)
        yj_true = c.P.exact_solution(x_test, c.BATCH_SIZE_TEST)# problem-specific
        
        ## TRAIN
        
        mstep, fstep, yj_test_losses = 0, 0, []
        start, gpu_time = time.time(), 0.
        for i,active in enumerate(A):
            
            # update active if required
            if active is not None: 
                D.update_active(active)
                print(i, "Active updated:\n", active)
                
            gpu_start = time.time()
            xs, yjs, yjs_sum, loss = self._train_step(models, optimizers, c, D, i)
            for im,i1 in D.active_ims: mstep += models[im].size# record number of weights updated
            for im,i1 in D.active_fixed_neighbours_ims: fstep += models[im].flops(xs[i1].shape[0])# record number of FLOPS
            gpu_time += time.time()-gpu_start
            
            
            # METRICS
            
            if (i + 1) % c.SUMMARY_FREQ == 0:
                
                # set counters
                rate, gpu_time = c.SUMMARY_FREQ / gpu_time, 0.
                
                # print summary
                self._print_summary(i, loss, rate, start)
                
                # test step
                yj_test_losses = self._test_step(x_test, yj_true,   xs, yjs, yjs_sum,   models, c, D, i, mstep, fstep, writer, yj_test_losses)
            
            # SAVE
            
            if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                
                # save models, losses and active array
                for im,model in enumerate(models):
                    self._save_model(i, model, im)
                np.save(c.MODEL_OUT_DIR+"active_%.8i.npy"%(i + 1), D.active)
                np.save(c.MODEL_OUT_DIR+"loss_%.8i.npy"%(i + 1), np.array(yj_test_losses))
        
        # cleanup
        writer.close()
        print("Finished training")
        

#这段代码是一个 PyTorch 框架下的模型训练器（PINNTrainer），负责训练一个物理信息神经网络（Physics-Informed Neural Network）。这个训练器有几个关键函数：
#
# _train_step: 这个函数执行模型的训练步骤。在每次训练步骤中，它对模型进行了优化器梯度归零，
# 然后执行了模型的正向传播计算，根据问题特定的函数计算损失并进行反向传播更新模型参数。最后，它返回了输入 x、梯度 yj 以及损失值。
#
# _test_step: 这个函数用于模型的测试步骤。它使用测试数据进行模型的评估，并计算模型在测试数据上的损失。
# 同时，它还执行了一些数据可视化和保存操作，例如绘制图表和保存模型。这个函数主要用于验证模型在训练之后的性能。
#
# train: 这是执行整个模型训练的函数。它首先定义了模型结构，然后创建了优化器，并将模型放到合适的设备上进行训练。
# 在训练过程中，它迭代执行训练步骤 _train_step，定期输出训练过程的指标和日志，并在特定步骤执行测试步骤 _test_step 进行模型评估。
# 最后，它保存了模型和损失记录，并关闭了相应的写入器（writer）。
#
# 整个流程包括模型的定义、优化器的设置、训练过程中的指标记录和模型评估，并最终保存训练好的模型和相关数据。
# 这个框架通过 _Trainer 类提供了通用的模型训练流程，PINNTrainer 类在此基础上实现了针对物理信息神经网络的训练过程。
class PINNTrainer(_Trainer):
    "Standard PINN model trainer class"
    
    def _train_step(self, model, optimizer, c, i, mu, sd, device):# use separate function to ensure computational graph/memory is released

        optimizer.zero_grad()
        #将优化器中存储的参数梯度置零，以准备接收新的梯度。
        model.train()
        #模型设置为训练模式，这通常涉及到一些训练中特定的行为，比如启用 Dropout 或 Batch Normalization。

        sampler = _x_random if c.RANDOM else _x_mesh
        x = sampler(c.SUBDOMAIN_XS, c.BATCH_SIZE, device).requires_grad_(True)
        #根据条件 c.RANDOM 选择采样器 _x_random 或 _x_mesh。这个采样器用于生成输入数据 x，可能基于随机采样或者是网格状的采样。
        #使用选定的采样器生成输入数据 x，并将其设置为需要梯度计算的状态 (requires_grad_(True))。
        y = model((x-mu)/sd)
        #将输入数据 x 标准化后输入到模型中，得到模型的输出 y。
        y = y*c.Y_N[1] + c.Y_N[0]
        #对模型的输出 y 进行缩放和偏移操作，以调整输出的范围。

        # get gradients
        yj = c.P.get_gradients(x, y)# problem-specific
        #：调用 c.P 中的方法，根据输入 x 和输出 y 获取梯度信息 yj。
        #return y, j2, jj0, jj1, jj2

        # apply boundary conditions
        yj = c.P.boundary_condition(x, *yj, *c.BOUNDARY_N)# problem-specific
        #应用边界条件，对梯度信息 yj 进行处理。这是问题特定的操作，用于确保模型在边界处的行为符合预期。
        # return y_new, j2, jj0_new, jj1_new, jj2_new

        # backprop loss
        loss = c.P.physics_loss(x, *yj)# problem-specific
        #计算物理损失，根据输入 x 和梯度信息 yj 计算模型损失值 loss。这是根据问题特定的物理规律定义的损失函数。
        loss.backward()#对损失值 loss 进行反向传播，计算模型参数的梯度。
        optimizer.step()#：根据计算的梯度更新模型的参数，执行优化器的参数更新步骤。

        if (i % c.SUMMARY_FREQ) == 0:
            print(*[t.shape for t in yj], x.shape)
        #检查是否达到了输出摘要的频率。如果是，就打印出 yj 中张量的形状以及输入 x 的形状。
        # return result
        return x.detach(), [t.detach() for t in yj], loss.item()
    #返回经过 detach() 处理的输入 x，梯度信息 yj 的张量列表（每个张量也经过 detach() 处理），以及损失值 loss 的数值部分。
    # 这样可以确保这些张量不再与计算图关联，避免梯度更新影响之后的计算。
    
    def _test_step(self, x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses):# use separate function to ensure computational graph/memory is released
        
        # get full model solution using test data
        yj_full, y_full_raw = full_model_PINN(x_test, model, c)
        print(x_test.shape, yj_true[0].shape, yj_full[0].shape)
        
        # get losses over test data
        yj_test_loss = [losses.l1_loss(a,b).item() for a,b in zip(yj_true, yj_full)]
        physics_loss = c.P.physics_loss(x_test, *yj_full).item()# problem-specific
        yj_test_losses.append([i + 1, mstep, fstep]+yj_test_loss+[physics_loss])
        for j,l in enumerate(yj_test_loss): 
            for step,tag in zip([i + 1, mstep, fstep], ["istep", "mstep", "zfstep"]):
                writer.add_scalar("loss_%s/yj%i/test"%(tag,j), l, step)
        writer.add_scalar("loss_istep/zphysics/test", physics_loss, i + 1)
        
        # PLOTS
        
        if (i + 1) % c.TEST_FREQ == 0:
            
            # save figures
            fs = plot_main.plot_PINN(x_test, yj_true,   x, yj,   yj_full, y_full_raw,   yj_test_losses,   c, i + 1)
            if fs is not None: self._save_figs(i, fs)
            
        del x_test, yj_true,   x, yj,   yj_full, y_full_raw# fixes weird over-allocation of GPU memory bug caused by plotting (?)
        
        return yj_test_losses
                
    def train(self):
        "Train model"

        #从 self 对象中获取 c、device 和 writer 属性。这些属性包含了训练所需的配置、设备信息和用于记录训练进度的写入器。

        c, device, writer = self.c, self.device, self.writer
        
        # define model
        #创建了一个特定类型的模型。c.MODEL 可能是一个模型类的引用，它接受输入参数来定义模型的架构。
        # 这些参数包括问题的特定维度、隐藏层的大小和神经网络层数。

        model = c.MODEL(c.P.d[0], c.P.d[1], c.N_HIDDEN, c.N_LAYERS)# problem-specific
        
        # create optimiser
        #使用Adam优化器初始化模型的优化器，以便在训练过程中更新模型参数。这里的学习率为c.LRATE。

        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE)
        
        # put model on device
        #将模型移动到指定的设备（如 GPU 或 CPU）上进行训练。

        model.to(device)
        
        # get normalisation values
        # 计算输入数据的最小值和最大值，然后计算出数据的均值mu和标准差sd，用于标准化输入数据。
        xmin, xmax = torch.tensor([[x.min(), x.max()] for x in c.SUBDOMAIN_XS], dtype=torch.float32, device=device).T
        mu = (xmin + xmax)/2; sd = (xmax - xmin)/2
        print(mu, sd, mu.shape, sd.shape)# (nd)# broadcast below
        
        # get exact solution if it exists
        #使用测试数据集 _x_mesh 函数创建测试数据 x_test。

        x_test = _x_mesh(c.SUBDOMAIN_XS, c.BATCH_SIZE_TEST, device)

        # 获取测试数据对应的真实解（如果存在）。这个方法可能是用来获取问题的确切解决方案。
        yj_true = c.P.exact_solution(x_test, c.BATCH_SIZE_TEST)# problem-specific
        
        ## TRAIN
        #接下来，代码进入训练循环 for i in range(c.N_STEPS):，其中 c.N_STEPS 是训练的总步数。

        mstep, fstep, yj_test_losses = 0, 0, []
        start, gpu_time = time.time(), 0.

        # 执行单个训练步骤，计算模型的输出、损失值和梯度，并更新模型的参数。
        # mstep 和 fstep 记录权重更新的数量和浮点运算数（FLOPS）。
        for i in range(c.N_STEPS):
            gpu_start = time.time()
            x, yj, loss = self._train_step(model, optimizer, c, i, mu, sd, device)
            mstep += model.size# record number of weights updated
            fstep += model.flops(x.shape[0])# record number of FLOPS
            gpu_time += time.time()-gpu_start
            
            
            # METRICS
            #如果满足 (i + 1) % c.SUMMARY_FREQ == 0，则进行如下操作：
            # 计算训练速率并输出摘要信息。
            # 调用 self._test_step() 方法，对模型在测试数据上进行评估，并记录测试损失。
            if (i + 1) % c.SUMMARY_FREQ == 0:
                
                # set counters
                rate, gpu_time = c.SUMMARY_FREQ / gpu_time, 0.
                
                # print summary
                self._print_summary(i, loss, rate, start)
                
                # test step
                yj_test_losses = self._test_step(x_test, yj_true,   x, yj,   model, c, i, mstep, fstep, writer, yj_test_losses)
            
            # SAVE
            #如果满足 (i + 1) % c.MODEL_SAVE_FREQ == 0，则保存模型和损失信息到文件中。
            if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                
                # save model and losses
                self._save_model(i, model)
                np.save(c.MODEL_OUT_DIR+"loss_%.8i.npy"%(i + 1), np.array(yj_test_losses))
        
        # cleanup
        writer.close()
        print("Finished training")
    
    
if __name__ == "__main__":
    #'''
    c = Constants(
                  N_LAYERS=2,
                  N_HIDDEN=16,
                  TEST_FREQ=2000,
                  RANDOM=True,
                  )
    run = PINNTrainer(c)
    '''
    c = Constants(
                  N_LAYERS=4,
                  N_HIDDEN=64,
                  TEST_FREQ=1000,
                  RANDOM=True,
                  )
    run = PINNTrainer(c)
    '''

    run.train()