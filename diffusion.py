# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agents.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,#这个啥
                            Losses)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):#这个model是训练加噪去噪能力的扩散
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        #predict_epilon:是否预测噪声。
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model#MLP结构

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)#a拔#0-当前的累乘
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        alphas_cumprod_1 = alphas_cumprod/alphas
        #0-当前前一步的累成
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('sqrt_one_minus_betas', torch.sqrt(1.-betas))
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('one_minus_alphas_cumprod_a1', (1. -alphas_cumprod/alphas))
        self.register_buffer('sqrt_alphas_cumprod_a1',torch.sqrt(alphas_cumprod/alphas))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',#就是求后验xt-1的时候推导的  x0的系数
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',  #xt前面的系数
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''predict_start_from_noise 方法根据当前的噪声x预测起始状态 x0。
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:#这个控制输出是噪声还是直接x0
            return (  ##extract函数从张量a中按照索引t提取元素，并将其重塑为形状x_shape。
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):#用在后面计算模型均值和方差 不太懂
        posterior_mean = (#q_posterior 方法计算给定当前状态 x_t 和起始状态 x_start 的后验分布。
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
#返回 均值 方差 剪裁log后的方差
    def p_mean_variance(self, x, t, s):
        #p_mean_variance 方法计算模型的均值和方差。
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        #p_sample 方法根据当前状态和时间步采样一个动作。
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
#----自己写的一步采样的：
    def p2_sample(self,x,t,s):
        b, *_, device = *x.shape, x.device
        noise_pre= self.model(x,t,s)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return (extract(self.sqrt_recip_alphas_cumprod,t,x.shape)*x-extract(self.sqrt_recipm1_alphas_cumprod,t,x.shape)*noise_pre)
    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device
#p_sample_loop 方法通过循环执行 p_sample 方法，从噪声中重构原始数据
        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    def p2_sample_loop(self,state,shape, verbose=False, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x= torch.randn(shape,device=device)
        progress = Progress(self.n_timesteps) if verbose else Silent()
        timesteps = torch.full((batch_size,), self.n_timesteps-1, device=device, dtype=torch.long)
        x = self.p2_sample(x, timesteps, state)
        progress.update({'t': 0})
        progress.close()
        return x
    #求熵 1：
    def log_forward_transition(self, x1, x2, t1, t2):  # t1 < t2
        log_p = torch.log(2.0 * math.pi * (1.0-extract(self.))) \
                + (x1 - x2 * torch.exp(-tb)) ** 2 / (1 - torch.exp(-2 * tb))
        return -0.5 * log_p
    def log_f_1_t(self,x1,xt):
        log_p = torch.log(2.0 * math.pi * (extract(self.one_minus_alphas_cumprod_a1, self.n_timesteps - 1, x1.shape))) \
                + (xt - x1 *(extract(self.sqrt_alphas_cumprod_a1,self.n_timesteps-1,xt.shape)) ** 2 / (extract(self.one_minus_alpha_cumprod_a1,self.n_timesteps,x1.shape)))
        return -0.5 * log_p
    def log_f_0_1(self,x0,x1):
        log_p = torch.log(2.0 * math.pi * (extract(self.betas, 1, x1.shape))) \
                + (x1 - x0 * (extract(self.sqrt_one_minus_betas, 1, x1.shape)) ** 2 / (
            extract(self.betas, 1, x1.shape)))
        return -0.5 * log_p
    def log_f_0_t(self,x0,xt):
        log_p = torch.log(2.0 * math.pi * (extract(self.one_minus_alphas_cumprod, self.n_timesteps - 1, x0.shape))) \
                + (xt - x0 *(extract(self.sqrt_alphas_cumprod,self.n_timesteps-1,xt.shape)) ** 2 / (extract(self.one_minus_alpha_cumprod,self.n_timesteps-1,x0.shape)))
        return -0.5 * log_p
    def log_reverse_transition(self, x_0, x1, xt):  # t1 < t2 x:a0 a1 at
        return self.log_f_1_t(x1, xt) \
            + self.log_f_0_1(x_0, x1) \
            - self.log_f_0_t(x_0, xt)
    def sample_noise(self, tensor):
        batch_size = tensor.shape[0]
        return torch.randn(batch_size, self.action_dim).to(tensor.device)

    def entropy(self, a_0):
            a_1 = self.q_sample(a_0,1)
            log_p = self.log_reverse_transition(a_0, a_1, self.sample_noise(a_0))
            return -log_p.sum(dim=1, keepdim=True)
    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):#通过状态采样动作？
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p2_sample_loop(state, shape, *args, **kwargs)

        return action.clamp_(-self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):#前向
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
      #一键加噪
        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)#随便取个高斯噪声
#加噪
        #前向
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #逆向
        x_recon = self.model(x_noisy, t, state)#预测的噪声

        assert noise.shape == x_recon.shape

        if self.predict_epsilon: #动作的loss或者噪声的loss
            loss = self.loss_fn(x_recon, noise, weights)#噪声的loss
        else:
            loss = self.loss_fn(x_recon, x_start, weights)#动作的loss？

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        #随便取一个值作为t 训练吧
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

