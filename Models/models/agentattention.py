
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import  DropPath, trunc_normal_
from timm.models.registry import register_model

class AgentAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=25, window=5, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                             padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
        self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.ac_bias, std=.02)
        trunc_normal_(self.ca_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        agent_tokens = self.pool(q[:, :, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        # position_bias = position_bias1

        # position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        # agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        # v_ = v[:, :, :, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x[:, :, :] = x[:, :, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n - 1, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
if __name__ == '__main__':
    a=AgentAttention(dim=640)
    input=torch.rand(75,25,640)
    out=a(input)
    print(out.shape)