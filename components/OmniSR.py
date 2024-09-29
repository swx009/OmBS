#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import  torch
import  torch.nn as nn
from    ops.OSAG import OSAG
from    ops.pixelshuffle import pixelshuffle_block
import  torch.nn.functional as F


class AdaptiveJointAttention(nn.Module):       
    def __init__(self, in_channels):
        super(AdaptiveJointAttention, self).__init__()
        self.spatial_gate = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.channel_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
   
        spatial_att = torch.sigmoid(self.spatial_gate(x))
        channel_att = torch.sigmoid(self.channel_gate(F.adaptive_avg_pool2d(x, (1, 1))))
        weights = self.adaptive_weights(x)
        combined_att = weights[:, 0:1, None, None] * spatial_att + weights[:, 1:2, None, None] * channel_att
        return x * combined_att


class EdgeEnhancement(nn.Module):
    def __init__(self, in_channels):
        super(EdgeEnhancement, self).__init__()
        self.edge_detector = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        new_weights = torch.full_like(self.edge_detector.weight, -1)
        for i in range(in_channels):
            new_weights[i, i, 1, 1] = 8
        self.edge_detector.weight = torch.nn.Parameter(new_weights)

    def forward(self, x):
        edge_features = F.relu(self.edge_detector(x))
        return x + edge_features



class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()
        res_num = kwargs["res_num"]
        up_scale = kwargs["upsampling"]
        bias = kwargs["bias"]
        self.adaptive_joint_attention = AdaptiveJointAttention(num_feat)
        self.edge_enhancement = EdgeEnhancement(num_feat)

        residual_layer = []
        self.res_num = res_num
        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)
        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.input(x)
        x = self.adaptive_joint_attention(x)

        x = self.residual_layer(x)

        x = self.edge_enhancement(x)

        x = torch.add(self.output(x), x)
        x = self.up(x)
        x = x[:, :, :H * self.up_scale, :W * self.up_scale]
        return x

