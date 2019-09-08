# -*- coding: utf-8 -*-
"""
backbone.py: custom backbone
custom_layers.py: custom layers
upsample.py: custom layers
"""

import torch
import torch.nn as nn

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, main_channel,reduction=16,attention_channel=None):
        super(CALayer, self).__init__()
        if attention_channel is None:
            attention_channel=main_channel
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, main_channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.avg_pool(attention_feature)
        y = self.conv(y)
        return main_feature * y

## Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self,main_channel,reduction=16,attention_channel=None):
        super().__init__()
        if attention_channel is None:
            attention_channel=main_channel

        mid_c=max(8,main_channel//reduction)
        self.conv = self.conv = nn.Sequential(
                        nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, 1, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    )

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.conv(attention_feature)
        return main_feature * y

## Global Attention (GA) Layer
class GALayer(nn.Module):
    def __init__(self,main_channel,ga_channel=32,reduction=16,attention_channel=None):
        super().__init__()
        if attention_channel is None:
            attention_channel=main_channel

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                        nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, ga_channel, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    )

        self.ga_conv = nn.Sequential(
                        nn.Conv2d(main_channel+ga_channel, main_channel, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.avg_pool(attention_feature)
        y = self.conv(y)
        b,c,h,w=main_feature.shape
        y = y.repeat(1,1,h,w)
        y = torch.cat([main_feature,y],dim=1)
        y = self.ga_conv(y)
        return y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class transform_attention(nn.Module):
    def __init__(self,config,main_c,filter_c=None):
        super().__init__()
        self.config=config
        if filter_c is None:
            filter_c=main_c

        if 's' in self.attention_type:
            self.spatial_filter=SALayer(main_channel=main_c,attention_channel=filter_c)

        if 'c' in self.attention_type:
            self.channel_filter=CALayer(main_channel=main_c,attention_channel=filter_c)

        if 'g' in self.attention_type:
            self.global_filter=GALayer(main_channel=main_c,attention_channel=filter_c)

    def forward(self,x):
        for c in self.attention_type:
            if c=='s':
                x=self.spatial_filter(x)
            elif c=='c':
                x=self.channel_filter(x)
            elif c=='g':
                x=self.global_filter(x)
            elif c=='n':
                pass
            else:
                assert False,'unknonw attention type {}'.format(self.attention_type)

        return x