# -*- coding: utf-8 -*-
import torchvision as TV
import torch.nn as TN
import numpy as np
import torch.nn.functional as F
import torch

class upsample_duc(TN.Module):
    def __init__(self,in_channels,out_channels,upsample_ratio):
        """
        out_channels: class number
        """
        super(upsample_duc,self).__init__()
        self.duc=TN.PixelShuffle(upsample_ratio)
        self.duc_conv=TN.Conv2d(in_channels=in_channels,
                                out_channels=out_channels*upsample_ratio*upsample_ratio,
                                kernel_size=3,
                                padding=1,
                                stride=1)
        self.duc_norm=TN.BatchNorm2d(num_features=out_channels*upsample_ratio*upsample_ratio)
        
    def forward(self,x):
        x=self.duc_conv(x)
        x=self.duc_norm(x)
        x=F.relu(x)
        x=self.duc(x)
        
        return x
    
class upsample_bilinear(TN.Module):
    def __init__(self,in_channels,out_channels,output_shape):
        """
        out_channels: class number
        """
        self.output_shape=output_shape
        self.duc_conv=TN.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                padding=1,
                                stride=1)
        self.duc_norm=TN.BatchNorm2d(num_features=out_channels)
    
    def forward(self,x):
        x=self.duc_conv(x)
        x=self.duc_norm(x)
        x=F.relu(x)
        x = F.upsample(x, size=self.output_shape, mode='bilinear')
        return x
# TODO
class transform_psp(TN.Module):
    def __init__(self,pool_sizes):
        self.pool_paths=[]
        for pool_size in pool_sizes:
            self.pool_paths.append(TN.Sequential())
    
    def forward(self,x):
        
        return x

class conv2DBatchNormRelu(TN.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = TN.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = TN.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn:
            self.cbr_unit = TN.Sequential(conv_mod,
                                          TN.BatchNorm2d(int(n_filters)),
                                          TN.ReLU(inplace=True),)
        else:
            self.cbr_unit = TN.Sequential(conv_mod,
                                          TN.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class pyramidPooling(TN.Module):
    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, with_bn=with_bn))

        self.path_module_list = TN.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.training or self.model_name != 'icnet': # general settings or pspnet
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h/pool_size), int(w/pool_size)))
                strides.append((int(h/pool_size), int(w/pool_size)))
        else: # eval mode and icnet: pre-trained for 1025 x 2049
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                pp_sum = pp_sum + out

            return pp_sum
        