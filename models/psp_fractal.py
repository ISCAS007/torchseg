# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from torch.autograd import Variable
from models.backbone import backbone
from utils.metrics import runningScore
from utils.torch_tools import freeze_layer
from models.upsample import transform_fractal,transform_psp,upsample_duc,upsample_bilinear
import numpy as np
from tensorboardX import SummaryWriter
import time
import os


class psp_fractal(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__
        self.backbone = backbone(config)

        if hasattr(self.config, 'backbone_lr_ratio'):
            backbone_lr_raio = self.config.backbone_lr_ratio
            if backbone_lr_raio == 0:
                freeze_layer(self.backbone)

        self.upsample_type = self.config.upsample_type
        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset.name
#        self.midnet_type = self.config.midnet_type
        self.midnet_pool_sizes = self.config.midnet_pool_sizes
        self.midnet_scale = self.config.midnet_scale

        self.midnet_in_channels = self.backbone.get_feature_map_channel(
            self.upsample_layer)
        self.midnet_out_channels = self.config.midnet_out_channels
        self.midnet_out_size = self.backbone.get_feature_map_size(
            self.upsample_layer, self.input_shape[0:2])

        self.midnet = transform_psp(self.midnet_pool_sizes,
                                    self.midnet_scale,
                                    self.midnet_in_channels,
                                    self.midnet_out_channels,
                                    self.midnet_out_size)

        self.before_upsample=self.config.before_upsample
        self.fractal_depth=self.config.fractal_depth
        self.fractal_fusion_type=self.config.fractal_fusion_type
        if self.before_upsample:
            self.fractal_net=transform_fractal(in_channels=2*self.midnet_out_channels,
                                               depth=self.fractal_depth,
                                               class_number=self.class_number,
                                               fusion_type=self.fractal_fusion_type,
                                               do_fusion=True)
            # psp net will output channels with 2*self.midnet_out_channels
            if self.upsample_type == 'duc':
                r = 2**self.upsample_layer
                self.seg_decoder = upsample_duc(
                    self.class_number, self.class_number, r)
            elif self.upsample_type == 'bilinear':
                self.seg_decoder = upsample_bilinear(
                    self.class_number, self.class_number, self.input_shape[0:2])
            else:
                assert False, 'unknown upsample type %s' % self.upsample_type
        else:
            # psp net will output channels with 2*self.midnet_out_channels
            if self.upsample_type == 'duc':
                r = 2**self.upsample_layer
                self.seg_decoder = upsample_duc(
                    2*self.midnet_out_channels, self.class_number, r)
            elif self.upsample_type == 'bilinear':
                self.seg_decoder = upsample_bilinear(
                    2*self.midnet_out_channels, self.class_number, self.input_shape[0:2])
            else:
                assert False, 'unknown upsample type %s' % self.upsample_type

            self.fractal_net=transform_fractal(in_channels=self.class_number,
                                               depth=self.fractal_depth,
                                               class_number=self.class_number,
                                               fusion_type=self.fractal_fusion_type,
                                               do_fusion=True)

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        x = self.midnet(feature_map)
        if self.before_upsample:
            x=self.fractal_net(x)
            x=self.seg_decoder(x)
        else:
            x=self.seg_decoder(x)
            x=self.fractal_net(x)
        return x