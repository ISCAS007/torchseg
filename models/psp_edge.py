# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from models.backbone import backbone
from utils.metrics import runningScore
from models.upsample import get_midnet, get_suffix_net, conv_bn_relu
import numpy as np
from tensorboardX import SummaryWriter
from utils.torch_tools import get_optimizer, poly_lr_scheduler, freeze_layer
from dataset.dataset_generalize import image_normalizations
import json
import time
import os


class psp_edge(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__
        self.backbone = backbone(config)
        if hasattr(self.config, 'backbone_freeze'):
            if self.config.backbone_freeze:
                print('freeze backbone weights'+'*'*30)
                freeze_layer(self.backbone)

        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index
        self.edge_class_num = self.config.dataset.edge_class_num

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        if hasattr(self.config, 'edge_seg_order'):
            self.edge_seg_order = self.config.edge_seg_order
            print('the edge and seg order is %s'%self.edge_seg_order,'*'*30)
            assert self.edge_seg_order in ['same','first','later'],'unexcepted edge seg order %s'%self.edge_seg_order
        else:
            self.edge_seg_order = 'same'

        if self.edge_seg_order == 'same':
            self.seg_decoder = get_suffix_net(
                self.config, self.midnet_out_channels, self.class_number)
            self.edge_decoder = get_suffix_net(
                self.config, self.midnet_out_channels, self.edge_class_num)
        elif self.edge_seg_order == 'later':
            self.seg_decoder = get_suffix_net(
                self.config, self.midnet_out_channels, self.class_number)
            self.edge_decoder = get_suffix_net(
                self.config, 512, self.edge_class_num)
        else:
            self.edge_decoder = get_suffix_net(
                self.config, self.midnet_out_channels, self.edge_class_num)
            self.feature_conv = conv_bn_relu(in_channels=self.midnet_out_channels,
                                             out_channels=512,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
            # the input is torch.cat[self.edge_class_num,self.class_number]
            self.seg_conv = conv_bn_relu(in_channels=512+512,
                                         out_channels=512,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
            self.seg_decoder = get_suffix_net(
                self.config, 512, self.class_number)


    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)

        if self.edge_seg_order == 'same':
            seg = self.seg_decoder(feature_mid)
            edge = self.edge_decoder(feature_mid)
        elif self.edge_seg_order == 'later':
            seg_feature, seg = self.seg_decoder(feature_mid,need_upsample_feature=True)
            edge = self.edge_decoder(seg_feature)
        else:
            edge_feature,edge = self.edge_decoder(feature_mid,need_upsample_feature=True)
            x_feature = self.feature_conv(feature_mid)
            x_merge = torch.cat([edge_feature, x_feature], dim=1)
            x_seg = self.seg_conv(x_merge)
            seg = self.seg_decoder(x_seg)

        return {'seg':seg, 'edge':edge}