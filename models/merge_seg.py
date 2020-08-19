# -*- coding: utf-8 -*-
import torch
import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net, conv_bn_relu
from utils.torch_tools import freeze_layer

class merge_seg(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config)

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

        # out feature channels 512
        self.branch_edge=get_suffix_net(
                self.config, self.midnet_out_channels, self.edge_class_num)
        # out feature channels 512
        self.branch_seg=get_suffix_net(
                self.config, self.midnet_out_channels, self.class_number)
        # input=concat(512,512)
        self.feature_conv = conv_bn_relu(in_channels=512+512,
                                             out_channels=512,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

        self.seg=get_suffix_net(
                self.config, 512, self.class_number)

    def forward(self,x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)

        branch_edge_feature, branch_edge = self.branch_edge(feature_mid,need_upsample_feature=True)
        branch_seg_feature, branch_seg = self.branch_seg(feature_mid,need_upsample_feature=True)

        merge_feature=torch.cat([branch_edge_feature,branch_seg_feature],dim=1)
        x=self.feature_conv(merge_feature)
        seg=self.seg(x)

        # how to summery image, metrics, with branch show in branch, otherwise will show in root dir
        # for seg will run with seg, for edge will run with edge
        return {'seg_1':branch_seg,'edge':branch_edge,'seg':seg}