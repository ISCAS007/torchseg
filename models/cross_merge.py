# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net, conv_bn_relu
from utils.torch_tools import freeze_layer

class cross_merge(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config.model)
        
        if hasattr(self.config.model, 'backbone_freeze'):
            if self.config.model.backbone_freeze:
                print('freeze backbone weights'+'*'*30)
                freeze_layer(self.backbone)
        
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index
        self.edge_class_num = self.config.dataset.edge_class_num
        self.cross_merge_times=self.config.model.cross_merge_times

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)
    
        self.seg0 = get_suffix_net(
            self.config, self.midnet_out_channels, self.class_number)
        self.edge0 = get_suffix_net(
            self.config, self.midnet_out_channels, self.edge_class_num)

        seg_list = []
        edge_list= []
        # before concat
        seg_conv_list=[]
        edge_conv_list=[]
        
        feature_channel=512
        concat_channel=feature_channel//2
        #TODO use psp other than conv
        for i in range(self.cross_merge_times):
            seg_conv_list.append(conv_bn_relu(in_channels=feature_channel,
                                              out_channels=concat_channel,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
            edge_conv_list.append(conv_bn_relu(in_channels=feature_channel,
                                              out_channels=concat_channel,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
            seg_list.append(get_suffix_net(
                self.config, 2*concat_channel, self.class_number))
            edge_list.append(get_suffix_net(
                self.config, 2*concat_channel, self.edge_class_num))
            
        self.seg_list = TN.ModuleList(seg_list)
        self.edge_list = TN.ModuleList(edge_list)
        self.seg_conv_list = TN.ModuleList(seg_conv_list)
        self.edge_conv_list = TN.ModuleList(edge_conv_list)
        
    def forward(self,x):
        outputs_dict={}
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        
        edge_fi, edge_i = self.edge0(feature_mid,need_upsample_feature=True)
        seg_fi, seg_i = self.seg0(feature_mid,need_upsample_feature=True)
        for i in range(self.cross_merge_times):
            outputs_dict['edge_%d'%i]=edge_i
            outputs_dict['seg_%d'%i]=seg_i
            
            edge_fi=self.edge_conv_list[i](edge_fi)
            seg_fi=self.seg_conv_list[i](seg_fi)
            x=torch.cat([edge_fi,seg_fi],dim=1)
            seg_fi,seg_i=self.seg_list[i](x,need_upsample_feature=True)
            edge_fi,edge_i=self.edge_list[i](x,need_upsample_feature=True)
            
        outputs_dict['edge']=edge_i
        outputs_dict['seg']=seg_i
            
        return outputs_dict