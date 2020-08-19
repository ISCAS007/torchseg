# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
import torch.nn.functional as F
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net
from utils.torch_tools import freeze_layer

class psp_hed(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__

        use_none_layer=config.use_none_layer
        self.backbone = backbone(config, use_none_layer=use_none_layer)

        if hasattr(self.config,'backbone_freeze'):
            if self.config.backbone_freeze:
#                print('freeze backbone weights'+'*'*30)
                freeze_layer(self.backbone)

        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index
        self.edge_class_num=self.config.dataset.edge_class_num

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
#        self.midnet_out_channels=self.config.midnet_out_channels
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        self.decoder = get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)

        layer_shapes=self.backbone.get_layer_shapes(self.input_shape)
        print('layer shapes',layer_shapes)
        edge_aux_list=[]
        for i in range(self.upsample_layer-1):
            edge_aux_list.append(TN.Conv2d(in_channels=layer_shapes[i+1][1],
                                           out_channels=self.edge_class_num,
                                           kernel_size=1))
        self.edge_aux_list=TN.ModuleList(edge_aux_list)
        self.edge_fusion_conv=TN.Conv2d(in_channels=self.edge_class_num*(self.upsample_layer-1),
                                   out_channels=self.edge_class_num,
                                   kernel_size=1)

        self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1},
                                 {'params': self.edge_aux_list.parameters(), 'lr_mult':1},
                                 {'params': self.edge_fusion_conv.parameters(), 'lr_mult':1},
                                 {'params': self.midnet.parameters(),'lr_mult': 10},
                                 {'params': self.decoder.parameters(), 'lr_mult': 10}]

    def forward(self, x):
        features_map = self.backbone.forward_layers(x)
        feature_mid = self.midnet(features_map[self.upsample_layer])

        results={}
        results['seg'] = self.decoder(feature_mid)

        b,h,w,c=feature_mid.shape
        edge_aux_list=[]
        for i in range(self.upsample_layer-1):
#            print('feature shapes',features_map[i].shape)
            edge_aux=F.upsample(self.edge_aux_list[i](features_map[i+1]),size=(h,w),mode='bilinear',align_corners=True)
            edge_aux_list.append(edge_aux)

        edge_fusion=self.edge_fusion_conv(torch.cat(edge_aux_list,dim=1))

        for i in range(self.upsample_layer-1):
            results['edge_%d'%i]=F.upsample(edge_aux_list[i],size=self.input_shape[0:2],mode='bilinear',align_corners=True)

        results['edge']=F.upsample(edge_fusion,size=self.input_shape[0:2],mode='bilinear',align_corners=True)
        return results
