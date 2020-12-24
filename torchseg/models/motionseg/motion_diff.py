# -*- coding: utf-8 -*-
"""
input: I1+I2+G1 or I1+I2
output: G2-G1, G1-G2, G1 union G2

note: use sigmoid as the final activation, need use specific loss: nn.BCEWithLogitsLoss()
run: python test/fbms_train.py --net_name motion_diff --input_format ng --dataset cdnet2014/FBMS --note test
based on motion_panet2.
"""

import torch.nn as nn
import warnings
from .motion_backbone import (motion_backbone,
                                              motionnet_upsample_bilinear
                                              )
from .motion_panet import panet,get_input_channel,transform_panet2
from easydict import EasyDict as edict

class motion_diff(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config.upsample_layer
        self.min_channel_number=config.min_channel_number
        self.max_channel_number=config.max_channel_number

        if config.net_name.find('flow')>=0:
            self.use_flow=True
            self.share_backbone=False
            if config.share_backbone:
                warnings.warn('share backbone not worked for {}'.format(config.net_name))
        else:
            self.use_flow=False
            self.share_backbone=config.share_backbone

        self.main_backbone=motion_backbone(config,use_none_layer=config.use_none_layer)

        aux_in_channels=get_input_channel(self.config.input_format)

        if self.config.input_format.lower()!='n':
            self.share_backbone=False
            if config.share_backbone:
                warnings.warn('share backbone not worked for {}'.format(config.net_name))
        else:
            self.share_backbone=config.share_backbone

        if aux_in_channels==0:
            self.use_aux_input=False
        else:
            self.use_aux_input=True

        self.aux_backbone=None
        if self.share_backbone:
            if self.use_aux_input:
                self.aux_backbone=self.main_backbone

            if config.aux_backbone is not None:
                warnings.warn('aux backbone not worked when share_backbone')
                if config.aux_backbone != config.backbone_name:
                    assert False
        else:
            if self.use_aux_input:
                if config.aux_backbone is None:
                    config.aux_backbone=config.backbone_name
                aux_config=edict(config.copy())
                aux_config.backbone_name=aux_config.aux_backbone
                aux_config.freeze_layer=0
                aux_config.backbone_pretrained=False
                self.aux_backbone=motion_backbone(aux_config,use_none_layer=config.use_none_layer,in_channels=aux_in_channels)

        self.main_panet=None
        if config.main_panet:
            self.main_panet=panet(config,in_c=3)

        self.aux_panet=None
        if config.aux_panet:
            if self.share_backbone and self.main_panet is not None:
                self.aux_panet=self.main_panet
            else:
                self.aux_panet=panet(config,in_c=aux_in_channels)

        self.get_midnet()
        self.midnet_out_channels=min(max(self.main_backbone.get_feature_map_channel(self.upsample_layer),
                                     self.min_channel_number),self.max_channel_number)
        self.class_number=config.class_number

        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                 out_channels=3,
                                                 output_shape=self.input_shape[0:2])
        self.activate=nn.Sigmoid()

    def get_midnet(self):
        keys=['main_backbone','aux_backbone','main_panet','aux_panet']
        none_backbones={'main_backbone':self.main_backbone,
                   'aux_backbone':self.aux_backbone,
                   'main_panet':self.main_panet,
                   'aux_panet':self.aux_panet}
        backbones={}
        for key in keys:
            if none_backbones[key] is not None:
                backbones[key]=none_backbones[key]

        self.midnet=transform_panet2(backbones,self.config)

    def forward(self,imgs):
        features={}
        features['main_backbone']=self.main_backbone.forward_layers(imgs[0])
        if self.use_aux_input or self.use_flow:
            features['aux_backbone']=self.aux_backbone.forward_layers(imgs[1])

        if self.config.main_panet:
            features['main_panet']=self.main_panet.forward_layers(imgs[0])

        if self.config.aux_panet:
            features['aux_panet']=self.aux_panet.forward_layers(imgs[1])

        x=self.midnet(features)
        y=self.decoder(x)
        return {'masks':[y]}


