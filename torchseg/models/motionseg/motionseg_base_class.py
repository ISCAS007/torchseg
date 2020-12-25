# -*- coding: utf-8 -*-

import torch.nn as nn
import warnings
from easydict import EasyDict as edict
from .motion_backbone import motion_backbone

def get_aux_input_channel(aux_input_format):
    aux_in_channels=0
    for c in aux_input_format:
        if c.lower()=='b':
            assert False
            # background image
            aux_in_channels+=3
        elif c.lower()=='o':
            # optical flow
            aux_in_channels+=2
        elif c.lower()=='n':
            # neighbor image
            aux_in_channels+=3
        elif c.lower()=='-':
            pass
        elif c.lower()=='g':
            # neighbor groundtruth
            aux_in_channels+=1
        else:
            assert False

    return aux_in_channels

class motion(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config.upsample_layer
        self.min_channel_number=config.min_channel_number
        self.max_channel_number=config.max_channel_number

        self.main_backbone=motion_backbone(config,use_none_layer=config.use_none_layer)
        self.get_aux_backbone()

        self.midnet_out_channels=min(max(self.main_backbone.get_feature_map_channel(self.upsample_layer),
                                     self.min_channel_number),self.max_channel_number)
        self.class_number=config.class_number

    def get_aux_backbone(self):
        aux_in_channels=get_aux_input_channel(self.config.input_format)

        assert aux_in_channels>0,'changenet must have aux input, please change the input_format: {}'.format(self.config.input_format)
        if aux_in_channels==0:
            self.use_aux_input=False
        else:
            self.use_aux_input=True

        if self.config.input_format.lower()=='n':
            self.share_backbone=self.config.share_backbone
        else:
            self.share_backbone=False
            if self.config.share_backbone:
                warnings.warn('share backbone not worked for {} with input_format {}'.format(self.config.net_name,self.config.input_format))

        self.aux_backbone=None

        if self.use_aux_input:
            if self.share_backbone:
                self.aux_backbone=self.main_backbone
                if self.config.aux_backbone is not None:
                    warnings.warn('aux backbone not worked when share_backbone')
                    if self.config.aux_backbone != self.config.backbone_name:
                        assert False,'conflict config for share_backbone={}, aux_backbone={} and backbone_name={}'.format(self.config.share_backbone,self.config.aux_backbone,self.config.backbone_name)
            else:
                aux_config=edict(self.config.copy())
                if aux_config.aux_backbone is None:
                    aux_config.aux_backbone=self.config.backbone_name

                aux_config.backbone_name=aux_config.aux_backbone
                aux_config.freeze_layer=0
                aux_config.backbone_pretrained=False
                self.aux_backbone=motion_backbone(aux_config,use_none_layer=aux_config.use_none_layer,in_channels=aux_in_channels)

        return self.aux_backbone

    def forward(self,imgs):
        pass