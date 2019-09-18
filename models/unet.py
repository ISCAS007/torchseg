# -*- coding: utf-8 -*-

import torch.nn as nn
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net
from utils.torch_tools import freeze_layer
from utils.disc_tools import get_backbone_optimizer_params
from models.custom_layers import MergeLayer,CascadeMergeLayer
from models.motionseg.motion_backbone import motion_backbone
from easydict import EasyDict as edict
import warnings

class UNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__

        assert self.config.backbone_name.find('vgg')>=0, 'UNet current only support vgg series'
        use_none_layer=config.use_none_layer
        self.backbone = backbone(config, use_none_layer=use_none_layer)

        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset_name
        self.ignore_index = self.config.ignore_index

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
#        self.midnet_out_channels=self.config.midnet_out_channels
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        backbone_channels_list=[self.backbone.get_feature_map_channel(i+1) for i in range(5)]

        b,c,h,w=self.midnet_input_shape
        self.merge_layer=MergeLayer(self.config,backbone_channels_list,(b,self.midnet_out_channels,h,w))

        self.decoder = get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        features=self.backbone.forward_layers(x)
        ## for vgg layers, len(features)=6, [x,x/2,x/4,x/8,x/16,x/32]
        features=features[-5:]
        x1=self.merge_layer(features)
        x2=self.midnet(features[self.upsample_layer-1])
        x=self.decoder(x1+x2)
        return x

class PSPUNet(nn.Module):
    def __init__(self,config):
        super().__init__()
#        config.deconv_layer=5
#        config.upsample_layer=1
        config.min_channel_number=128
        config.max_channel_number=256
        config.decode_main_layer=1
        self.config=config
        self.name=self.__class__.__name__
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset_name
        self.ignore_index = self.config.ignore_index
        self.min_channel_number=self.config.min_channel_number
        self.max_channel_number=self.config.max_channel_number

#        new_config=edict(config.copy())
        self.backbone=motion_backbone(config,use_none_layer=config.use_none_layer)
        self.midnet=CascadeMergeLayer(self.backbone,config)

        self.midnet_out_channels=min(max(self.backbone.get_feature_map_channel(config.upsample_layer),
                      self.min_channel_number),self.max_channel_number)
        self.decoder=get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)
    def forward(self,x):
        feature_map = self.backbone.forward_layers(x)
        feature_mid = self.midnet(feature_map)
        x = self.decoder(feature_mid)

        return x