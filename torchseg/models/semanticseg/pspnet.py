# -*- coding: utf-8 -*-

import torch.nn as TN
from ..backbone import backbone
from ..upsample import get_midnet, get_suffix_net
from ...utils.disc_tools import get_backbone_optimizer_params
import warnings

class pspnet(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name=self.__class__.__name__

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

        self.decoder = get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)

        self.center_channels=self.decoder.center_channels

        # for pretrained module, use small lr_mult=1
        # for modified module, use middle lr_mult=10
        # for new module, use largest lr_mult=20
        # for resnet, the begin layers is newed and the end layers is changed
        if self.config.backbone_freeze:
            self.optimizer_params = [{'params': self.midnet.parameters(),
                                      'lr_mult': 1},
                                     {'params': self.decoder.parameters(), 'lr_mult': 1}]
        elif config.use_lr_mult:
            if use_none_layer and config.backbone_pretrained and self.upsample_layer >= 4:
                backbone_optmizer_params = get_backbone_optimizer_params(config.backbone_name,
                                                                         self.backbone,
                                                                         unchanged_lr_mult=config.pre_lr_mult,
                                                                         changed_lr_mult=config.changed_lr_mult,
                                                                         new_lr_mult=config.new_lr_mult)

            else:
                warnings.warn('config.use_lr_mult is True but not fully wored')
                backbone_optmizer_params = [{'params': [
                    p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': config.pre_lr_mult}]

            self.optimizer_params = backbone_optmizer_params + [{'params': self.midnet.parameters(), 'lr_mult':                      config.new_lr_mult},
                                                                {'params': self.decoder.parameters(), 'lr_mult': config.new_lr_mult}]
        else:
            self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1},
                                     {'params': self.midnet.parameters(),
                                      'lr_mult': 1},
                                     {'params': self.decoder.parameters(), 'lr_mult': 1}]


    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        x = self.decoder(feature_mid)
        self.center_feature=self.decoder.center_feature
        return x
