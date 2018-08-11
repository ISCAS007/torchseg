# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net


class pspnet(TN.Module):
    def __init__(self, config):
        super(pspnet, self).__init__()
        self.config = config
        self.name = self.__class__.__name__

        if hasattr(self.config.model, 'use_momentum'):
            use_momentum = self.config.model.use_momentum
        else:
            use_momentum = False

        self.backbone = backbone(config.model, use_momentum=use_momentum)

        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
#        self.midnet_out_channels=self.config.model.midnet_out_channels
        self.midnet_out_channels = 2*self.midnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        self.decoder = get_suffix_net(config,
                                      self.midnet_out_channels,
                                      self.class_number)
        
        self.optimizer_params = [{'params': self.backbone.parameters(), 'lr_mult': 1},
                                 {'params': self.midnet.parameters(),
                                  'lr_mult': 10},
                                 {'params': self.decoder.parameters(), 'lr_mult': 20}]

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        x = self.decoder(feature_mid)

        return x
