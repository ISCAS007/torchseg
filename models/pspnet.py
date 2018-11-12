# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net
from utils.torch_tools import freeze_layer
from utils.disc_tools import get_backbone_optimizer_params


class pspnet(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__

        if hasattr(self.config.model, 'use_momentum'):
            use_momentum = self.config.model.use_momentum
        else:
            use_momentum = False

        self.backbone = backbone(config.model, use_momentum=use_momentum)

        if self.config.model.backbone_freeze:
            freeze_layer(self.backbone)

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

        # for pretrained module, use small lr_mult=1
        # for modified module, use middle lr_mult=10
        # for new module, use largest lr_mult=20
        # for resnet, the begin layers is newed and the end layers is changed
        if self.config.model.backbone_freeze:
            self.optimizer_params = [{'params': self.midnet.parameters(),
                                      'lr_mult': 1},
                                     {'params': self.decoder.parameters(), 'lr_mult': 1}]
        elif config.model.use_lr_mult:
            if use_momentum and config.model.backbone_pretrained and self.upsample_layer >= 4:
                backbone_optmizer_params = get_backbone_optimizer_params(config.model.backbone_name,
                                                                         self.backbone.model,
                                                                         unchanged_lr_mult=1,
                                                                         changed_lr_mult=config.model.changed_lr_mult,
                                                                         new_lr_mult=config.model.new_lr_mult)
            else:
                backbone_optmizer_params = [{'params': [
                    p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1}]

            self.optimizer_params = backbone_optmizer_params + [{'params': self.midnet.parameters(), 'lr_mult':                      config.model.new_lr_mult},
                                                                {'params': self.decoder.parameters(), 'lr_mult': config.model.new_lr_mult}]
        else:
            self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1},
                                     {'params': self.midnet.parameters(),
                                      'lr_mult': 1},
                                     {'params': self.decoder.parameters(), 'lr_mult': 1}]
            

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        x = self.decoder(feature_mid)

        return x
