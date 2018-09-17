# -*- coding: utf-8 -*-
import torch
import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net
from tensorboardX import SummaryWriter
from utils.metrics import runningScore
from utils.torch_tools import get_optimizer
import numpy as np
import time
import os


class psp_aux(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__

        use_momentum = self.config.model.use_momentum if hasattr(
            self.config.model, 'use_momentum') else False
        self.backbone = backbone(config.model, use_momentum=use_momentum)
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index

        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
        self.auxnet_layer = self.config.model.auxnet_layer
        self.auxnet_input_shape = self.backbone.get_output_shape(
            self.auxnet_layer, self.input_shape)
        self.midnet_out_channels = 2*self.midnet_input_shape[1]
        self.auxnet_out_channels = self.auxnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        self.auxnet = get_suffix_net(self.config,
                                     self.auxnet_out_channels,
                                     self.class_number,
                                     aux=True)

        self.decoder = get_suffix_net(self.config,
                                      self.midnet_out_channels,
                                      self.class_number)

        self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1},
                                 {'params': self.midnet.parameters(), 'lr_mult': 10},
                                 {'params': self.auxnet.parameters(), 'lr_mult': 20},
                                 {'params': self.decoder.parameters(), 'lr_mult': 20}]

        print('class number is %d' % self.class_number,
              'ignore_index is %d' % self.ignore_index, '*'*30)

    def forward(self, x):
        main, aux = self.backbone.forward_aux(
            x, self.upsample_layer, self.auxnet_layer)
#        print('main,aux shape is',main.shape,aux.shape)
        main = self.midnet(main)
        main = self.decoder(main)
        aux = self.auxnet(aux)

        return main, aux