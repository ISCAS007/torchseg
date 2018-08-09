# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_suffix_net

class fcn(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        
        if hasattr(self.config.model,'use_momentum'):
            use_momentum=self.config.model.use_momentum
        else:
            use_momentum=False
        
        self.backbone=backbone(config.model,use_momentum=use_momentum)
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.midnet_out_channels=self.midnet_input_shape[1]
        self.decoder=get_suffix_net(config,
                                    self.midnet_out_channels,
                                    self.class_number)
        
    def forward(self,x):
        x=self.backbone.forward(x,self.upsample_layer)
        x=self.decoder(x)
        return x