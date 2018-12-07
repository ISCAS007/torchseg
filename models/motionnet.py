# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import transform_segnet,get_suffix_net

class motionnet(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        
        print('warnniing: use_none_layer is false for motionnet')
        self.backbone=backbone(config.model,use_none_layer=config.model.use_none_layer)
        
        self.upsample_type = self.config.model.upsample_type
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        self.midnet=transform_segnet(self.backbone,self.config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.decoder=get_suffix_net(config,
                                    self.midnet_out_channels,
                                    self.class_number)

    def forward(self, x):        
        feature_map=self.backbone.forward_layers(x)
        feature_transform=self.midnet.forward(feature_map)
        y=self.decoder(feature_transform)
        return y