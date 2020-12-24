# -*- coding: utf-8 -*-
"""
support net_name ['Unet','UnetPlusPlus','DeepLabV3','FPN','PAN','PSPNet','Linknet','PSPNet','DeepLabV3Plus']
"""
import torch.nn as nn
import segmentation_models_pytorch as smp

class SemanticSegBaseline(nn.Module):
    def __init__(self,config):
        super().__init__()
        
        self.config=config
        self.name=config.net_name
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset_name
        self.ignore_index = self.config.ignore_index
        
        if config.backbone_pretrained:
            weight='imagenet'
        else:
            weight=None
        
        in_channels=3
        if config.net_name=='PAN':
            model=smp.__dict__[config.net_name](encoder_name=config.backbone_name,
                                                encoder_weights=weight,
                                                classes=self.class_number,
                                                in_channels=in_channels)
        else:
            model=smp.__dict__[config.net_name](encoder_name=config.backbone_name,
                                                encoder_weights=weight,
                                                classes=self.class_number,
                                                in_channels=in_channels)
        
        self.net=model
        
    def forward(self,x):
        return self.net(x)
    
    