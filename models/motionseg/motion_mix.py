# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.motionseg.motion_backbone import (motion_backbone,
                                              transform_motionnet,
                                              )

from models.motionseg.motion_unet import get_decoder

class motion_mix(nn.Module):
    """
    concat inputs early (fusion early)
    """
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        if self.config.net_name== 'motion_mix':
            self.in_channels=6
        elif self.config.net_name=='motion_mix_flow':
            self.in_channels=5
        else:
            assert False
            
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'],in_channels=self.in_channels)
        
        config.use_aux_input=False
        self.midnet=transform_motionnet(self.backbone,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer,)
        self.class_number=2
        self.decoder=get_decoder(self)
        
    def forward(self,imgs):
        f=torch.cat(imgs,dim=1)
        feature_transform=self.midnet.forward(self.backbone.forward_layers(f))
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}
    
class motion_mix_flow(motion_mix):
    def __init__(self,config):
        super().__init__(config)