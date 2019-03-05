# -*- coding: utf-8 -*-
import torch.nn as nn
from models.motionseg.motion_backbone import motion_backbone,transform_motion_psp,motionnet_upsample_bilinear
from models.motionseg.motion_fcn import stn,dict2edict

class motion_psp(nn.Module):
    """
    for use_none_layer=False
    input shape = (240,240)
        240 = min_common_multiplier(midnet_pool_sizes)*midnet_scale*upsample_ratio
            = min_common_multiplier([1,2,3,6])*5*(2**upsample_layer)
    
    """
    def __init__(self,config):
        super().__init__()
        self.config=config
        
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        
        self.midnet_input_shape = self.backbone.get_output_shape(
            self.upsample_layer, self.input_shape)
        
        self.midnet_out_channels=4*self.midnet_input_shape[1]
        
        midnet_pool_sizes=[1,2,3,6]
        midnet_scale=config.psp_scale
        self.midnet=transform_motion_psp(midnet_pool_sizes,
                                         midnet_scale,
                                         self.midnet_input_shape,
                                         self.midnet_out_channels)
        self.class_number=2
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        
    def forward(self,imgs):
        features=[self.backbone.forward(img,self.upsample_layer) for img in imgs]
        main,aux=tuple(features)
        feature_transform=self.midnet.forward(main,aux)
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}