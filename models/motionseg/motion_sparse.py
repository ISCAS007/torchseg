# -*- coding: utf-8 -*-

import torch.nn as nn
from models.motionseg.motion_backbone import motion_backbone,transform_sparse,motionnet_upsample_bilinear
from models.motionseg.motion_fcn import stn,dict2edict

class motion_sparse(nn.Module):
    """
    like densenet + pspnet
    import parameters:
        upsample_layer
        deconv_layer
        sparse_ratio
    """
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        
        self.midnet=transform_sparse(self.backbone,config)
        self.class_number=2
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet.concat_channels,
                                                         out_channels=self.class_number,
                                                         output_shape=self.config.input_shape[0:2])
    def forward(self,imgs):
        features=[self.backbone.forward_layers(img) for img in imgs]
        main,aux=tuple(features)
        feature_transform=self.midnet.forward(main,aux)
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}