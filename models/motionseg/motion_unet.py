# -*- coding: utf-8 -*-

import torch.nn as nn
from models.motionseg.motion_backbone import motion_backbone,transform_motionnet,motionnet_upsample_bilinear,transform_motionnet_flow
from models.motionseg.motion_fcn import stn,dict2edict

class motion_unet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.use_aux_input=config.use_aux_input
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        
        self.midnet=transform_motionnet(self.backbone,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        
    def forward(self,imgs):
        if self.use_aux_input:
            features=[self.backbone.forward_layers(img) for img in imgs]
            main,aux=tuple(features)
            feature_transform=self.midnet.forward(main,aux)
        else:
            feature_transform=self.midnet.forward(self.backbone.forward_layers(imgs[0]))
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}
    
class motion_unet_stn(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.stn=stn(config)
        self.motion_unet=motion_unet(config)
        
    def forward(self,imgs):
        results=self.stn(imgs)
        masks=self.motion_unet(results['stn_images'])
        results['masks']=masks['masks']
        
        return results
    
class motion_unet_flow(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        
        self.midnet=transform_motionnet_flow(self.backbone,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        
    def forward(self,imgs):
        main=self.backbone.forward_layers(imgs[0])
        flow=imgs[1]
        feature_transform=self.midnet.forward(main,flow)
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}