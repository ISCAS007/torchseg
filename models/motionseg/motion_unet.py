# -*- coding: utf-8 -*-

import torch.nn as nn
from models.motionseg.motion_backbone import motion_backbone,transform_motionnet
from models.upsample import get_suffix_net, transform_segnet
from models.motionseg.motion_fcn import stn,dict2edict

class motion_unet(nn.Module):
    def __init__(self,config):
        super().__init__()
        decoder_config=dict2edict(config)
        decoder_config.model.merge_type='concat'
        self.input_shape=decoder_config.model.input_shape
        self.upsample_layer=config['upsample_layer']
        self.backbone=motion_backbone(decoder_config.model,use_none_layer=config['use_none_layer'])
        
        self.midnet=transform_motionnet(self.backbone,decoder_config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        self.decoder=get_suffix_net(decoder_config,
                                    self.midnet_out_channels,
                                    self.class_number)
        
    def forward(self,imgs):
        features=[self.backbone.forward_layers(img) for img in imgs]
        main,aux=tuple(features)
        feature_transform=self.midnet.forward(main,aux)
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