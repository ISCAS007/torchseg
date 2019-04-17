# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.motionseg.motion_backbone import (motion_backbone,
                                              transform_motionnet,
                                              motionnet_upsample_bilinear,
                                              transform_motionnet_flow,
                                              conv_bn_relu
                                              )

from models.motionseg.motion_fcn import stn

class upsample_subclass(nn.Module):
    def __init__(self,in_channels, out_channels, output_shape,use_sigmoid):
        """
        in_channels
        out_channels
        sub_class_number=in_channels//out_channels
        mid_channels=out_channels*sub_class_number
        """
        super().__init__()
        self.output_shape=output_shape
        self.class_number=out_channels
        self.sub_class_number=in_channels//out_channels
        assert self.sub_class_number>1
        self.midnet=conv_bn_relu(in_channels,self.class_number*self.sub_class_number)
        
        self.use_sigmoid=use_sigmoid
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        x=self.midnet(x)
        b,c,h,w=x.shape
        x=x.permute(0,2,3,1).reshape(b,h,w,self.class_number,-1)
        if self.use_sigmoid:
            x=self.sigmoid(x)
        x=torch.sum(x,dim=-1).permute(0,3,1,2)
        x = F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)
        return x
    
class upsample_smooth(nn.Module):
    """
    smooth the decrease of channel nummber
    for example: in_c=512, out_c=2
    smooth version: 512->64->8->2
    normal version: 512->2
    """
    def __init__(self,in_c,out_c,output_shape,smooth_ratio=8):
        super().__init__()    
        self.output_shape=output_shape
        self.class_number=out_c
        self.smooth_ratio=smooth_ratio
        
        conv_layers=[]
        while in_c//smooth_ratio>self.class_number:
            conv_layers.append(conv_bn_relu(in_c,in_c//smooth_ratio))
            in_c=in_c//smooth_ratio
            
        conv_layers.append(in_c,self.class_number)
        self.midnet=nn.Sequential(*conv_layers)
        
    def forward(self,x):
        x=self.midnet(x)
        x=F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)
        return x
    
class mid_decoder(nn.Module):
    def __init__(self,in_c,mid_c,config,ratio=10):
        """
        in_c --> mid_c --> class_number
        in_c --> mid_c*ratio --> mid_c --> class_number
        """
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.class_number=2
        self.midnet=conv_bn_relu(in_c,mid_c*ratio)
        self.midnet_out_channels=mid_c
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        self.sigmoid=nn.Sigmoid()
        self.use_sigmoid=config.subclass_sigmoid
        
    def forward(self,x):
        x=self.midnet(x)
        b,c,h,w=x.shape
        x=x.permute(0,2,3,1).reshape(b,h,w,self.midnet_out_channels,-1)
        if self.use_sigmoid:
            x=self.sigmoid(x)
        x=torch.sum(x,dim=-1).permute(0,3,1,2)
        
        return self.decoder(x)
    
def get_decoder(self):
    if self.config.upsample_type=='bilinear':
        decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
    elif self.config.upsample_type=='subclass':
        decoder=upsample_subclass(in_channels=self.midnet_out_channels,
                                         out_channels=self.class_number,
                                         output_shape=self.input_shape[0:2],
                                         use_sigmoid=self.config.subclass_sigmoid)
    elif self.config.upsample_type=='mid_decoder':
        decoder=mid_decoder(in_c=self.midnet_out_channels,
                                 mid_c=self.class_number*8,
                                 config=self.config,
                                 ratio=8)
    elif self.config.upsample_type=='smooth':
        decoder=upsample_smooth(in_c=self.midnet_out_channels,
                                out_c=self.class_number,
                                output_shape=self.input_shape[0:2],
                                smooth_ratio=self.config.smooth_ratio)
    else:
        assert False,'unknonw upsample type {}'.format(self.config.upsample_type)
        
    return decoder
            
class motion_unet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.use_aux_input=config.use_aux_input
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        
        self.midnet=transform_motionnet(self.backbone,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        self.decoder=get_decoder(self)
        
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