# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from models.motionseg.motion_backbone import motion_backbone,motionnet_upsample_bilinear,conv_bn_relu
from models.motionseg.motion_unet import motion_unet
from models.Anet.layers import Anet
from easydict import EasyDict as edict

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
        
    def forward(self,x):
        x=self.midnet(x)
        b,c,h,w=x.shape
        x=x.permute(0,2,3,1).reshape(b,h,w,self.midnet_out_channels,-1)
        x=torch.sum(x,dim=-1).permute(0,3,1,2)
        
        return self.decoder(x)
    
class motion_anet(motion_unet):
    def __init__(self,config):
        assert config.backbone_name=='Anet'
        super().__init__(config)
        self.backbone=Anet(config)
        
        x=torch.rand(2,3,224,224)
        stage_f_list,mid_f_list=self.backbone(x,return_mid_feature=True)
        
        mid_decoders=[]
        if self.use_aux_input:
            indexs=[5-i for i in range(5)]
            for idx,mid_f in zip(indexs,mid_f_list):
                in_c=mid_f.size(1)
                mid_c=2*idx*10
                mid_decoders.append(mid_decoder(2*in_c,mid_c,config))
        else:
            indexs=[5-i for i in range(5)]
            for idx,mid_f in zip(mid_f_list,indexs):
                in_c=mid_f.size(1)
                mid_c=2*idx*10
                mid_decoders.append(mid_decoder(in_c,mid_c,config))
        
        self.mid_decoders=nn.ModuleList(mid_decoders)
                
    def forward(self,imgs):
        if self.use_aux_input:
            anet_features=[self.backbone(img,True) for img in imgs]
            mid_f_list=[torch.cat([x1,x2],dim=1) for x1,x2 in zip(anet_features[0][1],anet_features[1][1])]
            feature_transform=self.midnet.forward(anet_features[0][0],anet_features[1][0])
        else:
            stage_f_list,mid_f_list=self.backbone(imgs[0],return_mid_feature=True)
            feature_transform=self.midnet.forward(stage_f_list)
        y=self.decoder(feature_transform)
        
        masks=[y]
        for mid_f,decoder in zip(mid_f_list,self.mid_decoders):
            masks.append(decoder(mid_f))
            
        return {'masks':masks}