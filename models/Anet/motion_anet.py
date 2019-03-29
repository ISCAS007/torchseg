# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from models.motionseg.motion_backbone import motion_backbone,motionnet_upsample_bilinear,conv_bn_relu
from models.motionseg.motion_unet import motion_unet,mid_decoder
from models.Anet.layers import Anet
from easydict import EasyDict as edict
    
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