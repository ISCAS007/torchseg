# -*- coding: utf-8 -*-
"""
use changenet structure to prediction foreground area
- ChangeNet: A deep learning architecture for visual change detection 2019

```
python test/fbms_train.py --net_name motion_changenet --upsample_layer 3 --deconv_layer 5 --input_format n
```
"""

import torch
import torch.nn as nn

from .motionseg_base_class import motion
from .motion_backbone import conv_bn_relu
from .motion_unet import get_decoder

class transform_sum(nn.Module):
    def __init__(self,backbones,config):
        super().__init__()
        self.config=config
        self.out_channel=11
        self.upsample_layer=self.config.upsample_layer
        self.deconv_layer=self.config.deconv_layer
        self.min_channel_number=self.config.min_channel_number
        self.max_channel_number=self.config.max_channel_number

        self.main_layers=[]
        self.aux_layers=[]
        self.merge_layers=[]
        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.main_layers.append(None)
                self.aux_layers.append(None)
                self.merge_layers.append(None)
                continue

#            main_c=min(max(backbones['main_backbone'].get_feature_map_channel(idx),
#                            self.min_channel_number),self.max_channel_number)
#            aux_c=min(max(backbones['aux_backbone'].get_feature_map_channel(idx),
#                            self.min_channel_number),self.max_channel_number)
            main_c=backbones['main_backbone'].get_feature_map_channel(idx)
            aux_c=backbones['aux_backbone'].get_feature_map_channel(idx)

            out_c=self.out_channel

#            print(idx,main_c,aux_c,out_c)

            ratio=2**(idx-1)
            main_current_layer=[conv_bn_relu(in_channels=main_c,
                                         out_channels=out_c,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         inplace=inplace)]
            main_current_layer.append(nn.ConvTranspose2d(out_c,out_c,kernel_size=4*ratio,stride=2*ratio,padding=1*ratio,bias=False))

            aux_current_layer=[conv_bn_relu(in_channels=aux_c,
                                         out_channels=out_c,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         inplace=inplace)]
            aux_current_layer.append(nn.ConvTranspose2d(out_c,out_c,kernel_size=4*ratio,stride=2*ratio,padding=1*ratio,bias=False))

            merge_current_layer=[conv_bn_relu(in_channels=2*out_c,
                                         out_channels=out_c,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         inplace=inplace)]
            self.main_layers.append(nn.Sequential(*main_current_layer))
            self.aux_layers.append(nn.Sequential(*aux_current_layer))
            self.merge_layers.append(nn.Sequential(*merge_current_layer))

        self.main_models=nn.ModuleList([layer for layer in self.main_layers if layer is not None])
        self.aux_models=nn.ModuleList([layer for layer in self.aux_layers if layer is not None])
        self.merge_models=nn.ModuleList(layer for layer in self.merge_layers if layer is not None)

    def forward(self,features):
        assert isinstance(features,dict)

        fcn_features=[]
        for idx in range(self.upsample_layer,self.deconv_layer+1):
#            print(idx,features['main_backbone'][idx].shape,features['aux_backbone'][idx].shape)
            main_f=self.main_layers[idx](features['main_backbone'][idx])
            aux_f=self.aux_layers[idx](features['aux_backbone'][idx])
            merge_f=torch.cat([main_f,aux_f],dim=1)

            fcn_features.append(self.merge_layers[idx](merge_f))

        sum_f=sum(fcn_features)
        return sum_f

    def get_out_channels(self):
        return self.out_channel

class motion_changenet(motion):
    def __init__(self,config):
        super().__init__(config)
        backbones={"main_backbone":self.main_backbone,
                   "aux_backbone":self.aux_backbone}

        self.midnet=transform_sum(backbones,self.config)
        self.midnet_out_channels=self.midnet.get_out_channels()
        self.decoder=get_decoder(self)

    def forward(self,imgs):
        features={}
        features['main_backbone']=self.main_backbone.forward_layers(imgs[0])
        features['aux_backbone']=self.aux_backbone.forward_layers(imgs[1])

        x=self.midnet(features)
        y=self.decoder(x)
        return {'masks':[y]}

