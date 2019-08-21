# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import warnings
from models.motionseg.motion_backbone import conv_bn_relu

from models.motionseg.motion_fcn import stn
from models.motionseg.motion_panet import motion_panet2
from easydict import EasyDict as edict

class transform_filter(nn.Module):
    def __init__(self,backbones,config):
        super().__init__()
        self.deconv_layer=config.deconv_layer
        self.upsample_layer=config.upsample_layer
        self.fusion_type=config.fusion_type
        self.use_none_layer=config.use_none_layer
        self.decode_main_layer=config.decode_main_layer
        self.min_channel_number=config.min_channel_number
        self.max_channel_number=config.max_channel_number
        self.backbones=backbones
        self.filter_type=config.filter_type
        self.filter_feature=config.filter_feature
        self.filter_relu=config.filter_relu

        if config.net_name.find('flow')>=0:
            self.use_flow=True
        else:
            self.use_flow=False

        if self.filter_feature is None:
            self.filter_feature='aux' if self.use_flow else 'all'

        self.layers=[]
        self.filter_layers=[]
        self.keys=['main_backbone','aux_backbone','main_panet','aux_panet']
        channels={}
        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.layers.append(None)
                self.filter_layers.append(None)
                continue
            elif idx==self.deconv_layer:
                init_c=merge_c=0
            else:
                init_c=merge_c=min(max(backbones['main_backbone'].get_feature_map_channel(idx+1),
                            self.min_channel_number),self.max_channel_number)

            out_c=min(max(backbones['main_backbone'].get_feature_map_channel(idx),
                      self.min_channel_number),self.max_channel_number)

            for key,value in backbones.items():
                assert key in self.keys
                channels[key]=value.get_feature_map_channel(idx)

                if self.fusion_type=='all' or key.find('main')>=0:
                    merge_c+=channels[key]
                elif self.fusion_type in ['first','HR'] and idx==self.upsample_layer:
                    merge_c+=channels[key]
                elif self.fusion_type in ['last','LR'] and idx==self.deconv_layer:
                    merge_c+=channels[key]

            # add filter conv
            if (self.fusion_type in ['first','HR'] and idx==self.upsample_layer) or \
                (self.fusion_type in ['last','LR'] and idx==self.deconv_layer) or \
                self.fusion_type=='all':
                if self.filter_feature=='aux':
                    filter_c=sum([value for key,value in channels.items() if key.find('aux')>=0])
                else:
                    filter_c=merge_c

                self.filter_layers.append(nn.Sequential(
                        conv_bn_relu(in_channels=filter_c,
                                     out_channels=32,
                                     inplace=inplace),
                        conv_bn_relu(in_channels=32,
                                     out_channels=1,
                                     inplace=inplace,
                                     use_relu=self.filter_relu,
                                     use_bn=self.filter_relu,
                                     bias=not self.filter_relu),
                        nn.Sigmoid()
                        ))

                if self.filter_type=='main':
                    merge_c=sum([value for key,value in channels.items() if key.find('main')>=0])+init_c
            else:
                self.filter_layers.append(None)

            current_layer=[conv_bn_relu(in_channels=merge_c,
                                                 out_channels=out_c,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 inplace=inplace)]

            if idx==0 or (self.use_none_layer and idx>3) or idx==self.upsample_layer:
                pass
            else:
                current_layer.append(nn.ConvTranspose2d(out_c,out_c,kernel_size=4,stride=2,padding=1,bias=False))

            for _ in range(self.decode_main_layer):
                current_layer.append(conv_bn_relu(in_channels=out_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
            self.layers.append(nn.Sequential(*current_layer))

        self.model_layers=nn.ModuleList([layer for layer in self.layers if layer is not None])
        self.model_filter_layers=nn.ModuleList([layer for layer in self.filter_layers if layer is not None])

    def forward(self,features):
        assert isinstance(features,dict)
        feature=None
        for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
            f_list=[feature]
            for key,value in features.items():
                assert key in self.keys
                if self.fusion_type=='all' or key.find('main')>=0:
                    f_list.append(features[key][idx])
                elif self.fusion_type in ['first','HR'] and idx==self.upsample_layer:
                    f_list.append(features[key][idx])
                elif self.fusion_type in ['last','LR'] and idx==self.deconv_layer:
                    f_list.append(features[key][idx])

            if (self.fusion_type in ['first','HR'] and idx==self.upsample_layer) or \
                (self.fusion_type in ['last','LR'] and idx==self.deconv_layer) or \
                self.fusion_type=='all':

                if self.filter_type=='main':
                    main_f_list=[value[idx] for key,value in features.items() if key.find('main')>=0]+[feature]
                else:
                    main_f_list=f_list

                main_f_list=[f for f in main_f_list if f is not None]
                main_feature=torch.cat(main_f_list,dim=1)
                if self.filter_feature=='aux':
                    aux_f_list=[value[idx] for key,value in features.items() if key.find('aux')>=0]
                    aux_feature=torch.cat(aux_f_list,dim=1)
                    self.attention=self.filter_layers[idx](aux_feature)
                    feature=main_feature * self.attention
                else:
                    f_list=[f for f in f_list if f is not None]
                    feature=torch.cat(f_list,dim=1)

                    self.attention=self.filter_layers[idx](feature)
                    feature=main_feature * self.attention
            else:
                f_list=[f for f in f_list if f is not None]
                feature=torch.cat(f_list,dim=1)

            feature=self.layers[idx](feature)
        return feature

class motion_filter(motion_panet2):
    def __init__(self,config):
        super().__init__(config)

    def get_midnet(self):
        keys=['main_backbone','aux_backbone','main_panet','aux_panet']
        none_backbones={'main_backbone':self.main_backbone,
                   'aux_backbone':self.aux_backbone,
                   'main_panet':self.main_panet,
                   'aux_panet':self.aux_panet}
        backbones={}
        for key in keys:
            if none_backbones[key] is not None:
                backbones[key]=none_backbones[key]

        self.midnet=transform_filter(backbones,self.config)

class motion_filter_flow(motion_filter):
    def __init__(self,config):
        super().__init__(config)