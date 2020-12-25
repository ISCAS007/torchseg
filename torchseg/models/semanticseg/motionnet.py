# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
from ..backbone import backbone
from ..upsample import transform_segnet,get_suffix_net,conv_bn_relu
from ..custom_layers import AttentionLayer
from ..psp_vgg import make_layers
from easydict import EasyDict as edict

class motionnet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__

        print('warnniing: use_none_layer is false for motionnet')
        self.backbone=backbone(config,use_none_layer=config.use_none_layer)

        self.upsample_type = self.config.upsample_type
        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name=self.config.dataset_name
        self.ignore_index=self.config.ignore_index

        self.midnet=transform_segnet(self.backbone,self.config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)

        self.attention=AttentionLayer(config,self.midnet_out_channels)
        self.decoder=get_suffix_net(config,
                                    self.midnet_out_channels,
                                    self.class_number)

    def forward(self, x):
        feature_map=self.backbone.forward_layers(x)
        feature_transform=self.midnet.forward(feature_map)
        feature_transform=self.attention(feature_transform)
        y=self.decoder(feature_transform)
        return y

class panet(backbone):
    def __init__(self,config):
        self.in_channels=3
        super().__init__(config,config.use_none_layer)


    def get_layers(self):
        self.format='vgg'
        cfg=[8,'M',16,'M',32,32,'M',64,64,'N',64,64,'N',64,64]
        self.features=make_layers(cfg,batch_norm=False,eps=self.eps,
                                  momentum=self.momentum,
                                  use_none_layer=self.use_none_layer,
                                  in_channels=self.in_channels)
        self.df=self.get_dataframe()
        self.layer_depths=self.get_layer_depths()

        self._initialize_weights()

    def freeeze_layers(self):
        pass

    def get_feature_map_channel(self,level):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        x=torch.rand(2,self.in_channels,224,224)
        x=Variable(x.to(device).float())
        x=self.forward(x,level)
        return x.shape[1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class transform_panet(nn.Module):
    def __init__(self,backbone,panet,config):
        super().__init__()
        self.config=config
        self.deconv_layer=config.deconv_layer
        self.upsample_layer=config.upsample_layer
        self.use_none_layer=config.use_none_layer
        self.layers=[]
        self.get_layers(backbone,panet)

    def get_layers(self,backbone,panet):
        """
        merge layers:
            stage_input: stage_c/backbone_c/main_c
            panet_input: panet_c
            upsample_input: upsample_c
            merge_output: merge_c
        """

        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.layers.append(None)
                continue
            elif idx==self.deconv_layer:
                upsample_c=0
            else:
                upsample_c=backbone.get_feature_map_channel(idx+1)

            backbone_c=backbone.get_feature_map_channel(idx)
            panet_c=panet.get_feature_map_channel(idx)
            merge_c=backbone_c+panet_c+upsample_c
            current_layers=[conv_bn_relu(in_channels=merge_c,
                                             out_channels=backbone_c,
                                             inplace=inplace)]

            if idx>0 and (not self.use_none_layer or idx<=3):
                current_layers+=[nn.ConvTranspose2d(backbone_c,backbone_c,kernel_size=4,stride=2,padding=1,bias=False)]
            current_layers+=[conv_bn_relu(in_channels=backbone_c,
                                                 out_channels=backbone_c,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 inplace=inplace)]
            self.layers.append(nn.Sequential(*current_layers))
        self.model_layers=nn.ModuleList([layer for layer in self.layers if layer is not None])
    def forward(self,backbone_features,panet_features):
        feature=None
        for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
            f_list=[f for f in [feature,backbone_features[idx],panet_features[idx]] if f is not None]
            feature=torch.cat(f_list,dim=1)
            feature=self.layers[idx](feature)
        return feature

class motion_panet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__

        self.upsample_layer=config.upsample_layer
        self.class_number=config.class_number
        self.backbone=backbone(config,use_none_layer=config.use_none_layer)
        self.panet=panet(config)
        self.midnet=transform_panet(self.backbone,self.panet,self.config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.decoder=get_suffix_net(config,
                                    self.midnet_out_channels,
                                    self.class_number)

    def forward(self,x):
        backbone_features=self.backbone.forward_layers(x)
        panet_features=self.panet.forward_layers(x)
        feature_transform=self.midnet.forward(backbone_features,panet_features)
        y=self.decoder(feature_transform)

        return y

if __name__ == '__main__':
    config=edict()
    config=edict()
    config.upsample_layer=2
    config.deconv_layer=5
    config.backbone_freeze=False
    config.freeze_layer=0
    config.freeze_ratio=0.0
    config.backbone_name='vgg11'
    config.layer_preference='last'
    config.net_name='motion_panet'
    config.use_none_layer=False

    for name in ['vgg11']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=panet(config)
        bb.show_layers()