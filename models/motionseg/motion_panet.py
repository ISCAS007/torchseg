# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings
from models.motionseg.motion_backbone import (motion_backbone,
                                              motionnet_upsample_bilinear,
                                              transform_motionnet,
                                              conv_bn_relu
                                              )
from models.psp_vgg import make_layers
from models.motionseg.motion_fcn import stn
from easydict import EasyDict as edict

class panet(motion_backbone):
    """
    use for panet for image/flow and backbone for flow
    with network_mode=config.flow_backbone
    """
    def __init__(self,config,in_c=None,network_mode=None):
        if in_c is None:
            if config.net_name.find('flow')>=0:
                self.in_channels=2
            else:
                self.in_channels=3
        else:
            self.in_channels=in_c
        
        self.network_mode=network_mode
        super().__init__(config,config.use_none_layer)
        
        
    def get_layers(self):
        self.format='vgg'
        
        cfg_modes = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512,'N', 512, 512, 'N'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'N', 512, 512, 'N'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
        }
        if self.network_mode is None:
            cfg=[8,'M',16,'M',32,32,'M',64,64,'N',64,64,'N',64,64]
        else:
            cfg=cfg_modes[self.network_mode]
            
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

class transform_panet(transform_motionnet):
    def __init__(self,backbone,panet,config,flow_panet=None):
        super().__init__(backbone,config)
        if flow_panet is None:
            self.get_layers(backbone,panet)
        else:
            self.get_layers(backbone,panet,flow_panet)
        
    def get_layers(self,backbone,panet=None,flow_panet=None):
        """
        concat layer:
            main_input: main_c
            aux_input: aux_c
            main_panet_input: main_panet_c
            aux_panet_input: aux_panet_c
            previous_input: pre_c
            current_output: cur_c
            concat([main_c,aux_c,main_panet_c,aux_panet_c,pre_c])
            conv(main_c+aux_c+pre_c,cur_c)
        deconv layer:
            previsou_input: pre_c
            deconv_output: deconv_c
            assert deconv_c=pre_c
        """
        if panet is None:
            print('okay: panet is None in parent')
            return 0
        
        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.layers.append(None)
                self.concat_layers.append(None)
            elif idx==self.deconv_layer:
                out_c=main_c=backbone.get_feature_map_channel(idx)
                main_panet_c=panet.get_feature_map_channel(idx)
                if self.config.net_name.find('flow')>=0:
                    aux_c=flow_panet.get_feature_map_channel(idx)
                    aux_panet_c=0
                else:
                    aux_c=backbone.get_feature_map_channel(idx)
                    aux_panet_c=panet.get_feature_map_channel(idx)
                
                if self.merge_type=='concat':
                    merge_c=main_c+aux_c+main_panet_c+aux_panet_c if self.use_aux_input else main_c+main_panet_c
                    if self.use_aux_input:
                        self.concat_layers.append(conv_bn_relu(in_channels=merge_c,
                                                        out_channels=main_c,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        inplace=inplace))
                    else:
                        self.concat_layers.append(None)
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                    
                if self.use_none_layer and idx>3:
                    layer=nn.Sequential(conv_bn_relu(in_channels=main_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=nn.Sequential(nn.ConvTranspose2d(main_c,main_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=main_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.layers.append(layer)
                
            else:
                pre_c=backbone.get_feature_map_channel(idx+1)
                out_c=main_c=backbone.get_feature_map_channel(idx)
                main_panet_c=panet.get_feature_map_channel(idx)
                if self.config.net_name.find('flow')>=0:
                    aux_c=flow_panet.get_feature_map_channel(idx)
                    aux_panet_c=0
                else:
                    aux_c=backbone.get_feature_map_channel(idx)
                    aux_panet_c=panet.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                if self.merge_type=='concat':
                    merge_c=pre_c+main_c+main_panet_c+aux_c+aux_panet_c if self.use_aux_input else pre_c+main_c+main_panet_c
                    self.concat_layers.append(conv_bn_relu(in_channels=merge_c,
                                                    out_channels=main_c,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    inplace=inplace))
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                    
                if (self.use_none_layer and idx>3) or idx==0:
                    layer=nn.Sequential(conv_bn_relu(in_channels=main_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=nn.Sequential(nn.ConvTranspose2d(main_c,main_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=main_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.layers.append(layer)
        self.model_layers=nn.ModuleList([layer for layer in self.layers if layer is not None])
        if self.merge_type=='concat':
            self.merge_layers=nn.ModuleList([layer for layer in self.concat_layers if layer is not None])
        else:
            assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
            
    def forward(self,backbone_features,panet_features):
        feature=None
        for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
            if isinstance(backbone_features[0],(tuple,list)):
                main=backbone_features[0][idx]
                aux=backbone_features[1][idx]
            else:
                main=backbone_features[idx]
                aux=None
            
            if isinstance(panet_features[0],(tuple,list)):
                main_panet=panet_features[0][idx]
                aux_panet=panet_features[1][idx]
            else:
                main_panet=panet_features[idx]
                aux_panet=None
                
            f_list=[f for f in [feature,main,main_panet,aux,aux_panet] if f is not None]
            feature=torch.cat(f_list,dim=1)
            feature=self.concat_layers[idx](feature)
            feature=self.layers[idx](feature)
        return feature
        
class motion_panet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.use_aux_input=config.use_aux_input
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        self.panet=panet(config,in_c=3)
        if config.net_name.find('flow')>=0:
            self.use_flow=True
            self.flow_panet=panet(config,in_c=2)
            self.midnet=transform_panet(self.backbone,self.panet,config,self.flow_panet)
        else:
            self.use_flow=False
            self.midnet=transform_panet(self.backbone,self.panet,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        self.config=config
        
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        
    def forward(self,imgs):
        if self.use_aux_input or self.use_flow:
            if self.use_flow:
                backbone_features=self.backbone.forward_layers(imgs[0])
                panet_features=[self.panet.forward_layers(imgs[0]),
                                self.flow_panet.forward_layers(imgs[1])]
                feature_transform=self.midnet.forward(backbone_features,panet_features)
            else:
                backbone_features=[self.backbone.forward_layers(img) for img in imgs]
                panet_features=[self.panet.forward_layers(img) for img in imgs]
                feature_transform=self.midnet.forward(backbone_features,panet_features)
        else: 
            backbone_features=self.backbone.forward_layers(imgs[0])
            panet_features=self.panet.forward_layers(imgs[0])
            feature_transform=self.midnet.forward(backbone_features,panet_features)
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}

class motion_panet_flow(motion_panet):
    def __init__(self,config):
        super().__init__(config)

class transform_panet2(nn.Module):
    def __init__(self,backbones,config):
        super().__init__()
        self.deconv_layer=config.deconv_layer
        self.upsample_layer=config.upsample_layer
        self.fusion_type=config.fusion_type
        self.use_none_layer=config.use_none_layer
        self.decode_main_layer=config.decode_main_layer
        self.min_channel_number=config.min_channel_number
        self.backbones=backbones
        
        self.layers=[]
        self.keys=['main_backbone','aux_backbone','main_panet','aux_panet']
        channels={}
        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.layers.append(None)
                continue
            elif idx==self.deconv_layer:
                merge_c=0
            else:
                merge_c=max(backbones['main_backbone'].get_feature_map_channel(idx+1),
                            self.min_channel_number)
            
            out_c=max(backbones['main_backbone'].get_feature_map_channel(idx),
                      self.min_channel_number)
            
            for key,value in backbones.items():
                assert key in self.keys
                channels[key]=value.get_feature_map_channel(idx)
                if self.fusion_type=='all' or key.find('main')>=0:
                    merge_c+=channels[key]
                elif self.fusion_type in ['first','HR'] and idx==self.upsample_layer:
                    merge_c+=channels[key]
                elif self.fusion_type in ['last','LR'] and idx==self.deconv_layer:
                    merge_c+=channels[key]
                
            current_layer=[conv_bn_relu(in_channels=merge_c,
                                                 out_channels=out_c,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1,
                                                 inplace=inplace)]
                    
            if idx==0 or (self.use_none_layer and idx>3):
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

            f_list=[f for f in f_list if f is not None]
            feature=torch.cat(f_list,dim=1)
            feature=self.layers[idx](feature)
        return feature
        
class motion_panet2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.input_shape=config.input_shape
        self.upsample_layer=config.upsample_layer
        self.use_aux_input=config.use_aux_input
        self.min_channel_number=config.min_channel_number
        if config.net_name.find('flow')>=0:
            self.use_flow=True
            self.share_backbone=False
            if config.share_backbone:
                warnings.warn('share backbone not worked for {}'.format(config.net_name))
        else:
            self.use_flow=False
            self.share_backbone=config.share_backbone
        
        self.main_backbone=motion_backbone(config,use_none_layer=config.use_none_layer)
        
        self.aux_backbone=None
        if self.share_backbone:
            if self.use_aux_input:
                self.aux_backbone=self.main_backbone
        else:
            if self.use_flow:
                self.aux_backbone=panet(config,in_c=2,network_mode=config.flow_backbone)
            elif self.use_aux_input:
                self.aux_backbone=motion_backbone(config,use_none_layer=config.use_none_layer)
        
        self.main_panet=None
        if config.main_panet:
            self.main_panet=panet(config,in_c=3)
        
        self.aux_panet=None
        if config.aux_panet:
            in_c=2 if self.use_flow else 3
            if self.share_backbone and self.main_panet is not None:
                self.aux_panet=self.main_panet
            else:
                self.aux_panet=panet(config,in_c=in_c)
        
        
        self.get_midnet()
        self.midnet_out_channels=max(self.main_backbone.get_feature_map_channel(self.upsample_layer),
                                     self.min_channel_number)
        self.class_number=2
        
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
    
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
                
        self.midnet=transform_panet2(backbones,self.config)
        
    def forward(self,imgs):
        features={}
        features['main_backbone']=self.main_backbone.forward_layers(imgs[0])
        if self.use_aux_input or self.use_flow:
            features['aux_backbone']=self.aux_backbone.forward_layers(imgs[1])
        
        if self.config.main_panet:
            features['main_panet']=self.main_panet.forward_layers(imgs[0])
        
        if self.config.aux_panet:
            features['aux_panet']=self.aux_panet.forward_layers(imgs[1])
        
        x=self.midnet(features)
        y=self.decoder(x)
        return {'masks':[y]}
        
class motion_panet2_flow(motion_panet2):
    def __init__(self,config):
        super().__init__(config)

class motion_panet2_stn(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.stn=stn(config)
        self.motion_panet2=motion_panet2(config)
        
    def forward(self,imgs):
        results=self.stn(imgs)
        masks=self.motion_panet2(results['stn_images'])
        results['masks']=masks['masks']
        
        return results
    
if __name__ == '__main__':
    config=edict()
    config.backbone_name='resnet152'
    config.layer_preference='last'
    config.backbone_freeze=False
    config.freeze_layer=0
    config.freeze_ratio=0
    config.upsample_layer=3
    config.net_name='pspnet'
    config.modify_resnet_head=False
    config.use_none_layer=False
    config.deconv_layer=5
    
    for name in ['vgg21_bn']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=panet(config)
        bb.show_layers()