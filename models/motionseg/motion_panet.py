# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from models.motionseg.motion_backbone import motion_backbone,motionnet_upsample_bilinear,transform_motionnet,conv_bn_relu
from models.psp_vgg import make_layers
from easydict import EasyDict as edict

class panet(motion_backbone):
    def __init__(self,config):
        if config.net_name.find('flow')>=0:
            self.in_channels=2
        else:
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

class transform_panet(transform_motionnet):
    def __init__(self,backbone,panet,config):
        super().__init__(backbone,config)
        self.get_layers(backbone,panet)
        
    def get_layers(self,backbone,panet=None):
        """
        concat layer:
            main_input: main_c
            aux_input: aux_c
            previous_input: pre_c
            current_output: cur_c
            concat([main_c,aux_c,pre_c])
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
                aux_c=panet.get_feature_map_channel(idx)
                
                if self.merge_type=='concat':
                    merge_c=main_c+aux_c if self.use_aux_input else main_c
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
                aux_c=panet.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                if self.merge_type=='concat':
                    merge_c=pre_c+main_c+aux_c if self.use_aux_input else pre_c+main_c
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
        
class motion_panet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.input_shape=config.input_shape
        self.upsample_layer=config['upsample_layer']
        self.use_aux_input=config.use_aux_input
        self.backbone=motion_backbone(config,use_none_layer=config['use_none_layer'])
        self.panet=panet(config)
        
        self.midnet=transform_panet(self.backbone,self.panet,config)
        self.midnet_out_channels=self.backbone.get_feature_map_channel(self.upsample_layer)
        self.class_number=2
        
        self.decoder=motionnet_upsample_bilinear(in_channels=self.midnet_out_channels,
                                                     out_channels=self.class_number,
                                                     output_shape=self.input_shape[0:2])
        
    def forward(self,imgs):
        if self.use_aux_input:
            main=self.backbone.forward_layers(imgs[0])
            aux=self.panet.forward_layers(imgs[1])
            feature_transform=self.midnet.forward(main,aux)
        else:
            feature_transform=self.midnet.forward(self.backbone.forward_layers(imgs[0]))
        y=self.decoder(feature_transform)
        
        return {'masks':[y]}

class motion_panet_flow(motion_panet):
    def __init__(self,config):
        super().__init__(config)
    
if __name__ == '__main__':
    config=edict()
    config.backbone_name='resnet152'
    config.layer_preference='first'
    config.backbone_freeze=False
    config.freeze_layer=0
    config.freeze_ratio=0
    config.upsample_layer=3
    config.net_name='pspnet'
    config.modify_resnet_head=False
    config.use_none_layer=True
    config.deconv_layer=5
    
    for name in ['zzz']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=panet(config)
        bb.show_layers()