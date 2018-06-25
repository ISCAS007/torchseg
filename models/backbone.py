# -*- coding: utf-8 -*-
import pandas as pd
import torchvision as TV
import torch.nn as TN
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.models import *
from easydict import EasyDict as edict

class backbone(TN.Module):
    def __init__(self,config,model=None):
        super(backbone,self).__init__()
        self.config=config
        if hasattr(self.config,'pretrained'):
            self.pretrained=self.config.pretrained
        else:
            self.pretrained=False
        
        if model is None:
            self.model=self.get_model()
        else:
            self.model=model
        
        
        if self.config.backbone_name.find('vgg')>=0:
            self.format='vgg'
            self.df=self.get_dataframe()
            self.layer_depths=self.get_layer_depths()
        elif self.config.backbone_name.find('resnet')>=0:
            self.format='resnet'
            self.conv1 = TN.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
            self.bn1 = TN.BatchNorm2d(64)
            self.relu = TN.ReLU(inplace=True)
            self.maxpool = TN.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            assert False,'unknown backbone name %s'%self.config.backbone_name
        
    def forward(self,x,level):
        assert level in [1,2,3,4,5],'feature level %d not in range(0,5)'%level
        
        if self.format=='vgg':
            assert hasattr(self.layer_depths,str(level))
            for idx,layer in enumerate(self.model.features):
                x=layer(x)
                if idx == self.layer_depths[str(level)]:
                    return x
            
        elif self.format=='resnet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if level==1:
                return x
            
            x = self.maxpool(x)
            x = self.layer1(x)
            # layer 1 not change feature map height and width
            if level==2:
                return x
            x = self.layer2(x)
            if level==3:
                return x
            x = self.layer3(x)
            if level==4:
                return x
            x = self.layer4(x)
            if level==5:
                return x
            
        assert False,'unexpected level %d for format %s'%(level,self.format)
        
    def get_feature_map_channel(self,level):
        self.cuda()
        x=torch.rand(2,3,224,224)
        x=Variable(x.cuda().float())
        x=self.forward(x,level)
        return x.shape[1]
    
    def get_feature_map_size(self,level,input_size):
        self.cuda()
        x=torch.rand(2,3,input_size[0],input_size[1])
        x=Variable(x.cuda().float())
        x=self.forward(x,level)
        return x.shape[2:4]
        
    def get_model(self):
        assert self.config.backbone_name in globals().keys(), 'undefine backbone name %s'%self.config.backbone_name
        return globals()[self.config.backbone_name](pretrained=self.pretrained)
    
    def get_dataframe(self):
        assert self.format=='vgg','only vgg models have features'
        df=pd.DataFrame(columns=['level','layer_depth','layer_name'])
        level=0
        for idx,layer in enumerate(self.model.features):
            name=layer.__class__.__name__
#            if name in ['ZeroPadding2D','Dropout','Reshape'] or name.find('Pool')>=0:
#                continue
            if name in ['Conv2d'] or name.find('Pool2d')>=0:
                if layer.stride==2:
                    level=level+1

            df=df.append({'level':level,
                       'layer_depth':idx,
                       'layer_name':name},ignore_index=True)
    
        return df

    def get_layer_depths(self):
        assert self.format=='vgg','only vgg models have features'
        df=self.df
        levels=np.unique(df['level'].tolist())
        layer_depths=edict()
        for level in levels:
            if self.config.layer_preference=='first':
                d=df[df.level==level].head(n=1)
                depth=d.layer_depth.tolist()[0]
            elif self.config.layer_preference=='last':
                d=df[df.level==level].tail(n=1)
                depth=d.layer_depth.tolist()[0]
            elif self.config.layer_preference in ['random','rand']:    
                d=df[df.level==level]
                depth=np.random.choice(d.layer_depth.tolist())
            else:
                print('undefined layer preference',self.config.layer_preference)
                assert False
            
            layer_depths[str(level)]=depth
        
        return layer_depths
    
    def get_layer_outputs(self,x):
        assert self.format=='vgg','only vgg models have features'
        layer_outputs=[]
        for idx,layer in enumerate(self.model.features):
            x=layer(x)
            if idx in self.layer_depths.values():
                layer_outputs.append(x)
        
        return layer_outputs        
    
    def show_layers(self):
        if self.format=='vgg':
            for idx,layer in enumerate(self.model.features):
                if idx in self.layer_depths.values():
                    print(idx,layer)
        else:
            print('layer 1 '+'*'*50)
            print(self.model.layer1)
            print('layer 2 '+'*'*50)
            print(self.model.layer2)
            print('layer 3 '+'*'*50)
            print(self.model.layer3)
            print('layer 4 '+'*'*50)
            print(self.model.layer4)
    
if __name__ == '__main__':
    config=edict()
    config.backbone_name='resnet152'
    config.layer_preference='last'
    
    for name in ['vgg16','vgg19','vgg16_bn','vgg19_bn','resnet18','resnet34','resnet50','resnet101','resnet152']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=backbone(config)
        bb.show_layers()
        break