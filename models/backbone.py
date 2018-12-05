# -*- coding: utf-8 -*-
import pandas as pd
import torchvision as TV
import torch.nn as TN
import numpy as np
import torch
from torch.autograd import Variable
from easydict import EasyDict as edict
import warnings

class backbone(TN.Module):
    def __init__(self,config,use_none_layer=False):
        super().__init__()
        self.config=config
        if hasattr(self.config,'eps'):
            self.eps=self.config.eps
        else:
            self.eps=1e-5
        
        if hasattr(self.config,'momentum'):
            self.momentum=self.config.momentum
        else:
            self.momentum=0.1
        
        if use_none_layer == False:
            self.use_none_layer=False
            model=self.get_model()
            if self.config.backbone_name.find('vgg')>=0:
                self.format='vgg'
                self.features=model.features
                self.df=self.get_dataframe()
                self.layer_depths=self.get_layer_depths()
            elif self.config.backbone_name.find('resnet')>=0:
                self.format='resnet'
                self.prefix_net = TN.Sequential(model.conv1,
                                                model.bn1,
                                                model.relu,
                                                model.maxpool)
                
                self.layer1=model.layer1
                self.layer2=model.layer2
                if config.upsample_layer>=4:
                    self.layer3=model.layer3
                if config.upsample_layer>=5:
                    self.layer4=model.layer4
            else:
                assert False,'unknown backbone name %s'%self.config.backbone_name
        else:
            self.use_none_layer=True
            model=self.get_model()
            if self.config.backbone_name.find('vgg')>=0:
                self.format='vgg'
                self.features=model.features
                self.df=self.get_dataframe()
                self.layer_depths=self.get_layer_depths()
            elif self.config.backbone_name.find('resnet')>=0:
                # the output size of resnet layer is different from stand model!
                # raise NotImplementedError
                self.format='resnet'
                self.prefix_net = model.prefix_net
                self.layer1=model.layer1
                self.layer2=model.layer2
                if config.upsample_layer>=4:
                    self.layer3=model.layer3
                if config.upsample_layer>=5:
                    self.layer4=model.layer4
            else:
                assert False,'unknown backbone name %s'%self.config.backbone_name
                
        if self.config.backbone_freeze:
            for param in self.parameters():
                param.requrires_grad=False
        
        if self.config.freeze_layer > 0:
            if self.config.backbone_freeze:
                warnings.warn("it's not good to use freeze layer with backbone_freeze")

            freeze_layer=self.config.freeze_layer
            if self.format=='vgg':
                for idx,layer in enumerate(self.features):
                    if idx <= self.layer_depths[str(freeze_layer)]:                        
                        for param in layer.parameters():
                            param.requires_grad = False
            else:
                if freeze_layer>0:
                    for param in self.prefix_net.parameters():
                        param.requires_grad = False
                if freeze_layer>1:
                    for param in self.layer1.parameters():
                        param.requires_grad = False
                if freeze_layer>2:
                    for param in self.layer2.parameters():
                        param.requires_grad = False
                if freeze_layer>3:
                    for param in self.layer3.parameters():
                        param.requires_grad = False
                if freeze_layer>4:
                    for param in self.layer4.parameters():
                        param.requires_grad = False
        
        if self.config.freeze_ratio > 0.0:
            if self.config.backbone_freeze or self.config.freeze_layer:
                warnings.warn("it's not good to use freeze ratio with freeze layer or backbone")
            
            if self.format=='vgg':
                freeze_index=len(self.features)*self.config.freeze_ratio    
                for idx,layer in enumerate(self.features):
                    if idx < freeze_index:                        
                        for param in layer.parameters():
                            param.requires_grad = False
            else:
                valid_layer_number=0
                for name,param in self.named_parameters():
                    valid_layer_number+=1 

                freeze_index=valid_layer_number*self.config.freeze_ratio
                
                for idx,(name,param) in enumerate(self.named_parameters()):
                    if idx < freeze_index:
                        print('freeze weight of',name)
                        param.requires_grad=False
        
        # if modify resnet head worked, train the modified resnet head
        if config.modify_resnet_head and self.config.use_none_layer and self.format=='resnet':
            for param in self.prefix_net.parameters():
                param.requires_grad = True
    
    def forward_layers(self,x):
        features=[]
        if self.format=='vgg':
            layer_num=0
            assert hasattr(self.layer_depths,str(layer_num))
            for idx,layer in enumerate(self.features):
                x=layer(x)
                if idx == self.layer_depths[str(layer_num)]:
                    features.append(x)
                    layer_num+=1
                
        elif self.format=='resnet':
            features.append(x)
            x=self.prefix_net(x)
            features.append(x)
            x = self.layer1(x)
            features.append(x)
            x = self.layer2(x)
            features.append(x)
            x = self.layer3(x)
            features.append(x)
            x = self.layer4(x)
            features.append(x)
        else:
            assert False,'unexpected format %s'%(self.format)
        
        return features
    
    def forward_aux(self,x,main_level,aux_level):
        assert main_level in [1,2,3,4,5],'main feature level %d not in range(0,5)'%main_level
        assert aux_level in [1,2,3,4,5],'aux feature level %d not in range(0,5)'%aux_level
        
        features=[]
        if self.format=='vgg':
            layer_num=0
            assert hasattr(self.layer_depths,str(layer_num))
            for idx,layer in enumerate(self.features):
                x=layer(x)
                if idx == self.layer_depths[str(layer_num)]:
                    features.append(x)
                    layer_num+=1
        elif self.format=='resnet':
            features.append(x)
            x=self.prefix_net(x)
            features.append(x)
            x = self.layer1(x)
            features.append(x)
            x = self.layer2(x)
            features.append(x)
            x = self.layer3(x)
            features.append(x)
            x = self.layer4(x)
            features.append(x)
        else:
            assert False,'unexpected format %s'%(self.format)
        
        return features[main_level],features[aux_level]
        
    def forward(self,x,level):
        assert level in [1,2,3,4,5],'feature level %d not in range(0,5)'%level
        
        if self.format=='vgg':
            assert hasattr(self.layer_depths,str(level))
            for idx,layer in enumerate(self.features):
                x=layer(x)
                if idx == self.layer_depths[str(level)]:
                    return x
            
        elif self.format=='resnet':
            x=self.prefix_net(x)
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
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        x=torch.rand(2,3,224,224)
        x=Variable(x.to(device).float())
        x=self.forward(x,level)
        return x.shape[1]
    
    def get_feature_map_size(self,level,input_size):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        x=torch.rand(2,3,input_size[0],input_size[1])
        x=Variable(x.to(device).float())
        x=self.forward(x,level)
        return x.shape[2:4]
    
    def get_output_shape(self,level,input_size):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        x=torch.rand(2,3,input_size[0],input_size[1])
        x=torch.autograd.Variable(x.to(device).float())
        x=self.forward(x,level)
        return x.shape
    
    def get_layer_shapes(self,input_size):
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        x=torch.rand(2,3,input_size[0],input_size[1])
        x=torch.autograd.Variable(x.to(device).float())
        features=self.forward_layers(x)
        shapes=[f.shape for f in features]
        return shapes
    
    def get_model(self):
        if hasattr(self.config,'backbone_pretrained'):
            pretrained=self.config.backbone_pretrained
        else:
            pretrained=False
                
        if self.use_none_layer:
            from models.psp_resnet import resnet50,resnet101
            from models.psp_vgg import vgg16,vgg19,vgg16_bn,vgg19_bn,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16_gn,vgg19_gn
            #assert self.config.backbone_name in locals().keys(), 'undefine backbone name %s'%self.config.backbone_name
            #assert self.config.backbone_name.find('vgg')>=0,'resnet with momentum is implement in psp_caffe, not here'
            if self.config.backbone_name in ['vgg16','vgg19','vgg16_bn','vgg19_bn','vgg11','vgg11_bn','vgg13','vgg13_bn']:
                return locals()[self.config.backbone_name](pretrained=pretrained, eps=self.eps, momentum=self.momentum)
            else:
                return locals()[self.config.backbone_name](momentum=self.momentum)
        else:
#            print('pretrained=%s backbone in image net'%str(pretrained),'*'*50)
            from torchvision.models import vgg16,vgg19,vgg16_bn,vgg19_bn,resnet50,resnet101,vgg11,vgg11_bn,vgg13,vgg13_bn
            assert self.config.backbone_name in locals().keys(), 'undefine backbone name %s'%self.config.backbone_name
            return locals()[self.config.backbone_name](pretrained=pretrained)
    
    def get_dataframe(self):
        assert self.format=='vgg','only vgg models have features'
        df=pd.DataFrame(columns=['level','layer_depth','layer_name'])
        level=0
        for idx,layer in enumerate(self.features):
            name=layer.__class__.__name__
#            if name in ['ZeroPadding2D','Dropout','Reshape'] or name.find('Pool')>=0:
#                continue
            if name in ['Conv2d'] or name.find('Pool2d')>=0:
                if layer.stride==2:
                    level=level+1
            elif name.find('NoneLayer')>=0:
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
        for idx,layer in enumerate(self.features):
            x=layer(x)
            if idx in self.layer_depths.values():
                layer_outputs.append(x)
        
        return layer_outputs        
    
    def show_layers(self):
        if self.format=='vgg':
            for idx,layer in enumerate(self.features):
                if idx in self.layer_depths.values():
                    print(idx,layer)
        else:
            print('layer 1 '+'*'*50)
            print(self.layer1)
            print('layer 2 '+'*'*50)
            print(self.layer2)
            if config.upsample_layer>=4:
                print('layer 3 '+'*'*50)
                print(self.layer3)
            
            if config.upsample_layer>=5:
                print('layer 4 '+'*'*50)
                print(self.layer4)

if __name__ == '__main__':
    config=edict()
    config.backbone_name='resnet152'
    config.layer_preference='first'
    
    for name in ['vgg16','vgg19','vgg16_bn','vgg19_bn','resnet18','resnet34','resnet50','resnet101','resnet152']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=backbone(config)
        bb.show_layers()
        break