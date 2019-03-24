# -*- coding: utf-8 -*-
import pandas as pd
import torch.nn.functional as F
import torch.nn as TN
import numpy as np
import torch
from torch.autograd import Variable
from easydict import EasyDict as edict
import warnings
from models.upsample import local_upsample
from models.MobileNetV2 import mobilenet2

class conv_bn_relu(TN.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 eps=1e-5,
                 momentum=0.1,
                 inplace=False):
        """
        out_channels: class number
        upsample_ratio: 2**upsample_layer
        """
        super().__init__()
        bias = False
        self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    stride=stride,
                                                    bias=bias),
                                          TN.BatchNorm2d(num_features=out_channels,
                                                   eps=eps,
                                                   momentum=momentum),
                                          TN.ReLU(inplace=inplace))
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x
    
class motion_backbone(TN.Module):
    def __init__(self,config,use_none_layer=False):
        """
        use_none_layer: use NoneLayer to replace MaxPool in backbone
        """
        super().__init__()
        self.config=config
        self.use_none_layer=use_none_layer
        self.upsample_layer=self.config.upsample_layer
        self.deconv_layer=self.config.deconv_layer
        
        if config.net_name.find('unet')>=0 or config.net_name.find('motion_sparse')>=0:
            assert self.deconv_layer > self.upsample_layer,'deconv %d must > decoder %d'%(self.deconv_layer,self.upsample_layer)
        elif config.net_name.find('fcn')>=0 or config.net_name.find('motion_psp')>=0:
            self.deconv_layer = self.upsample_layer
        else:
            print('unknown net name {} for max extracted layer index'.format(config.net_name))
            assert self.deconv_layer > self.upsample_layer
    
        if hasattr(self.config,'eps'):
            self.eps=self.config.eps
        else:
            self.eps=1e-5
        
        if hasattr(self.config,'momentum'):
            self.momentum=self.config.momentum
        else:
            self.momentum=0.1
            
        if use_none_layer == False:
            model=self.get_model()
            if self.config.backbone_name.find('vgg')>=0 or self.config.backbone_name.lower().find('mobilenet')>=0:
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
                self.layer3=model.layer3
                self.layer4=model.layer4
            else:
                assert False,'unknown backbone name %s'%self.config.backbone_name
        else:
            model=self.get_model()
            if self.config.backbone_name.find('vgg')>=0 or self.config.backbone_name.lower().find('mobilenet')>=0:
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
                self.layer3=model.layer3
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
        
        # if modify resnet head worked, train the modified resnet head
        if config.modify_resnet_head and self.config.use_none_layer and self.format=='resnet':
            for param in self.prefix_net.parameters():
                param.requires_grad = True
    
    def forward_layers(self,x):
        features=[]
        if self.format=='vgg':
            layer_num=0
            if not hasattr(self.layer_depths,str(layer_num)):
                features.append(x)
                layer_num+=1

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
        
    def forward(self,x,level):        
        if self.format=='vgg':
            if not hasattr(self.layer_depths,str(level)):
                return x
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
            print('warning: backbone is not pretrained!!!')
            pretrained=False
                
        if self.use_none_layer:
            print('use none layer'+'*'*30)
            from models.psp_resnet import resnet50,resnet101
            from models.psp_vgg import vgg16,vgg19,vgg16_bn,vgg19_bn,vgg11,vgg11_bn,vgg13,vgg13_bn,vgg16_gn,vgg19_gn
            #assert self.config.backbone_name in locals().keys(), 'undefine backbone name %s'%self.config.backbone_name
            #assert self.config.backbone_name.find('vgg')>=0,'resnet with momentum is implement in psp_caffe, not here'
            if self.config.backbone_name in ['vgg16','vgg19','vgg16_bn','vgg19_bn','vgg11','vgg11_bn','vgg13','vgg13_bn']:
                return locals()[self.config.backbone_name](pretrained=pretrained, eps=self.eps, momentum=self.momentum)
            elif self.config.backbone_name == 'MobileNetV2':
                return mobilenet2(pretrained=pretrained)
            else:
                return locals()[self.config.backbone_name](momentum=self.momentum)
        else:
#            print('pretrained=%s backbone in image net'%str(pretrained),'*'*50)
            from torchvision.models import vgg16,vgg19,vgg16_bn,vgg19_bn,resnet50,resnet101,vgg11,vgg11_bn,vgg13,vgg13_bn
            from models.psp_vgg import vgg16_gn,vgg19_gn
            if self.config.backbone_name == 'MobileNetV2':
                return mobilenet2(pretrained=pretrained)
            else:
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
            if name in ['Conv2d','InvertedResidual'] or name.find('Pool2d')>=0:
                if layer.stride==2 or layer.stride==(2,2) :
                    level=level+1
            elif name.find('NoneLayer')>=0:
                level=level+1
            elif name == 'Sequential':
                for l in layer:
                    l_name=l.__class__.__name__
                    if l_name in ['Conv2d'] or l_name.find('Pool2d')>=0:
                        if l.stride==2 or l.stride==(2,2):
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

class transform_motionnet(TN.Module):
    """
    segnet or unet as midnet
    """
    def __init__(self,backbone,config):
        super().__init__()
        self.config=config
        self.use_none_layer=self.config.use_none_layer
        self.upsample_layer=self.config.upsample_layer
        self.deconv_layer=self.config.deconv_layer
        self.merge_type=self.config.merge_type
        self.use_aux_input=self.config.use_aux_input
        
        self.layers=[]
        
        self.concat_layers=[]
        
        
        inplace=True
        in_c=out_c=0
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.layers.append(None)
                self.concat_layers.append(None)
            elif idx==self.deconv_layer:
                in_c=out_c=backbone.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                if self.use_none_layer and idx>3:
                    layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=TN.Sequential(TN.ConvTranspose2d(in_c,in_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.layers.append(layer)
                if self.merge_type=='concat':
                    merge_c=2*out_c if self.use_aux_input else out_c
                    if self.use_aux_input:
                        self.concat_layers.append(conv_bn_relu(in_channels=merge_c,
                                                        out_channels=in_c,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        inplace=inplace))
                    else:
                        self.concat_layers.append(None)
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
            else:
                in_c=backbone.get_feature_map_channel(idx+1)
                out_c=backbone.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                
                if (self.use_none_layer and idx>3) or idx==0:
                    layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=TN.Sequential(TN.ConvTranspose2d(in_c,in_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.layers.append(layer)
                if self.merge_type=='concat':
                    merge_c=in_c+2*out_c if self.use_aux_input else in_c+out_c
                    self.concat_layers.append(conv_bn_relu(in_channels=merge_c,
                                                    out_channels=in_c,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    inplace=inplace))
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
            
        self.model_layers=TN.ModuleList([layer for layer in self.layers if layer is not None])
        if self.merge_type=='concat':
            self.merge_layers=TN.ModuleList([layer for layer in self.concat_layers if layer is not None])
        else:
            assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
    
    def forward(self,main,aux=None):
        if self.use_aux_input:
            for x in [main,aux]:
                assert isinstance(x,(list,tuple)),'input for segnet should be list or tuple'
                assert len(x)==6
            
            for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
                if idx==self.deconv_layer:
                    if self.merge_type=='concat':
                        feature=torch.cat([main[idx],aux[idx]],dim=1)
                        feature=self.concat_layers[idx](feature)
                    else:
                        assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                        feature=main[idx]+aux[idx]
                    feature=self.layers[idx](feature)
                else:
                    
                    if self.merge_type=='concat':
                        feature=torch.cat([feature,main[idx],aux[idx]],dim=1)
                        feature=self.concat_layers[idx](feature)
                    else:
                        assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                        feature+=main[idx]+aux[idx]
                    
                    feature=self.layers[idx](feature)
            return feature
        else:
            assert aux is None
            assert isinstance(main,(list,tuple)),'main input for segnet should be list or tuple'
            assert len(main)==6
            
            for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
                if idx==self.deconv_layer:
                    feature=main[idx]
                else:
                    if self.merge_type=='concat':
                        feature=torch.cat([feature,main[idx]],dim=1)
                        feature=self.concat_layers[idx](feature)
                    else:
                        assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                        feature+=main[idx]
                        
                feature=self.layers[idx](feature)
                
            return feature

class upsample_merge(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.merge_type=config.merge_type
        
    def forward(self,features):
        s=features[0].shape
        
        y=[]
        for f in features:
            if f.shape[2:4]!=s[2:4]:
                f=F.interpolate(f, size=s[2:4],
                                mode='bilinear', align_corners=True)
            y.append(f)
        
        if self.merge_type=='concat':
                merge_feature=torch.cat(y,dim=1)
        else:
            assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
            for idx in range(len(y)):
                if idx==0:
                    merge_feature=y[idx]
                else:
                    merge_feature+=y[idx]
        
        return merge_feature
    
class transform_motionnet_flow(TN.Module):
    def __init__(self,backbone,config):
        super().__init__()
        self.config=config
        self.use_none_layer=self.config.use_none_layer
        self.upsample_layer=self.config.upsample_layer
        self.deconv_layer=self.config.deconv_layer
        self.merge_type=self.config.merge_type
        self.always_merge_flow=self.config.always_merge_flow
        
        assert self.deconv_layer > self.upsample_layer
        assert self.merge_type=='concat'
        self.upsample_merge_layers=[]
        self.deconv_layers=[]
        inplace=True
        for idx in range(self.deconv_layer+1):
            if idx<self.upsample_layer:
                self.upsample_merge_layers.append(None)
                self.deconv_layers.append(None)
            elif idx==self.deconv_layer:
                in_c=out_c=backbone.get_feature_map_channel(idx)
                merge_c=in_c+2
                self.upsample_merge_layers.append(TN.Sequential(upsample_merge(config),
                                                               conv_bn_relu(in_channels=merge_c,
                                                                out_channels=in_c,
                                                                kernel_size=1,
                                                                stride=1,
                                                                padding=0,
                                                                inplace=inplace)))
                
                if self.use_none_layer and idx>3:
                    layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=TN.Sequential(TN.ConvTranspose2d(in_c,in_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.deconv_layers.append(layer)
            else:
                in_c=backbone.get_feature_map_channel(idx+1)
                out_c=backbone.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                merge_c=in_c+out_c
                
                if self.always_merge_flow:
                    merge_c+=2
                self.upsample_merge_layers.append(TN.Sequential(upsample_merge(config),
                                                               conv_bn_relu(in_channels=merge_c,
                                                                out_channels=in_c,
                                                                kernel_size=1,
                                                                stride=1,
                                                                padding=0,
                                                                inplace=inplace)))
                if (self.use_none_layer and idx>3) or idx==0:
                    layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                else:
                    layer=TN.Sequential(TN.ConvTranspose2d(in_c,in_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     inplace=inplace))
                self.deconv_layers.append(layer)
                
        self.model_layers=TN.ModuleList([layer for layer in self.deconv_layers if layer is not None])
        self.merge_layers=TN.ModuleList([layer for layer in self.upsample_merge_layers if layer is not None])
            
    def forward(self,main,flow):
        assert isinstance(main,(list,tuple)),'main input for segnet should be list or tuple'
        assert len(main)==6
        
        for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
            if idx==self.deconv_layer:
                feature=self.upsample_merge_layers[idx]((main[idx],flow))
            else:
                if self.always_merge_flow:
                    feature=self.upsample_merge_layers[idx]((main[idx],feature,flow))
                else:
                    feature=self.upsample_merge_layers[idx]((main[idx],feature))
            feature=self.deconv_layers[idx](feature)
        return feature
    

class motionnet_upsample_bilinear(TN.Module):
    def __init__(self, in_channels, out_channels, output_shape, eps=1e-5, momentum=0.1):
        """
        out_channels: class number
        """
        super().__init__()
        self.output_shape = output_shape
        self.conv = TN.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=False)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    # TODO upsampel feature is self.conv_bn_relu(x) or self.conv(x)
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)

        return x
    
class transform_sparse(TN.Module):
    """
    midnet of motion_sparse
    """
    def __init__(self,backbone,config):
        super().__init__()
        self.config=config
        self.upsample_layer=self.config.upsample_layer
        self.deconv_layer=self.config.deconv_layer
        self.sparse_ratio=self.config.sparse_ratio
        self.sparse_conv=self.config.sparse_conv
        self.concat_size=backbone.get_feature_map_size(self.upsample_layer,self.config.input_shape)
        
        self.concat_channels=0
        sparse_convs=[]
        shapes=backbone.get_layer_shapes(self.config.input_shape)
        for idx in range(self.upsample_layer,self.deconv_layer+1):
            in_channels=shapes[idx][1]
            out_channels=int(in_channels*self.sparse_ratio)
            sparse_convs.append(TN.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=False))
            self.concat_channels+=out_channels
        self.concat_channels*=2
        self.sparse_convs=TN.ModuleList(sparse_convs)
            
        
    def forward(self,main,aux):
        for x in [main,aux]:
            assert isinstance(x,(list,tuple)),'input for segnet should be list or tuple'
            assert len(x)==6
        
        features=[]
        for idx in range(self.upsample_layer,self.deconv_layer+1):
            channels=int(main[idx].shape[1]*self.sparse_ratio)
            if self.sparse_conv:
                main_sparse=self.sparse_convs[idx-self.upsample_layer](main[idx])
                aux_sparse=self.sparse_convs[idx-self.upsample_layer](aux[idx])
                features+=[F.interpolate(x, size=self.concat_size,
                          mode='bilinear', align_corners=True) \
                        for x in [main_sparse,aux_sparse]]
            else:
                features+=[F.interpolate(x, size=self.concat_size,
                              mode='bilinear', align_corners=True) \
                            for x in [main[idx][:,0:channels],aux[idx][:,0:channels]]]
        
        concat_feature=torch.cat(features,dim=1)
        return concat_feature
    
class transform_motion_psp(TN.Module):
    """x->4x[pool->conv->bn->relu->upsample]->concat
    input_shape[batch_size,channel,height,width]
    height:lcm(pool_sizes)*scale*(2**upsample_ratio)
    width:lcm(pool_sizes)*scale*(2**upsample_ratio)
    lcm: least common multiple, lcm([4,5])=20, lcm([2,3,6])=6

    for feature layer with output size (batch_size,channel,height=x,width=x)
    digraph G {
      "feature[x,x]" -> "pool6[x/6*scale,x/6*scale]" -> "conv_bn_relu6[x/6*scale,x/6*scale]" -> "interp6[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool3[x/3*scale,x/3*scale]" -> "conv_bn_relu3[x/3*scale,x/3*scale]" -> "interp3[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool2[x/2*scale,x/2*scale]" -> "conv_bn_relu2[x/2*scale,x/2*scale]" -> "interp2[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool1[x/1*scale,x/1*scale]" -> "conv_bn_relu1[x/1*scale,x/1*scale]" -> "interp1[x,x]" -> "concat[x,x]"
    }
    """

    def __init__(self, pool_sizes, scale, input_shape, out_channels, eps=1e-5, momentum=0.1):
        """
        pool_sizes = [1,2,3,6]
        scale = 5,10
        out_channels = the output channel for transform_psp
        out_size = ?
        """
        super().__init__()
        self.input_shape = input_shape
        b, in_channels, height, width = input_shape
        in_channels*=2
        
        path_out_c_list = []
        N = len(pool_sizes)

        assert out_channels > in_channels, 'out_channels will concat inputh, so out_chanels=%d should >= in_chanels=%d' % (
            out_channels, in_channels)
        # the output channel for pool_paths
        pool_out_channels = out_channels-in_channels

        mean_c = pool_out_channels//N
        for i in range(N-1):
            path_out_c_list.append(mean_c)

        path_out_c_list.append(pool_out_channels+mean_c-mean_c*N)
        
        self.pool_sizes = pool_sizes
        self.scale = scale
        pool_paths = []
        for pool_size, out_c in zip(pool_sizes, path_out_c_list):
            pool_path = TN.Sequential(TN.AvgPool2d(kernel_size=pool_size*scale,
                                                   stride=pool_size*scale,
                                                   padding=0),
                                      conv_bn_relu(in_channels=in_channels,
                                                out_channels=out_c,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0),
                                      local_upsample(size=(height, width), mode='bilinear', align_corners=True))
            pool_paths.append(pool_path)

        self.pool_paths = TN.ModuleList(pool_paths)

        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, main,aux):
        x=torch.cat([main,aux],dim=1)
        output_slices = [x]
        min_input_size = max(self.pool_sizes)*self.scale
        in_size = x.shape
        assert in_size[2] >= min_input_size, 'psp in size %d should >= %d' % (
            in_size[2], min_input_size)

        for module in self.pool_paths:
            y = module(x)
            output_slices.append(y)

        x = torch.cat(output_slices, dim=1)
        return x
    
if __name__ == '__main__':
    config=edict()
    config.backbone_name='resnet152'
    config.layer_preference='first'
    config.upsample_layer=1
    config.deconv_layer=5
    config.freeze_layer=1
    config.backbone_freeze=False
    config.modify_resnet_head=False
    
    for name in ['vgg11','vgg16','vgg19_bn','resnet50']:
        print(name+'*'*50)
        config.backbone_name=name
        bb=motion_backbone(config)
        bb.show_layers()
        print(bb.get_layer_shapes([224,224]))