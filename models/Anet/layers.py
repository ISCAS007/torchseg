# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import warnings
from easydict import EasyDict as edict

class NoneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return x

class conv_model(nn.Module):
    def __init__(self,in_channels,out_channels,depth,config):
        super().__init__()
        layers=[]
        in_c=in_channels

        for i in range(depth):
            if i>=(depth//2):
                out_c=out_channels
            else:
                out_c=in_c
                
            if config.batch_norm:
                conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
                layers += [conv2d, nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=True)
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            in_c=out_c
            
        self.layers=nn.Sequential(*layers)
        self._initialize_weights()
        
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
    
    def forward(self,x):
        return self.layers(x)
                
class base_model(nn.Module):
    def __init__(self,in_channels,mid_channels,out_channels,depth,config):
        super().__init__()
        layers=[]
        in_c=in_channels
        self.mid_channels=mid_channels
        
        if mid_channels>=in_channels:
            warnings.warn('mid_c {} >= in_c {}'.format(mid_channels,in_channels))
        
        assert depth>2
        self.mid_index=-1
        idx=-1
        for i in range(depth):
            if i<(depth//2):
                out_c=in_c
            elif i==(depth//2):
                out_c=mid_channels
            else:
                out_c=out_channels

            if config.batch_norm:
                conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
                layers += [conv2d, nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
                idx+=3
            else:
                conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=True)
                layers += [conv2d, nn.ReLU(inplace=True)]
                idx+=2                
            if i==(depth//2):
                self.mid_index=idx
            in_c=out_c
        
        self.layers=nn.Sequential(*layers)
        self._initialize_weights()
        
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
    
    def forward(self,x,return_mid_feature=False):
        if return_mid_feature:
            for idx,layer in enumerate(self.layers):
                name=layer.__class__.__name__
                x=layer(x)
                if idx == self.mid_index:
                    mid_feature=x
                    assert name == 'ReLU','name={},idx={}'.format(name,idx)
            return x,mid_feature
        else:
            return self.layers(x)
        
def make_layers(config,in_channels=3):
    layers=[]
    
    if config.use_none_layer:
        pool_cfg=['M','M','M','N','N']
    else:
        pool_cfg=['M','M','M','M','M']
        
    if config.a_depth_mode=='increase':
        depth_cfg=[1,2,3,4,5]
    elif config.a_depth_mode in [str(i) for i in range(5)]:
        depth=int(config.a_depth_mode)
        depth_cfg=[depth for i in range(5)]
    else:
        assert False
        
    if config.a_out_c_mode=='increase':
        out_c_cfg=[64,128,256,512,512]
    else:
        assert False
        
    if config.a_mid_c_mode=='class_power':
        mid_c=config.class_number
        mid_c_cfg=[mid_c*2**i for i in range(5)]
    elif config.a_mid_c_mode=='class_poly':
        mid_c=config.class_number
        mid_c_cfg=[mid_c*2*(i+1) for i in range(5)]
    elif config.a_mid_c_mode=='reduction':
        mid_c_cfg=[c//4 for c in out_c_cfg]
    elif config.a_mid_c_mode in [str(i*10) for i in range(10)]:
        mid_c=int(config.a_mid_channels)
        mid_c_cfg=[mid_c for i in range(5)]
    else:
        assert False
    
    in_c=in_channels
    out_c=out_c_cfg[0]//2
    if config.batch_norm:
        conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=False)
        layers += [conv2d, nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
    else:
        conv2d = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1,bias=True)
        layers += [conv2d, nn.ReLU(inplace=True)]
    
    in_c=out_c
    for idx in range(5):
        if pool_cfg[idx]=='M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif pool_cfg[idx]=='N':
            if config.use_none_layer:
                layers += [NoneLayer()]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            assert False
        
        if depth_cfg[idx]>2:
            a_model=base_model(in_channels=in_c,
                               mid_channels=mid_c_cfg[idx],
                               out_channels=out_c_cfg[idx],
                               depth=depth_cfg[idx],
                               config=config)
        else:
            a_model=conv_model(in_channels=in_c,
                               out_channels=out_c_cfg[idx],
                               depth=depth_cfg[idx],
                               config=config)
            
        layers +=[a_model]
        in_c=out_c_cfg[idx]
            
    return nn.Sequential(*layers)

class Anet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.features=make_layers(config)
        
        self._initialize_weights()
        
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
    
    def get_feature_map_channel(self,level):
        x=torch.rand(2,3,224,224)
        stage_f_list,mid_f_list=self.forward(x,return_mid_feature=True)
        
        return stage_f_list[level].size(1)
                
    def forward(self,x,return_mid_feature=False,level=5):    
        if return_mid_feature:
            mid_f_list=[]
            stage_f_list=[]
            for layer in self.features:
                name=layer.__class__.__name__
                if name=='base_model':
                    x,mid_f=layer(x,True)
                    mid_f_list.append(mid_f)
                elif name in ['MaxPool2d','NoneLayer']:
                    stage_f_list.append(x)
                    x=layer(x)
                else:
                    x=layer(x)
                
                if len(stage_f_list)>level:
                    break
            
            stage_f_list.append(x)
            return stage_f_list,mid_f_list
        else:
            return self.features(x)
        
if __name__ == '__main__':
    config=edict()
    config.class_number=2
    config.use_none_layer=False
    config.a_depth_mode='increase'
    config.a_out_c_mode='increase'
    config.a_mid_c_mode='reduction'
    config.batch_norm=False
    
    anet=Anet(config)
    x=torch.rand(2,3,224,224)
    stage_f_list,mid_f_list=anet(x,return_mid_feature=True)
    for f in stage_f_list:
        print(f.shape)
    for f in mid_f_list:
        print(f.shape)