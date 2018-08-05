# -*- coding: utf-8 -*-
import os
from keras.applications import MobileNet,VGG16,VGG19,ResNet50, DenseNet121, DenseNet169, DenseNet201,NASNetMobile, InceptionV3, Xception, InceptionResNetV2
#from resnet import ResNet101,ResNet152
import pandas as pd
import numpy as np
from easydict import EasyDict as edict
import warnings

#BackBoneConfig=namedtuple(typename='BackBoneConfig',
#                          field_names=[
#                                  'application',#VGG16,VGG19,...
#                                  'input_shape',#(224,224,3)
#                                  'weights',#None or ImageNet
#                                  'layer_preference',#first,last,rand
#                                  ])
mobilenet=MobileNet
vgg16=VGG16
vgg19=VGG19
resnet50=ResNet50
#resnet101=ResNet101
#resnet152=ResNet152
densenet121=DenseNet121
densenet169=DenseNet169
densenet201=DenseNet201
nasnetmobile=NASNetMobile
inceptionv3=InceptionV3
xception=Xception
inceptionresnetv2=InceptionResNetV2

class BackBone_Standard():
    def __init__(self,config):
        self.config=config
        
        backbone = self.config.application
        h, w, c = self.config.input_shape
        print('input shape is',self.config.input_shape)
        weights=self.config.weights
        
        if self.config.weights is None:
            self.model = globals()[backbone](include_top=False,weights=weights,input_shape=(h,w,3))
        else:
            h, w, c = self.config.input_shape
            assert h==w,'height and width must be the same'
            if h is not None and h !=224:
                self.model=globals()[backbone](include_top=False, weights=None, input_shape=(h, w, 3))
                weight_model=globals()[backbone](include_top=False, weights='imagenet', input_shape=(224, 224, 3))
                for new_layer, layer in zip(self.model.layers[1:], weight_model.layers[1:]):
                    new_layer.set_weights(layer.get_weights())
            else:
                self.model = globals()[backbone](include_top=False,weights=weights,input_shape=(h,w,3))
                
#        self.model.summary()
        self.df=self.get_dataframe()
                         
    def get_dataframe(self):
        df=pd.DataFrame(columns=['output_size','output_width','output_height','layer_depth','layer_name'])
        for idx,layer in enumerate(self.model.layers):
            name=layer.__class__.__name__
            if name in ['ZeroPadding2D','Dropout','Reshape']:
                continue
            df=df.append({'output_size':layer.output_shape,
                       'output_width':layer.output_shape[1],
                       'output_height':layer.output_shape[2],
                       'layer_depth':idx,
                       'layer_name':layer.name},ignore_index=True)
    
        return df

    def get_layers(self):
        df=self.df
        
        widths=np.unique(df['output_width'].tolist())
        layer_depths=[]
        for width in widths:
            if self.config.layer_preference=='first':
                d=df[df.output_width==width].head(n=1)
                depth=d.layer_depth.tolist()[0]
            elif self.config.layer_preference=='last':
                depth_list=df[df.output_width<width].layer_depth.tolist()
                if len(depth_list)==0:
                    max_depth=len(self.model.layers)+1
                else:
                    max_depth=np.min(depth_list)
                
                d=df[np.logical_and((df.output_width==width),(df.layer_depth<max_depth))].tail(n=1)
                depth=d.layer_depth.tolist()[0]
            elif self.config.layer_preference in ['random','rand']:
                depth_list=df[df.output_width<width].layer_depth.tolist()
                if len(depth_list)==0:
                    max_depth=len(self.model.layers)+1
                else:
                    max_depth=np.min(depth_list)
                
                d=df[np.logical_and((df.output_width==width),(df.layer_depth<max_depth))]
                depth=np.random.choice(d.layer_depth.tolist())
            else:
                print('undefined layer preference',self.config.layer_preference)
                assert False
            
            layer_depths.append(depth)
            
        layers=[self.model.get_layer(index=depth) for depth in layer_depths]
        
        return layers

class BackBone_Custome():
    def __init__(self,config):
        self.config=config

if __name__ == '__main__':
    config=edict()
    config.application='vgg19'
    config.input_shape=(224,224,3)
    config.weights='imagenet'
    config.layer_preference='first'
    
    backbone=BackBone_Standard(config)
    print(backbone.df)
    backbone.model.summary()
    
    layers=backbone.get_layers() 
    
    for l in layers:
        print(l.name,l.output_shape)