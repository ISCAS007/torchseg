# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import upsample_duc,upsample_bilinear,transform_psp,transform_aspp

class pspnet(TN.Module):
    def __init__(self,config):
        super(pspnet,self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        
        if hasattr(self.config.model,'use_momentum'):
            use_momentum=self.config.model.use_momentum
        else:
            use_momentum=False
        
        self.backbone=backbone(config.model,use_momentum=use_momentum)
        
        self.upsample_type = self.config.model.upsample_type
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        if hasattr(self.config.model,'midnet_name'):
            self.midnet_name = self.config.model.midnet_name
        else:
            self.midnet_name = 'psp'
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.midnet_out_channels=self.config.model.midnet_out_channels
        
        if hasattr(self.config.model,'eps'):
            eps=self.config.model.eps
        else:
            eps=1e-5
        
        if hasattr(self.config.model,'momentum'):
            momentum=self.config.model.momentum
        else:
            momentum=0.1
        
        if self.midnet_name=='psp':
            print('midnet is psp'+'*'*50)
            self.midnet_pool_sizes=self.config.model.midnet_pool_sizes
            self.midnet_scale=self.config.model.midnet_scale
            self.midnet=transform_psp(self.midnet_pool_sizes,
                                      self.midnet_scale,
                                      self.midnet_input_shape,
                                      self.midnet_out_channels,
                                      eps=eps, 
                                      momentum=momentum)
        elif self.midnet_name=='aspp':
            print('midnet is aspp'+'*'*50)
            output_stride=2**self.config.model.upsample_layer
            self.midnet=transform_aspp(output_stride=output_stride,
                                       input_shape=self.midnet_input_shape,
                                       out_channels=self.midnet_out_channels,
                                       eps=eps,
                                       momentum=momentum)
        
        # psp net will output channels with 2*self.midnet_out_channels
        if self.upsample_type=='duc':
            r=2**self.upsample_layer
            self.decoder=upsample_duc(self.midnet_out_channels,self.class_number,r,eps=eps,momentum=momentum)
        elif self.upsample_type=='bilinear':
            self.decoder=upsample_bilinear(self.midnet_out_channels,self.class_number,self.input_shape[0:2],eps=eps,momentum=momentum)
        else:
            assert False,'unknown upsample type %s'%self.upsample_type


    def forward(self, x):
        feature_map=self.backbone.forward(x,self.upsample_layer)
        feature_mid=self.midnet(feature_map)
        x=self.decoder(feature_mid)        

        return x