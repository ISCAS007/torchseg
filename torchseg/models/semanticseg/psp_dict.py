# -*- coding: utf-8 -*-

import torch.nn as TN
from ..backbone import backbone
from ...utils.torch_tools import freeze_layer
from ..upsample import transform_dict,upsample_bilinear,upsample_duc


class psp_dict(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__
        self.backbone = backbone(config)

        if hasattr(self.config, 'backbone_lr_ratio'):
            backbone_lr_raio = self.config.backbone_lr_ratio
            if backbone_lr_raio == 0:
                freeze_layer(self.backbone)

        self.upsample_type = self.config.upsample_type
        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset.name
#        self.midnet_type = self.config.midnet_type
        self.midnet_pool_sizes = self.config.midnet_pool_sizes
        self.midnet_scale = self.config.midnet_scale

        self.midnet_in_channels = self.backbone.get_feature_map_channel(
            self.upsample_layer)
        self.midnet_out_channels = self.config.midnet_out_channels
        self.midnet_out_size = self.backbone.get_feature_map_size(
            self.upsample_layer, self.input_shape[0:2])

        self.midnet = transform_psp(self.midnet_pool_sizes,
                                    self.midnet_scale,
                                    self.midnet_in_channels,
                                    self.midnet_out_channels,
                                    self.midnet_out_size)

        self.dict_number=self.config.dict_number
        assert self.dict_number>=self.class_number,'dict number %d should greadter than class number %d'%(self.dict_number,self.class_number)

        # psp net will output channels with 2*self.midnet_out_channels
        if self.upsample_type == 'duc':
            r = 2**self.upsample_layer
            self.seg_decoder = upsample_duc(
                2*self.midnet_out_channels, self.dict_number, r)
        elif self.upsample_type == 'bilinear':
            self.seg_decoder = upsample_bilinear(
                2*self.midnet_out_channels, self.dict_number, self.input_shape[0:2])
        else:
            assert False, 'unknown upsample type %s' % self.upsample_type


        self.dict_length=self.config.dict_length
        assert self.dict_length>=self.class_number,'dict length %d should greadter than class number %d'%(self.dict_length,self.class_number)
        self.dict_net=transform_dict(self.dict_number,self.dict_length)
        self.dict_conv=TN.Conv2d(in_channels=self.dict_length,
                                 out_channels=self.class_number,
                                 kernel_size=1)
        self.dict_sig=TN.Sigmoid()

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        feature_decoder = self.seg_decoder(feature_mid)
        feature_dict = self.dict_net(feature_decoder)
        seg = self.dict_sig(self.dict_conv(feature_dict))
        return seg