# -*- coding: utf-8 -*-

import torch.nn as TN
from ..backbone import backbone
from ...utils.torch_tools import freeze_layer
from ..upsample import upsample_duc, upsample_bilinear, transform_psp, transform_global


class psp_global(TN.Module):
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

        # psp net will output channels with 2*self.midnet_out_channels
        if self.upsample_type == 'duc':
            r = 2**self.upsample_layer
            self.seg_decoder = upsample_duc(
                2*self.midnet_out_channels, self.class_number, r)
        elif self.upsample_type == 'bilinear':
            self.seg_decoder = upsample_bilinear(
                2*self.midnet_out_channels, self.class_number, self.input_shape[0:2])
        else:
            assert False, 'unknown upsample type %s' % self.upsample_type

        self.gnet_dilation_sizes = self.config.gnet_dilation_sizes
        self.global_decoder = transform_global(
            self.gnet_dilation_sizes, self.class_number)

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        seg = self.seg_decoder(feature_mid)
        g_seg = self.global_decoder(seg)
        return g_seg,seg