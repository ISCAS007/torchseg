# -*- coding: utf-8 -*-
"""write psp model directly from caffe proto.txt
1. first model `from utils.torch_tools import do_train_or_val`
2. exclusive `transfrom_psp`
3. modified resnet
4. layer-wise learning rate
5. right pool size [6,3,2,1] and scale 5 due the limit of memory
6. init model with `msra` or `kaiming_normal` mode
7. no ignore_index supported, class number is 20 current
"""
import torch
import torch.nn as TN

import torch.nn.functional as F
from models.upsample import transform_aspp
from models.psp_resnet import resnet101, resnet50


class transform_psp(TN.Module):
    def __init__(self, pool_sizes, scale, in_channels, out_channels, out_size, momentum=0.1):
        super(transform_psp, self).__init__()
        path_out_c_list = []
        N = len(pool_sizes)
        mean_c = out_channels//(2*N)
        assert out_channels % 2 == 0, 'psp output channels %d %%2 !=0' % out_channels

        for i in range(N-1):
            path_out_c_list.append(mean_c)

        path_out_c_list.append(out_channels+mean_c-mean_c*N)

        self.pool_sizes = pool_sizes
        self.scale = scale
        self.out_size = out_size
        self.momentum = momentum

        pool_paths = []
        for pool_size, out_c in zip(pool_sizes, path_out_c_list):
            pool_path = TN.Sequential(TN.AvgPool2d(kernel_size=pool_size*scale,
                                                   stride=pool_size*scale,
                                                   padding=0),
                                      TN.Conv2d(in_channels=in_channels,
                                                out_channels=out_c,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False),
                                      TN.BatchNorm2d(
                                          num_features=out_c, momentum=self.momentum),
                                      TN.ReLU(inplace=False),
                                      TN.Upsample(size=out_size, mode='bilinear'))
            pool_paths.append(pool_path)

        self.pool_paths = TN.ModuleList(pool_paths)

        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):

        min_input_size = max(self.pool_sizes)*self.scale
        in_size = x.shape
        if in_size[-1] != min_input_size:
            psp_x = F.upsample(input=x,
                               size=min_input_size,
                               mode='bilinear',
                               align_corners=True)
        else:
            psp_x = x

        output_slices = [psp_x]
#        print('psp_x shape',psp_x.shape)
#        print('min_input_size is',min_input_size)
#        print('self out_size is',self.out_size)
        for module in self.pool_paths:
            x = module(psp_x)
            output_slices.append(x)

        x = torch.cat(output_slices, dim=1)
        return x


class psp_caffe(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__

        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.ignore_index = self.config.dataset.ignore_index
        self.dataset_name = self.config.dataset.name
        self.momentum = self.config.model.momentum
        if self.config.model.backbone_name == 'resnet50':
            self.backbone = resnet50(momentum=self.config.model.momentum,
                                     upsample_layer=self.config.model.upsample_layer)
        else:
            self.backbone = resnet101(momentum=self.config.model.momentum,
                                      upsample_layer=self.config.model.upsample_layer)

        if hasattr(self.config.model, 'midnet_name'):
            self.midnet_name = self.config.model.midnet_name
        else:
            self.midnet_name = 'psp'
        self.midnet = self.get_midnet()
        self.suffix_net = self.get_suffix_net()

        for m in self.suffix_net.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

        self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1},
                                 {'params': self.midnet.parameters(),
                                  'lr_mult': 10},
                                 {'params': self.suffix_net.parameters(), 'lr_mult': 20}]

        print('class number is %d' % self.class_number,
              'ignore_index is %d' % self.ignore_index, '*'*30)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.ignore_index)

#        print(self.backbone)

    def forward(self, x):
        x = self.backbone(x)
#        print('backbone output',x.shape)
        x = self.midnet(x)
#        print('midnet output',x.shape)
        x = self.suffix_net(x)
#        print('suffix net output',x.shape)

        return x

    @staticmethod
    def update_config(dst, src, key, default_value):
        if hasattr(src, key):
            dst[key] = src[key]
        else:
            dst[key] = default_value

        return dst

    def conv_bn_relu(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        config = {}
        config = self.update_config(config, self.config.model, 'momentum', 0.1)
        seq = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias),
                            TN.BatchNorm2d(num_features=out_channels,
                                           momentum=config['momentum']),
                            TN.ReLU(inplace=False))

        return seq

    def get_midnet(self):
        if self.midnet_name == 'psp':
            # [1,2,3,6]
            midnet_pool_sizes = self.config.model.midnet_pool_sizes
            #10 or 15
            midnet_scale = self.config.model.midnet_scale

            midnet_in_channels = self.config.model.midnet_in_channels
            midnet_out_channels = self.config.model.midnet_out_channels

            min_in_size = max(midnet_pool_sizes)*midnet_scale
            midnet_out_size = [min_in_size, min_in_size]

            return transform_psp(pool_sizes=midnet_pool_sizes,
                                 scale=midnet_scale,
                                 in_channels=midnet_in_channels,
                                 out_channels=midnet_out_channels,
                                 out_size=midnet_out_size,
                                 momentum=self.momentum)
        elif self.midnet_name == 'aspp':
            output_stride = 2**self.config.model.upsample_layer
            batch_size = None
            midnet_in_channels = self.config.model.midnet_in_channels
            midnet_out_channels = self.config.model.midnet_out_channels
            midnet_input_size = [
                a//output_stride for a in self.input_shape[0:2]]
            input_shape = (batch_size, midnet_in_channels,
                           midnet_input_size[0], midnet_input_size[1])
            if hasattr(self.config.model, 'eps'):
                eps = self.config.model.eps
            else:
                eps = 1e-5

            return transform_aspp(output_stride=output_stride,
                                  input_shape=input_shape,
                                  out_channels=midnet_out_channels,
                                  eps=eps,
                                  momentum=self.momentum)
        else:
            assert False, 'unknown midnet name %s' % self.midnet_name

    # TODO bias=True? scale_factor=None
    def get_suffix_net(self):
        """
        the last conv: bias=True or False ?
        """
        midnet_out_channels = self.config.model.midnet_out_channels
        suffix_out_channels = self.config.model.suffix_out_channels
        seq = TN.Sequential(self.conv_bn_relu(in_channels=midnet_out_channels,
                                              out_channels=suffix_out_channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=False),
                            TN.Dropout2d(p=0.1, inplace=False),
                            TN.Conv2d(in_channels=suffix_out_channels,
                                      out_channels=self.class_number,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True),
                            TN.Upsample(size=self.input_shape[0:2],
                                        scale_factor=None,
                                        mode='bilinear'))

        return seq
