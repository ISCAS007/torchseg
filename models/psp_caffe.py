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
from easydict import EasyDict as edict
import torch.utils.data as TD

from utils.torch_tools import do_train_or_val
from dataset.cityscapes import cityscapes
from utils.augmentor import Augmentations
from models.psp_resnet import resnet101,resnet50

class transform_psp(TN.Module):
    def __init__(self, config, pool_sizes, scale, in_channels, out_channels, out_size):
        super(transform_psp, self).__init__()
        path_out_c_list = []
        N = len(pool_sizes)
        mean_c = out_channels//N
        for i in range(N-1):
            path_out_c_list.append(mean_c)

        path_out_c_list.append(out_channels+mean_c-mean_c*N)

        self.pool_sizes = pool_sizes
        self.scale = scale
        self.out_size = out_size
        self.config = config
        self.align_corners=config['align_corners']
        self.inplace=config['inplace']
        self.momentum=config['momentum']
        
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
                                      TN.ReLU(inplace=self.inplace),
                                      TN.Upsample(size=out_size, mode='bilinear', align_corners=self.align_corners))
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
        in_size=x.shape
        if in_size[-1] != min_input_size:
            psp_x = F.upsample(input=x,
                               size=min_input_size,
                               mode='bilinear',
                               align_corners=self.align_corners)
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
        self.dataset_name=self.config.dataset.name
        if self.config.model.backbone_name == 'resnet50':
            self.backbone=resnet50(momentum=self.config.model.momentum)
        else:
            self.backbone=resnet101(momentum=self.config.model.momentum)
        
        self.midnet = self.get_midnet()
        self.suffix_net = self.get_suffix_net()
        
        for m in self.suffix_net.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)
        
        lr = 0.0001
        self.optimizer = torch.optim.Adam(params=[{'params': self.backbone.parameters(), 'lr': lr},
                                             {'params': self.midnet.parameters(),
                                              'lr': 10*lr},
                                             {'params': self.suffix_net.parameters(), 'lr': 20*lr}], lr=lr)
        
        if self.class_number==20:
            #loss_fn=random.choice([torch.nn.NLLLoss(),torch.nn.CrossEntropyLoss()])
            self.loss_fn = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            
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
        config = self.update_config(
            config, self.config.model, 'inplace', False)
        seq = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias),
                            TN.BatchNorm2d(num_features=out_channels,
                                           momentum=config['momentum']),
                            TN.ReLU(inplace=config['inplace']))

        return seq

    def get_midnet(self):
        config = {}
        config = self.update_config(config, self.config.model, 'momentum', 0.1)
        config = self.update_config(
            config, self.config.model, 'inplace', False)
        config = self.update_config(
            config, self.config.model, 'align_corners', False)

        # [1,2,3,6]
        midnet_pool_sizes = self.config.model.midnet_pool_sizes
        #10 or 15
        midnet_scale = self.config.model.midnet_scale

        midnet_in_channels = 2048
        midnet_out_channels = 2048
        
        min_in_size=max(midnet_pool_sizes)*midnet_scale
        midnet_out_size = [min_in_size,min_in_size] 

        return transform_psp(config=config,
                             pool_sizes=midnet_pool_sizes,
                             scale=midnet_scale,
                             in_channels=midnet_in_channels,
                             out_channels=midnet_out_channels,
                             out_size=midnet_out_size)

    # TODO bias=True? scale_factor=None
    def get_suffix_net(self):
        """
        the last conv: bias=True or False ?
        """
        config = {}
        config = self.update_config(config, self.config.model, 'momentum', 0.1)
        config = self.update_config(
            config, self.config.model, 'inplace', False)
        config = self.update_config(
            config, self.config.model, 'align_corners', False)

        seq = TN.Sequential(self.conv_bn_relu(in_channels=2048*2,
                                              out_channels=512,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1,
                                              bias=False),
                            TN.Dropout2d(p=0.1, inplace=config['inplace']),
                            TN.Conv2d(in_channels=512,
                                      out_channels=self.class_number,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      bias=True),
                            TN.Upsample(size=self.input_shape[0:2],
                                        scale_factor=None,
                                        mode='bilinear',
                                        align_corners=config['align_corners']))

        return seq

if __name__ == '__main__':
    config = edict()
    
    input_shape=(240,240)
    config.model = edict()
    config.model.class_number = 20
    config.model.input_shape = input_shape

    config.model.midnet_pool_sizes = [6, 3, 2, 1]
    config.model.midnet_scale = 5
    config.model.midnet_out_channels = 2048
    config.model.momentum=0.95
    config.model.inplace=False
    config.model.align_corners=False

    config.dataset = edict()
    config.dataset.root_path = '/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split = 'train'
    config.dataset.resize_shape = input_shape
    config.dataset.name = 'cityscapes'
    config.dataset.ignore_index=0

    config.args = edict()
    config.args.n_epoch = 100
    config.args.log_dir = '/home/yzbx/tmp/logs/pytorch'
    config.args.note = 'resnet_psp'
    # must change batch size here!!!
    batch_size = 2
    config.args.batch_size = batch_size

    # prefer setting
    config.dataset.norm = True

    augmentations = Augmentations(p=0.25)
    train_dataset = cityscapes(
        config.dataset, split='train', augmentations=augmentations)
    train_loader = TD.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = cityscapes(config.dataset, split='val',
                             augmentations=augmentations)
    val_loader = TD.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
    
    net=psp_caffe(config)
    do_train_or_val(net,config.args,train_loader,val_loader)