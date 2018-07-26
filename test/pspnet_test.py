# -*- coding: utf-8 -*-
import torch
import torch.utils.data as TD
import random
from dataset.cityscapes import cityscapes
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
import argparse

from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from models.psp_caffe import psp_caffe
from utils.augmentor import Augmentations
from utils.torch_tools import do_train_or_val

if __name__ == '__main__':
    config = edict()
    config.model = edict()
    config.model.upsample_type = 'duc'
    config.model.upsample_layer = 3
    config.model.class_number = 20
    config.model.backbone_name = 'vgg16'
    config.model.layer_preference = 'last'
    input_shape = (240, 240)
    config.model.input_shape = input_shape

    config.model.midnet_pool_sizes = [6, 3, 2, 1]
    config.model.midnet_scale = 5
    config.model.midnet_out_channels = 512

    config.dataset = edict()
    config.dataset.root_path = '/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split = random.choice(['test', 'val', 'train'])
    config.dataset.resize_shape = input_shape
    config.dataset.name = 'cityscapes'

    config.args = edict()
    config.args.n_epoch = 100
    config.args.log_dir = '/home/yzbx/tmp/logs/pytorch'
    config.args.note = 'aug'
    # must change batch size here!!!
    batch_size = 2
    config.args.batch_size = batch_size

    # prefer setting
    config.model.backbone_lr_ratio = 1.0
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

    choices = ['disc', 'ignore_index', 'norm', 'edge', 'global',
               'backbone', 'dict', 'fractal', 'optim', 'upsample_type', 'caffe']
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="test for choices",
                        choices=choices,
                        default=random.choice(choices))

    args = parser.parse_args()
    test = args.test
    if test == 'disc':
        for backbone_lr_ratio in [0.0, 1.0]:
            config.model.backbone_lr_ratio = backbone_lr_ratio
            for norm in [True, False]:
                config.args.note = 'aug'
                config.dataset.norm = norm
                norm_str = 'norm' if norm else 'no_norm'
                config.args.note = '_'.join(
                    [config.args.note, 'bb_lr', str(backbone_lr_ratio), norm_str])
                net = pspnet(config)
                net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'ignore_index':
        config.args.ignore_index = True
        config.args.note = 'ignore_index'
        config.dataset.norm = True
        config.dataset.ignore_index = 255
        config.model.class_number = 19
        norm_str = 'norm'
        backbone_lr_ratio = 1.0
        config.args.note = '_'.join(
            [config.args.note, 'bb_lr', str(backbone_lr_ratio), norm_str])

        train_dataset = cityscapes(
            config.dataset, split='train', augmentations=augmentations)
        train_loader = TD.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

        val_dataset = cityscapes(
            config.dataset, split='val', augmentations=augmentations)
        val_loader = TD.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

        net = pspnet(config)
        net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'norm':
        config.args.n_epoch = 300
        for norm in [True, False]:
            config.args.note = 'norm'
            config.dataset.norm = norm
            config.args.note = '_'.join([config.args.note, str(norm)])
            net = pspnet(config)
            net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'edge':
        config.args.n_epoch = 100
        config.dataset.with_edge = True
        for edge_width in [10]:
            config.args.note = 'edge_width'
            config.dataset.edge_width = edge_width
            config.args.note = '_'.join([config.args.note, str(edge_width)])
            net = psp_edge(config)
            net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'global':
        config.args.n_epoch = 100
        config.model.gnet_dilation_sizes = [16, 8, 4]
        config.args.note = 'default'
        net = psp_global(config)
        net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'backbone':
        config.args.n_epoch = 100
        for backbone in ['resnet50', 'resnet101']:
            config.model.backbone_name = backbone
            config.args.note = '_'.join([config.args.note, backbone])
            net = pspnet(config)
            net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'dict':
        config.args.n_epoch = 100
        config.args.note = 'dict'
        dict_number = config.model.class_number*5+1
        dict_lenght = config.model.class_number*2+1
        config.model.dict_number = dict_number
        config.model.dict_length = dict_lenght
        config.args.note = '_'.join(
            [config.args.note, '%dx%d' % (dict_number, dict_lenght)])
        net = psp_dict(config)
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'fractal':
        config.args.n_epoch = 100
        config.args.note = 'fractal'
        before_upsample = True
        fractal_depth = 8
        fractal_fusion_type = 'mean'
        config.model.before_upsample = before_upsample
        config.model.fractal_depth = fractal_depth
        config.model.fractal_fusion_type = fractal_fusion_type

        location_str = 'before' if before_upsample else 'after'
        config.args.note = '_'.join([config.args.note, location_str, 'depth', str(
            fractal_depth), 'fusion', fractal_fusion_type])
        net = psp_fractal(config)
        net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'optim':
        config.args.n_epoch = 100
        for optim in ['adam', 'sgd_simple', 'sgd_complex']:
            for lr in [0.01, 0.001, 0.0001]:
                config.args.note = '_'.join([optim, str(lr)])
                config.args.optim = optim

                net = pspnet(config)
                net.do_train_or_val(config.args, train_loader, val_loader)

            break
    elif test == 'upsample_type':
        config.args.n_epoch = 100
        backbone = 'resnet101'
        config.model.backbone_name = backbone
        for upsample_type in ['duc', 'bilinear']:
            config.model.upsample_type = upsample_type
            config.args.note = '_'.join([backbone, upsample_type, 'keras_psp'])
            net = pspnet(config)
            net.do_train_or_val(config.args, train_loader, val_loader)
    elif test == 'caffe':
        config.args.n_epoch = 100
        backbone = 'resnet50'
        config.model.backbone_name = backbone
        input_shape = (240, 240)
        config.model.input_shape = input_shape
        config.dataset.resize_shape = input_shape
        config.dataset.ignore_index = 255
        config.model.class_number = 19
        config.model.midnet_pool_sizes = [6, 3, 2, 1]
        config.model.midnet_scale = 5
        config.model.momentum = 0.95
        align_corners = False
        config.model.align_corners = align_corners

        # change dataset setting, the dataloader need be update
        train_dataset = cityscapes(
            config.dataset, split='train', augmentations=augmentations)
        train_loader = TD.DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

        val_dataset = cityscapes(
            config.dataset, split='val', augmentations=augmentations)
        val_loader = TD.DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
#        for momentum in [0.1, 0.5, 0.9]:
#            config.model.momentum = momentum
#            config.args.note = '_'.join([backbone,
#                                         'cls'+str(config.model.class_number),
#                                         'align'+str(align_corners)[0],
#                                         'mo'+str(momentum)
#                                         ])
#            net = psp_caffe(config)
#            do_train_or_val(net, config.args, train_loader, val_loader)

        for align_corners in [True, False]:
            config.model.align_corners = align_corners
            config.args.note = '_'.join([backbone,
                                         'cls'+str(config.model.class_number),
                                         'align'+str(align_corners)[0]
                                         ])
            net = psp_caffe(config)
            do_train_or_val(net, config.args, train_loader, val_loader)

    else:
        raise NotImplementedError
