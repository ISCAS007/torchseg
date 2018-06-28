# -*- coding: utf-8 -*-
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
from utils.augmentor import Augmentations

if __name__ == '__main__':
    config=edict()
    config.model=edict()
    config.model.upsample_type='duc'
    config.model.upsample_layer=3
    config.model.class_number=20
    config.model.backbone_name='vgg16'
    config.model.layer_preference='last'
    config.model.input_shape=(224,224)
    
    config.model.midnet_pool_sizes=[6,3,2,1]
    config.model.midnet_scale=5
    config.model.midnet_out_channels=512
    
    config.dataset=edict()
    config.dataset.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split=random.choice(['test','val','train'])
    config.dataset.resize_shape=(224,224)
    config.dataset.name='cityscapes'
    
    augmentations=Augmentations(p=0.25)
    train_dataset=cityscapes(config.dataset,split='train',augmentations=augmentations)
    train_loader=TD.DataLoader(dataset=train_dataset,batch_size=32, shuffle=True,drop_last=False,num_workers=8)
    
    val_dataset=cityscapes(config.dataset,split='val',augmentations=augmentations)
    val_loader=TD.DataLoader(dataset=val_dataset,batch_size=32, shuffle=True,drop_last=False,num_workers=8)
    
    config.args=edict()
    config.args.n_epoch=100
    config.args.log_dir='/home/yzbx/tmp/logs/pytorch'
    config.args.note='aug'
    
    # prefer setting
    config.model.backbone_lr_ratio=1.0
    config.dataset.norm=True
    
    
    choices=['disc','ignore_index','norm','edge','global','backbone']
    parser=argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="test for choices",
                        choices=choices,
                        default=random.choice(choices))

    args = parser.parse_args()
    test=args.test
    if test == 'disc':
        for backbone_lr_ratio in [0.0,1.0]:
            config.model.backbone_lr_ratio=backbone_lr_ratio
            for norm in [True,False]:
                config.args.note='aug'
                config.dataset.norm=norm
                norm_str='norm' if norm else 'no_norm'
                config.args.note='_'.join([config.args.note,'bb_lr',str(backbone_lr_ratio),norm_str])
                net=pspnet(config)
                net.do_train_or_val(config.args,train_loader,val_loader)
    elif test == 'ignore_index':
        config.args.ignore_index=True
        config.args.note='ignore_index'
        config.dataset.norm=True
        norm_str='norm'
        backbone_lr_ratio=1.0
        config.args.note='_'.join([config.args.note,'bb_lr',str(backbone_lr_ratio),norm_str])
        net=pspnet(config)
        net.do_train_or_val(config.args,train_loader,val_loader)
    elif test == 'norm':
        config.args.n_epoch=300
        for norm in [True,False]:
            config.args.note='norm'
            config.dataset.norm=norm
            config.args.note='_'.join([config.args.note,str(norm)])
            net=pspnet(config)
            net.do_train_or_val(config.args,train_loader,val_loader)
    elif test == 'edge':
        config.args.n_epoch=100
        config.dataset.with_edge=True
        for edge_width in [1,5,10]:
            config.args.note='edge_width'
            config.dataset.edge_width=edge_width
            config.args.note='_'.join([config.args.note,str(edge_width)])
            net=psp_edge(config)
            net.do_train_or_val(config.args,train_loader,val_loader)
    elif test == 'global':
        config.args.n_epoch=100
        config.model.gnet_dilation_sizes=[16,8,4]
        config.args.note='default'
        net=psp_global(config)
        net.do_train_or_val(config.args,train_loader,val_loader)
    elif test == 'backbone':
        config.args.n_epoch=300
        for backbone in ['resnet50','resnet101','resnet152']:
            config.model.backbone_name=backbone
            config.args.n_epoch=100
            config.args.note='_'.join([config.args.note,backbone])
            net=pspnet(config)
            net.do_train_or_val(config.args,train_loader,val_loader)
    else:
        assert False