# -*- coding: utf-8 -*-

import torch
import torch.utils.data as TD
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os
import argparse

from dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from models.psp_caffe import psp_caffe
from utils.augmentor import Augmentations
from utils.torch_tools import keras_fit
from utils.disc_tools import str2bool

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--net_name',
                        help='net name for semantic segmentaion',
                        choices=['pspnet','psp_edge','psp_global','psp_caffe','psp_fractal','psp_dict'],
                        default='psp_caffe')
    
    parser.add_argument("--backbone_name",
                        help="backbone name",
                        choices=['vgg16','vgg19','vgg16_bn','vgg19_bn','resnet18','resnet34','resnet50','resnet101','resnet152'],
                        default='resnet50')
                        
    parser.add_argument("--midnet_name",
                        help="midnet name",
                        choices=['psp','aspp'],
                        default='psp')
    
    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['ADEChallengeData2016','VOC2012','Kitti2015','Cityscapes','Cityscapes_Fine','Cityscapes_Coarse'],
                        default='Cityscapes')
    
    # train_extra is only available for Cityscapes_Coarse
    splits=['train','train_extra']
    parser.add_argument('--train_split',
                        help='split for dataset: %s'%str(splits),
                        choices=splits,
                        default='train')
    
    parser.add_argument("--batch_size",
                        help='batch size',
                        type=int,
                        default=2)
    
    parser.add_argument('--input_shape',
                        help='input shape',
                        type=int,
                        default=0)
    
    parser.add_argument('--midnet_scale',
                        help='pspnet scale',
                        type=int,
                        default=5)
    
    parser.add_argument('--n_epoch',
                        help='training/validating epoch',
                        type=int,
                        default=100)
    
    parser.add_argument('--augmentation',
                        help='true or false to do augmentation',
                        type=str2bool,
                        default=True)
    
    parser.add_argument('--upsample_type',
                        help='bilinear or duc upsample',
                        choices=['duc','bilinear'],
                        default='duc')
                        
    parser.add_argument('--upsample_layer',
                        help='layer number for upsample',
                        type=int,
                        choices=[1,2,3,4,5],
                        default=5)
    
    parser.add_argument('--note',
                        help='comment for tensorboard log',
                        default=None)
    
    args = parser.parse_args()
    config=edict()
    config.model=edict()
    config.model.upsample_type=args.upsample_type
    config.model.upsample_layer=args.upsample_layer
    config.model.backbone_name=args.backbone_name
    config.model.layer_preference='last'
    config.model.midnet_pool_sizes=[6,3,2,1]
    config.model.midnet_scale=args.midnet_scale
    #TODO 2048 or counted???
    resnet_out_channels=[128,256,512,1024,2048]
    config.model.midnet_in_channels=resnet_out_channels[args.upsample_layer-1]
    config.model.midnet_out_channels=512
    config.model.suffix_out_channels=512
    config.model.midnet_name=args.midnet_name
    config.model.momentum=0.95
    config.model.inplace=False
    config.model.align_corners=True
        
    if args.input_shape==0:
        if args.midnet_name=='psp':
            count_size=max(config.model.midnet_pool_sizes)*config.model.midnet_scale
            input_shape=(count_size,count_size)
        else:
            input_shape=(72*8,72*8)
    else:
        input_shape=(args.input_shape,args.input_shape)
        
    config.model.input_shape=input_shape
    
    config.dataset=edict()
    config.dataset=get_dataset_generalize_config(config.dataset,args.dataset_name)
    if config.dataset.ignore_index == 0:
        config.model.class_number=len(config.dataset.foreground_class_ids)+1
    else:
        config.model.class_number=len(config.dataset.foreground_class_ids)
    config.dataset.resize_shape=input_shape
    config.dataset.name=args.dataset_name.lower()
    config.dataset.norm=True
    
    config.args=edict()
    config.args.n_epoch=args.n_epoch
    config.args.log_dir='/home/yzbx/tmp/logs/pytorch'
    
    if args.note is None:
        config.args.note='_'.join([args.backbone_name,
                                   'bn%d'%args.batch_size,
                                   'aug'+str(args.augmentation)[0]])
    else:
        config.args.note='_'.join([args.note,
                                   args.backbone_name,
                                   'bn%d'%args.batch_size,
                                   'aug'+str(args.augmentation)[0]])
    
    if args.train_split=='train_extra':
        config.args.note+='_train_extra'
    
    if args.augmentation:
        augmentations=Augmentations(p=0.25)
    else:
        augmentations=None
    
    train_dataset=dataset_generalize(config.dataset,split=args.train_split,augmentations=augmentations)
    train_loader=TD.DataLoader(dataset=train_dataset,batch_size=args.batch_size, shuffle=True,drop_last=True,num_workers=8)
    
    val_dataset=dataset_generalize(config.dataset,split='val',augmentations=augmentations)
    val_loader=TD.DataLoader(dataset=val_dataset,batch_size=args.batch_size, shuffle=True,drop_last=False,num_workers=8)
    
    net=globals()[args.net_name](config)
    keras_fit(net,train_loader,val_loader)
    