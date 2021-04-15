#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 08:36:45 2021

@author: yzbx

online training for davis dataset
"""

import argparse
from torchseg.dataset.davis_dataset  import davis_dataset
import torch.utils.data as td
from torchseg.dataset.dataset_generalize import image_normalizations
from torchseg.utils.augmentor import Augmentations
from torchseg.utils.configs.motionseg_config import load_config
from torchseg.models.motionseg.motion_utils import get_load_convert_model
import os
from pprint import pprint
import glob
import torch
from torchseg.utils.losses import jaccard_loss,dice_loss
from tools.fbms_train import train

def get_loaders(config,category):
    normer=image_normalizations(ways='-1,1')
    augmentations = Augmentations()

    dataset_loaders={}
    for split in ['train','val']:
        xxx_dataset=davis_dataset(config,
                                  normalizations=normer,
                                  augmentations=augmentations,
                                  split=split,
                                  category=category)

        if split=='train':
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=config.batch_size,shuffle=True,num_workers=2*config.batch_size,pin_memory=False)
        else:
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=False)
        dataset_loaders[split]=xxx_loader

    return dataset_loaders

def get_loss_fn(config):
    if config.net_name == 'motion_diff' or not config.net_name.startswith('motion'):
        seg_loss_fn=torch.nn.BCEWithLogitsLoss()
    elif config.loss_name in ['iou','dice']:
        # iou loss not support ignore_index
        assert config.dataset not in ['cdnet2014','all','all2','all3']
        assert config.ignore_pad_area==0
        if config.loss_name=='iou':
            seg_loss_fn=jaccard_loss
        else:
            seg_loss_fn=dice_loss
    else:
        seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=255)

    return seg_loss_fn

def get_optimizer(config,model):
    # only fine tune decoder, not training encoder and midnet.
    for name,param in model.named_parameters():
        if not name.startswith('decoder'):
            param.requires_grad_(False)

    optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]

    if config.optimizer=='adam':
        optimizer = torch.optim.Adam(
                    optimizer_params, lr=config['init_lr'], amsgrad=False)
    else:
        optimizer = torch.optim.SGD(
                    optimizer_params, lr=config['init_lr'], momentum=0.9, weight_decay=1e-4)

    return optimizer

def online_finetune(config,category):
    model=get_load_convert_model(config)
    loaders=get_loaders(config,category)

    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_fn=get_loss_fn(config)
    optimizer=get_optimizer(config,model)

    if hasattr(config,'origin_note'):
        pass
    else:
        config.origin_note=config.note

    config.note=config.origin_note+'_{}'.format(category)
    train(config,model,loss_fn,optimizer,loaders)

def get_category_list(config,split='val'):
    if config.dataset.upper()=='DAVIS2017':
        year=2017
    elif config.dataset.upper()=='DAVIS2016':
        year=2016
    else:
        assert False
    txt_file=os.path.join(config.root_path,
                          'ImageSets',
                          str(year),
                          split+'.txt'
                          )

    f = open(txt_file, 'r')
    lines = f.readlines()
    f.close()

    return [line.strip() for line in lines]

if __name__ == '__main__':
    parser=argparse.ArgumentParser('online training')

    parser.add_argument('--config_txt_path',
                        help='the config txt path')

    parser.add_argument('--note',
                        help='the new note for online training',
                        default=None)

    parser.add_argument('--init_lr',
                        type=float,
                        default=None,
                        help='learning rate')

    parser.add_argument('--use_part_number',
                        type=int,
                        default=100,
                        help='the fine tune times')

    args=parser.parse_args()


    config=load_config(args.config_txt_path)

    val_category_list=get_category_list(config,'val')

    log_dir=os.path.dirname(args.config_txt_path)
    checkpoint_path_list=glob.glob(os.path.join(log_dir,'*.pkl'))
    assert len(checkpoint_path_list)>0,f'{log_dir} do not have checkpoint'
    if len(checkpoint_path_list) > 1:
        pprint(checkpoint_path_list)
    config.checkpoint_path=checkpoint_path_list[0]
    if args.note is not None:
        config.note=args.note

    if args.init_lr is None:
        config.init_lr=config.init_lr/10
    else:
        config.init_lr=args.init_lr

    config.use_part_number=args.use_part_number
    config.epoch=1
    config.save_model=False
    for category in val_category_list:
        online_finetune(config,category)