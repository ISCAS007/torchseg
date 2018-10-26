# -*- coding: utf-8 -*-

import torch
from torch.nn import Module,Conv2d
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as TD
import random
from dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import os
import cv2

config=edict()
config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
#config.cityscapes_split=random.choice(['test','val','train'])
config.resize_shape=(224,224)
config.print_path=False
config.with_path=False
config.with_edge=False
#config=get_dataset_generalize_config(config,'Cityscapes')
config=get_dataset_generalize_config(config,'VOC2012')


split='test'
dataset=dataset_generalize(config,split=split)
loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=False,drop_last=False)

save_root='/media/sdc/yzbx/benchmark_output/cityscapes'
os.makedirs(save_root,exist_ok=True)

for i, data in enumerate(loader):
    imgs=data['image']
    paths=data['filename']
    if split=='test':
        print(paths[0])
        print(imgs[0].shape)
    else:
        imgs, labels = imgs
        img_path,lbl_path = paths
        print(img_path,lbl_path)