# -*- coding: utf-8 -*-

import torch
import numpy as np
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from dataset.dataset_generalize import image_normalizations
import os
import random
import json
import cv2
from tqdm import trange

def get_text():
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
    
    return json.dumps(config,indent=2,sort_keys=True).replace('\n','\n\n').replace('  ','\t')

log_dir='/home/yzbx/tmp/logs/pytorch/test/norm_uint8'
os.makedirs(name=log_dir,exist_ok=True)
writer = SummaryWriter(log_dir)

writer.add_text(tag='config',text_string=get_text())
normalizations=image_normalizations('caffe')
image_bgr=cv2.imread('test/image.png')
image_rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
image_rgb_forward=normalizations.forward(image_rgb)
image_diff_forward=cv2.absdiff(image_rgb_forward.astype(np.uint8),image_rgb)
image_rgb_backward=normalizations.backward(image_rgb_forward)
for n_iter in trange(30):

    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('test/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('test/scalar2', dummy_s2[0], n_iter)

#    writer.add_scalars('test/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
#                                             'xcosx': n_iter * np.cos(n_iter),
#                                             'arctanx': np.arctan(n_iter)}, n_iter)
    
    if (n_iter+1)%10==0:
        image_a = torch.rand(3,100,100)
#        image_b = torch.rand(100,100,3)
        image_c = torch.rand(1,100,100)
#        image_d = torch.rand(100,100,1)
        image_e = torch.rand(100,100)
        
        image_f = np.random.rand(100,100,3)
#        image_g = np.random.rand(3,100,100)
#        image_h = np.random.rand(100,100)
        image_i = torch.rand(3,100,100)>image_a
        image_j = np.random.rand(100,100,3)>image_f
        image_k = (np.random.rand(100,100)>np.random.rand(100,100)).astype(np.uint8)
        image_k = torch.from_numpy(image_k)
        writer.add_image('test/image_a',image_a,global_step=n_iter)
#        writer.add_image('test/image_b',image_b,global_step=n_iter)
        writer.add_image('test/image_c',image_c,global_step=n_iter)
#        writer.add_image('test/image_d',image_d,global_step=n_iter)
        writer.add_image('test/image_e',image_e,global_step=n_iter)
        writer.add_image('test/image_f',image_f,global_step=n_iter)
#        writer.add_image('test/image_g',image_g,global_step=n_iter)
#        writer.add_image('test/image_h',image_h,global_step=n_iter)
        writer.add_image('test/image_i',image_i,global_step=n_iter)
        writer.add_image('test/image_j',image_j,global_step=n_iter)
        writer.add_image('test/image_k',image_k,global_step=n_iter)
        
        # rgb is the right way for tensorboard
        writer.add_image('test/image_rgb',image_rgb,global_step=n_iter)
        writer.add_image('test/image_bgr',image_bgr,global_step=n_iter)
        writer.add_image('test/image_rgb_forward',image_rgb_forward.astype(np.uint8),global_step=n_iter)
        writer.add_image('test/image_rgb_backward',image_rgb_backward.astype(np.uint8),global_step=n_iter)
        
writer.close()