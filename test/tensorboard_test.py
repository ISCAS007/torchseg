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
    config.args.log_dir = os.path.expanduser('~/tmp/logs/pytorch')
    config.args.note = 'aug'
    
    return json.dumps(config,indent=2,sort_keys=True).replace('\n','\n\n').replace('  ','\t')

log_dir=os.path.expanduser('~/tmp/logs/pytorch/test/histgrames')
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
    writer.add_scalar('test/acc/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('test/miou/scalar2', dummy_s2[0], n_iter)
    
    class_iou={0: 0.02561840174824816, 1: 0.02539294979398749, 2: 0.02540968514085804, 3: 0.025540797740621738, 4: 0.025543059621013713, 5: 0.024963605421254434, 6: 0.02547022343914425, 7: 0.02561703598995251, 8: 0.02568687353761584, 9: 0.02691302038648527, 10: 0.02606248470511461, 11: 0.025379828962022212, 12: 0.025659944253156253, 13: 0.02601991319887669, 14: 0.025118687075386757, 15: 0.026091755495825115, 16: 0.02615292066437595, 17: 0.025671171815895476, 18: 0.025110617213465733, 19: 0.025298578490067853}
    
    writer.add_histogram(tag='class_iou',values=np.array(list(class_iou.values())),global_step=n_iter)
    
    class_iou_dict={}
    for k,v in class_iou.items():
        class_iou_dict[str(k)]=v
    writer.add_scalars('test/class_iou', class_iou_dict, n_iter)

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