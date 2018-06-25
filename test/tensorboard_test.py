# -*- coding: utf-8 -*-

import torch
import numpy as np
from tensorboardX import SummaryWriter
import os

log_dir='/home/yzbx/tmp/logs/pytorch/test'
os.makedirs(name=log_dir,exist_ok=True)
writer = SummaryWriter(log_dir)

for n_iter in range(100):

    dummy_s1 = torch.rand(1)
    dummy_s2 = torch.rand(1)
    # data grouping by `slash`
    writer.add_scalar('test/scalar1', dummy_s1[0], n_iter)
    writer.add_scalar('test/scalar2', dummy_s2[0], n_iter)

    writer.add_scalars('test/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
                                             'xcosx': n_iter * np.cos(n_iter),
                                             'arctanx': np.arctan(n_iter)}, n_iter)
    
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
        
    
writer.close()