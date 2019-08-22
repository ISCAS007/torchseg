# -*- coding: utf-8 -*-
"""
use flownet to generate optical flow for main frame
suppose the frame gap is 5
use dataset cdnet2014, fbms, segtrack and bmcnet

## flownet
- https://github.com/NVIDIA/flownet2-pytorch
    - run in docker, need add demo code
- https://github.com/sniklaus/pytorch-liteflownet
    - python run.py --model default --first ./images/first.png --second ./images/second.png --out ./out.flo
"""

from easydict import EasyDict as edict
from models.motionseg.motion_utils import get_dataset
from dataset.segtrackv2_dataset import main2flow
from tqdm import trange
import os
import cv2

if __name__ == '__main__':
    config=edict()
    config.dataset='cdnet2014'
    config.frame_gap=5
    config.input_shape=[224,224]
    config.use_part_number=1000
    config.ignore_outOfRoi=True
    config.net_name='motion_panet2'
    config.share_backbone=False

    flow_execute_dir=os.path.expanduser('~/git/gnu/pytorch-liteflownet')
    os.chdir(flow_execute_dir)

    for dataset in ['cdnet2014','FBMS','segtrackv2','BMCnet']:
        if dataset == 'cdnet2014':
            continue
        for split in ['train','val']:
            config.dataset=dataset
            xxx_dataset=get_dataset(config,split)

            n=len(xxx_dataset)
            for idx in trange(n):
                main_path,aux_path,gt_path=xxx_dataset.__get_path__(idx)
                out_path=main2flow(main_path)
                dir_out_path=os.path.dirname(out_path)
                os.makedirs(dir_out_path,exist_ok=True)
                cmd='python run.py --model default --first {} --second {} --out {}'.format(aux_path,main_path,out_path)

                if not os.path.exists(out_path):
                    img1=cv2.imread(aux_path)
                    img2=cv2.imread(main_path)
                    if min(img1.shape)>0 and min(img2.shape)>0:
                        os.system(cmd)
                    else:
                        print(dataset,aux_path,main_path)
                    print(dataset,out_path)