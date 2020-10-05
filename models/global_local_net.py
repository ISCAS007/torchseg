# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from models.motionseg.motion_panet import motion_panet2
from utils.configs.motionseg_config import get_default_config

class GlobalLocalNet(nn.Module):
    """
    for semantic segmentation, we find the bigger input shape, the better miou.
    so we propose global local net, use local image and global image with input shape (h,w), to generate the
    segmentation result like input shape (2xh,2xw),(4xh,4xw),(8xh,8xw),...
    """
    def __init__(self,config):
        super().__init__()

        default_motionseg_config=get_default_config()
        for key in default_motionseg_config.keys():
            if key not in config.keys():
                print('add config[{}]={}'.format(key,default_motionseg_config[key]))
                config[key]=default_motionseg_config[key]

        self.config=config
        self.name=self.__class__.__name__
        self.origin_input_shape=config.origin_input_shape=config.input_shape
        config.input_shape=[x//2 for x in config.input_shape]

        self.base=motion_panet2(config)

    def forward(self,x):
        h,w=self.origin_input_shape
        assert h%4==0 and w%4 ==0
        x_local=x[:,:,h//4:(h*3)//4,w//4:(w*3)//4]
        x_global=F.interpolate(x, size=(h//2,w//2),
                          mode='bilinear', align_corners=True)
        return self.base([x_local,x_global])['masks'][0]
