# -*- coding: utf-8 -*-
"""
use flownet structure to predict foreground area
"""
import os
import torch
import torch.nn as nn
from models.motionseg.motionseg_base_class import motion
from models.motionseg.motion_backbone import conv_bn_relu
from models.motionseg.motion_unet import get_decoder

class motion_flownet(motion):
    def __init__(self,config):
        super().__init__(config)

    def forward(self,imgs):
        pass
