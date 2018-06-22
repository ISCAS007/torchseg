# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as TD
import random
from dataset.cityscapes import cityscapes
from models.backbone import backbone
from utils.metrics import runningScore,get_scores
from utils.torch_tools import freeze_layer
from models.upsample import upsample_duc,upsample_bilinear
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os

class pspnet(TN.Module):
    def __init__(self,config):
        super(pspnet,self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config.model)
        
        if hasattr(self.config.model,'backbone_lr_ratio'):
            backbone_lr_raio=self.config.model.backbone_lr_ratio
            if backbone_lr_raio==0:
                freeze_layer(self.backbone)
        
        self.upsample_type = self.config.model.upsample_type
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name