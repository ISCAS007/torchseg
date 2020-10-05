# -*- coding: utf-8 -*-

"""
MSRA10K(formally named as THUS10000; 195MB: images + binary masks):
    Pixel accurate salient object labeling for 10000 images from MSRA dataset.
    Please cite our paper [https://mmcheng.net/SalObj/] if you use it.
"""

import torch.utils.data as td
import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from PIL import Image


class MSRA10K_Dataset(td.Dataset):
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        pass
    
    def __getitem__(self,index):
        pass