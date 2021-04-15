#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:06:19 2021

@author: yzbx

statistic the log from online training
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
