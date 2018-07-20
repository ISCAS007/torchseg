# -*- coding: utf-8 -*-

from easydict import EasyDict as edict
import random

def get_config():
    config=edict()
    config.model=edict()
    config.model.class_number=20
    config.model.backbone_name='vgg16'
    config.model.layer_preference='last'
    config.model.input_shape=(224,224)
    
    config.dataset=edict()
    config.dataset.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split=random.choice(['test','val','train'])
    config.dataset.resize_shape=(224,224)
    config.dataset.name='cityscapes'
    
    config.training=edict()
    config.training.n_epoch=300
    config.training.batch_size=4
    config.training.log_dir='/home/yzbx/tmp/logs/pytorch'
    config.training.note='default'
    
    return config

def get_share_config(config):
    return config