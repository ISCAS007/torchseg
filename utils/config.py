# -*- coding: utf-8 -*-

from easydict import EasyDict as edict
import random
import json
import yaml
import os

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

def dump_config(config,log_dir,filename='config.txt'):
    os.makedirs(log_dir,exists_ok=True)
    config_path=os.path.join(log_dir,filename)
    config_file=open(config_path,'w')
    json.dump(config,config_file,sort_keys=True)
    
def load_config(config_file):
    f=open(config_file,'r')
    l=f.readline()
    f.close()

    d=yaml.load(l)
    config=edict(d)

    return config