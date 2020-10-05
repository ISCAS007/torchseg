# -*- coding: utf-8 -*-
"""
design patter: factory
"""

from dataset.fbms_dataset import fbms_dataset
from dataset.cdnet_dataset import cdnet_dataset
from dataset.segtrackv2_dataset import segtrackv2_dataset
from dataset.bmcnet_dataset import bmcnet_dataset
from dataset.davis_dataset import davis_dataset
from dataset.dataset_generalize import image_normalizations
from utils.augmentor import Augmentations
import torch.utils.data as td
from utils.configs.motionseg_config import get_default_config, dataset_root_dict
import numpy as np
from utils.disc_tools import show_images

def get_motionseg_dataset(config,split):
    dataset_dict={"fbms":fbms_dataset,
                  "cdnet2014":cdnet_dataset,
                  "segtrackv2":segtrackv2_dataset,
                  "bmcnet":bmcnet_dataset,
                  "davis2016":davis_dataset,
                  'davis2017':davis_dataset}

    normer=image_normalizations(ways='-1,1')
    augmentations = Augmentations()
    key=config.dataset.lower()
    if key in dataset_dict.keys():
        config.root_path=dataset_root_dict[key]
        xxx_dataset=dataset_dict[key](config,split,normalizations=normer,augmentations=augmentations)
    elif key == "all":
        dataset_set=[]
        for d in ['FBMS','cdnet2014','segtrackv2','BMCnet','DAVIS2017','DAVIS2016']:
            config.dataset=d
            dataset_set.append(get_motionseg_dataset(config,split))
        xxx_dataset=td.ConcatDataset(dataset_set)
    else:
        assert False,'dataset must in {} or all'.format(dataset_dict.keys())

    return xxx_dataset

if __name__ == '__main__':
    config=get_default_config()
    keys=dataset_root_dict.keys()
    for key in keys:
        for split in ['train','val']:
            config.dataset=key
            d=get_motionseg_dataset(config,split)
            N=len(d)
            idx=np.random.randint(N)
            data=d.__getitem__(idx)
            imgs=[img.transpose((1,2,0)) for img in data['images']]
            gts=data['labels']
            
            print(imgs[0].shape, gts[0].shape)
            gts=[gt.squeeze() for gt in gts]
            show_images(imgs+gts)
            print('dataset={}, split={}'.format(key,split) + '*'*30)
            print(data['main_path'],data['aux_path'],data['gt_path'])
            for img in imgs:
                print('img range in [{},{}]'.format(np.min(img),np.max(img)))
                
            for gt in gts:
                print('gt range in {}'.format(np.unique(gt)))