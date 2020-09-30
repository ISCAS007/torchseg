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
import os

def get_fgdet_dataset(config,split):
    dataset_dict={"fbms":fbms_dataset,
                  "cdnet2014":cdnet_dataset,
                  "segtrackv2":segtrackv2_dataset,
                  "bmcnet":bmcnet_dataset,
                  "davis2016":davis_dataset,
                  'davis2017':davis_dataset}

    dataset_root_dict={"fbms":os.path.expanduser('~/cvdataset/FBMS'),
                       "cdnet2014":os.path.expanduser('~/cvdataset/cdnet2014'),
                       "segtrackv2":os.path.expanduser('~/cvdataset/SegTrackv2'),
                       "bmcnet":os.path.expanduser('~/cvdataset/BMCnet'),
                       "davis2016":os.path.expanduser('~/cvdataset/DAVIS'),
                       "davis2017":os.path.expanduser('~/cvdataset/DAVIS')}

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
            dataset_set.append(get_fgdet_dataset(config,split))
        xxx_dataset=td.ConcatDataset(dataset_set)
    else:
        assert False,'dataset must in {} or all'.format(dataset_dict.keys())

    return xxx_dataset