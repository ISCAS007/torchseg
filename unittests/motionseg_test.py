# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import os
from dataset.fbms_dataset import fbms_dataset
from dataset.cdnet_dataset import cdnet_dataset
from dataset.segtrackv2_dataset import segtrackv2_dataset
from dataset.bmcnet_dataset import bmcnet_dataset
import unittest
import cv2
from tqdm import trange
import numpy as np

def get_dataset(config,split,normer=None,augmentations=None):
    if config.dataset=='FBMS':
        config['root_path']=os.path.expanduser('~/cvdataset/FBMS')
    elif config.dataset=='cdnet2014':
        config['root_path']=os.path.expanduser('~/cvdataset/cdnet2014')
    elif config.dataset=='segtrackv2':
        config['root_path']=os.path.expanduser('~/cvdataset/SegTrackv2')
    elif config.dataset=='BMCnet':
        config['root_path']=os.path.expanduser('~/cvdataset/BMCnet')
    else:
        assert False

    if config.dataset=='FBMS':
        xxx_dataset=fbms_dataset(config,split,normalizations=normer,augmentations=augmentations)
    elif config.dataset=='cdnet2014':
        xxx_dataset=cdnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
        print(xxx_dataset.train_set,xxx_dataset.val_set)
    elif config.dataset=='segtrackv2':
        xxx_dataset=segtrackv2_dataset(config,split,normalizations=normer,augmentations=augmentations)
    elif config.dataset=='BMCnet':
        xxx_dataset=bmcnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
    else:
        assert False

    return xxx_dataset

class Test(unittest.TestCase):
    config=edict()
    config.dataset='FBMS'
    config.frame_gap=0
    config.input_shape=(224,224)
    config.use_part_number=2000
    config.use_optical_flow=False
    def test_dataset(self):
        def test_img(p):
            try:
                img=cv2.imread(p)
            except Exception as e:
                print(dataset,p,e)
            else:
                self.assertIsNotNone(img,p)

        # 'FBMS','cdnet2014','segtrackv2',
        for dataset in ['BMCnet']:
            self.config.dataset=dataset
            for split in ['train','val']:
                xxx_dataset=get_dataset(self.config,split)
                N=min(10,len(xxx_dataset))
                for i in trange(N):
                    main,aux,gt=xxx_dataset.__get_path__(i)
                    for p in [main,aux,gt]:
                        if isinstance(p,str):
                            test_img(p)
                        else:
                            for x in p:
                                test_img(x)
    
    def test_ignore_pixel(self):
        def test_img(p,pixels):
            img=cv2.imread(p)
            new_pixels=np.unique(img)
            pixels=pixels.union(new_pixels)
            return pixels

        # 'FBMS','cdnet2014','segtrackv2',
        for dataset in ['BMCnet','FBMS','cdnet2014','segtrackv2']:
            self.config.dataset=dataset
            for split in ['train','val']:
                pixels=set()
                xxx_dataset=get_dataset(self.config,split)
                N=min(100,len(xxx_dataset))
                for i in trange(N):
                    main,aux,gt=xxx_dataset.__get_path__(i)
                    if isinstance(gt,str):
                        test_img(gt,pixels)
                    else:
                        for x in gt:
                            test_img(x,pixels)
                            
                print(dataset,split,pixels)

if __name__ == '__main__':
    unittest.main()