# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import os
from models.motionseg.motion_utils import get_dataset,get_default_config
import unittest
import cv2
from tqdm import trange
import numpy as np

class Test(unittest.TestCase):
    config=get_default_config()
    config.use_part_number=0

    def test_dataset(self):
        def test_img(p):
            try:
                img=cv2.imread(p)
            except Exception as e:
                print(dataset,p,e)
            else:
                self.assertIsNotNone(img,p)

        # 'FBMS','cdnet2014','segtrackv2', 'BMCnet'
        for dataset in ['DAVIS2016']:
            self.config.dataset=dataset
            for split in ['train','val']:
                xxx_dataset=get_dataset(self.config,split)
#                N=min(10,len(xxx_dataset))
                N=len(xxx_dataset)
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

        # 'FBMS','cdnet2014','segtrackv2', 'BMCnet', 'DAVIS2016', 'DAVIS2017'
        for dataset in ['DAVIS2016','DAVIS2017']:
            self.config.dataset=dataset
            for split in ['train','val']:
                pixels=set()
                xxx_dataset=get_dataset(self.config,split)
#                N=min(100,len(xxx_dataset))
                N=len(xxx_dataset)
                for i in trange(N):
                    main,aux,gt=xxx_dataset.__get_path__(i)
                    if isinstance(gt,str):
                        pixels=test_img(gt,pixels)
                    else:
                        for x in gt:
                            pixels=test_img(x,pixels)

                print(dataset,split,pixels)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()