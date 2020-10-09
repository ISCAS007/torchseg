# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import os
from models.motionseg.motion_utils import get_dataset,get_model
from utils.configs.motionseg_config import get_default_config
from utils.configs.semanticseg_config import load_config
from dataset.motionseg_dataset_factory import motionseg_show_images
from dataset.motionseg_dataset_factory import prepare_input_output
import torch.utils.data as td
import unittest
import cv2
from tqdm import trange
import numpy as np
import torch

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
        
    def test_motion_diff(self):
        """
        load model and show model output
        """
        #config_txt=os.path.expanduser('~/tmp/logs/motion/motion_diff/cdnet2014/test/2020-09-25___19-16-21/config.txt')
        config_txt=os.path.expanduser('~/tmp/logs/motion/motion_diff/FBMS/test/2020-09-25___18-52-18/config.txt')
        config=load_config(config_txt)
        
        model=get_model(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        for split in ['train','val']:
            xxx_loader=get_dataset(config,split)
            dataset_loader=td.DataLoader(dataset=xxx_loader,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=2)
            for data in dataset_loader:
            
                images,origin_labels,resize_labels=prepare_input_output(data=data,device=device,config=config)
                motionseg_show_images(images,origin_labels,[])
                
                outputs=model(frames)
                predict=outputs['masks'][0]
                motionseg_show_images([],[],predict)
                break
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
