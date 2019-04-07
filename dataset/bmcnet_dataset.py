# -*- coding: utf-8 -*-

import os
import glob
import torch.utils.data as td
import numpy as np
import random
import cv2
from dataset.segtrackv2_dataset import main2flow

class bmcnet_dataset(td.Dataset):
    """
    BMCnet
    ├── real
    │   ├── Video_001
    │   ├── Video_002
    │   ├── Video_003
    │   ├── Video_004
    │   ├── Video_005
    │   ├── Video_006
    │   ├── Video_007
    │   ├── Video_008
    │   └── Video_009
    ├── test
    │   ├── 112_png
    │   ├── 122_png
    │   ├── 212_png
    │   ├── 222_png
    │   ├── 312_png
    │   ├── 322_png
    │   ├── 412_png
    │   ├── 422_png
    │   ├── 512_png
    │   └── 522_png
    └── train
        ├── 111_png
        ├── 121_png
        ├── 211_png
        ├── 221_png
        ├── 311_png
        ├── 321_png
        ├── 411_png
        ├── 421_png
        ├── 511_png
        └── 521_png
    """
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        self.config=config
        self.split=split
        self.normalizations=normalizations
        self.augmentations=augmentations
        self.input_shape=tuple(config.input_shape)
        self.root_path=config.root_path
        self.main_files=self.get_main_files()
        print('dataset size = {}',len(self.main_files))
        n=len(self.main_files)
        if n > self.config['use_part_number'] > 0:
            gap=n//self.config['use_part_number']
            self.main_files=self.main_files[::gap]
            print('total dataset image %d, use %d'%(n,len(self.main_files)))
                
        self.frame_gap=config.frame_gap
        self.use_optical_flow=config.use_optical_flow
        
    def __len__(self):
        return len(self.main_files)
    
    def get_main_files(self):
        if self.split=='train':
            rootpath=os.path.join(self.root_path,'train')
        else:
            rootpath=os.path.join(self.root_path,'test')
        
        pattern=os.path.join(rootpath,'**','input','*.png')
        main_files=glob.glob(pattern,recursive=True)
        main_files.sort()
        
        valid_main_files=[f for f in main_files if os.path.exists(self.get_gt_file(f))]
        assert len(valid_main_files)>0,'rootpath={}'.format(rootpath)
        return valid_main_files
    
    def get_aux_file(self,main_file):
        dirname=os.path.dirname(main_file)
        pattern=os.path.join(dirname,'*.png')
        aux_files=glob.glob(pattern)
        aux_files.sort()
        assert len(aux_files)>0,'main_file={},pattern={}'.format(main_file,pattern)
        
        main_index=aux_files.index(main_file)
        aux_files.remove(main_file)
        n=len(aux_files)
        x=random.random()
        if x>0.5:
            aux_index=main_index+self.frame_gap
            aux_index=aux_index if aux_index<n else n-1
        else:
            aux_index=main_index-self.frame_gap
            aux_index=aux_index if aux_index>=0 else 0
        return aux_files[aux_index]
    
    def get_gt_file(self,main_file):
        if self.split=='train':
            gt_file=main_file.replace('input','truth')
        else:
            gt_file=main_file.replace('input','private_truth')
        return gt_file
    
    def __get_path__(self,index):
        main_file=self.main_files[index]
        aux_file=self.get_aux_file(main_file)
        gt_file=self.get_gt_file(main_file)
        
        return main_file,aux_file,gt_file
        
    def __getitem__(self,index):
        main_file=self.main_files[index]
        aux_file=self.get_aux_file(main_file)
        gt_file=self.get_gt_file(main_file)
        
        #print(main_file,aux_file,gt_files)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        gt_image=cv2.imread(gt_file,cv2.IMREAD_GRAYSCALE)
        
        # augmentation dataset
        if self.split=='train' and self.augmentations is not None:
            frame_images=[self.augmentations.transform(img) for img in frame_images]
            
        # resize image
        resize_frame_images=[cv2.resize(img,self.input_shape,interpolation=cv2.INTER_LINEAR) for img in frame_images]
        if self.split=='train':
            resize_gt_image=cv2.resize(gt_image,self.input_shape,interpolation=cv2.INTER_NEAREST)
        else:
            resize_gt_image=gt_image
        
        # normalize image
        if self.normalizations is not None:
            resize_frame_images = [self.normalizations.forward(img) for img in resize_frame_images]
            
        # bchw
        resize_frame_images=[img.transpose((2,0,1)) for img in resize_frame_images]
        resize_gt_image=np.expand_dims(resize_gt_image,0)
        
        resize_gt_image=(resize_gt_image!=0).astype(np.uint8)
    
        if self.use_optical_flow:
            flow_path=main2flow(main_file)
            flow_file=open(flow_path,'r')
            a=np.fromfile(flow_file,np.uint8,count=4)
            b=np.fromfile(flow_file,np.int32,count=2)
            flow=np.fromfile(flow_file,np.float32).reshape((b[1],b[0],2))
            flow=np.clip(flow,a_min=-50,a_max=50)/50.0
            optical_flow=cv2.resize(flow,self.input_shape,interpolation=cv2.INTER_LINEAR).transpose((2,0,1))
            return [resize_frame_images[0],optical_flow],resize_gt_image
        else:
            return resize_frame_images,resize_gt_image