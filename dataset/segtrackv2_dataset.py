# -*- coding: utf-8 -*-

import os
import glob
import torch.utils.data as td
import numpy as np
import random
import cv2

class segtrackv2_dataset(td.Dataset):
    """
    multi instances tracking dataset
    """
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        self.config=config
        self.split=split
        self.normalizations=normalizations
        self.augmentations=augmentations
        self.input_shape=tuple(config.input_shape)
        self.root_path=config.root_path
        self.main_files=self.get_main_files()
        self.frame_gap=config.frame_gap
        
    def __len__(self):
        return len(self.main_files)
    
    def get_gt_files(self,main_file):
        """
        GroundTruth
        ├── birdfall
        ├── bird_of_paradise
        ├── bmx
        │   ├── 1
        │   └── 2
        ...
        """
        path_strings=main_file.split(os.path.sep)
        basename=path_strings[-1]
        basename=basename.split('.')[0]
        video_name=path_strings[-2]
        
        gt_files=[]
        for suffix in ['bmp','png']:
            gt_path=os.path.join(self.root_path,'GroundTruth',video_name,'**',basename+'.'+suffix)
            gt_files+=glob.glob(gt_path,recursive=True)
        assert len(gt_files)>0,'gt_path={}'.format(gt_path)
        return gt_files
    
    def get_aux_file(self,main_file):
        path_strings=main_file.split(os.path.sep)
        basename=path_strings[-1]
        video_name=path_strings[-2]
        suffix=basename.split('.')[1]
        
        frame_path=os.path.join(self.root_path,'JPEGImages')
        video_path=os.path.join(frame_path,video_name,'*.'+suffix)
        video_files=glob.glob(video_path)
        video_files.sort()
        
        main_index=video_files.index(main_file)
        video_files.remove(main_file)
        n=len(video_files)
        assert n>0,'main_file={}'.format(main_file)
        assert main_index>=0,'main_file={}'.format(main_file)
        x=random.random()
        if x>0.5:
            aux_index=main_index+self.frame_gap
            aux_index=aux_index if aux_index<n else n-1
        else:
            aux_index=main_index-self.frame_gap
            aux_index=aux_index if aux_index>=0 else 0
            
        return video_files[aux_index]
        
    def get_main_files(self):
        frame_path=os.path.join(self.root_path,'JPEGImages')
        videos=[d for d in os.listdir(frame_path) if os.path.isdir(os.path.join(frame_path,d))]
        videos.sort()
        assert len(videos)>0,'frame_path={}'.format(frame_path)
        n=len(videos)
        train_size=int(n*0.7)
        if self.split=='train':
            videos=videos[:train_size]
        else:
            videos=videos[train_size:]
        
        frame_files=[]
        for v in videos:
            video_files=[]
            for suffix in ['bmp','png']:
                video_path=os.path.join(frame_path,v,'*.'+suffix)
                video_files+=glob.glob(video_path)
            assert len(video_files)>0,'video_path={}'.format(video_path)
            frame_files+=video_files
        
        frame_files.sort()
        assert len(frame_files)>0,'videos={}'.format(videos)
        return frame_files
    
    def __getitem__(self,index):
        main_file=self.main_files[index]
        aux_file=self.get_aux_file(main_file)
        gt_files=self.get_gt_files(main_file)
        
        #print(main_file,aux_file,gt_files)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        gt_images=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in gt_files]
        gt_image=np.zeros_like(gt_images[0])
        for gt in gt_images:
            gt_image+=gt
        
        # augmentation dataset
        if self.split=='train' and self.augmentations is not None:
            frame_images=[self.augmentations.transform(img) for img in frame_images]
            
        # resize image
        resize_frame_images=[cv2.resize(img,self.input_shape,interpolation=cv2.INTER_LINEAR) for img in frame_images]
        resize_gt_image=cv2.resize(gt_image,self.input_shape,interpolation=cv2.INTER_NEAREST)
        
        # normalize image
        if self.normalizations is not None:
            resize_frame_images = [self.normalizations.forward(img) for img in resize_frame_images]
            
        # bchw
        resize_frame_images=[img.transpose((2,0,1)) for img in resize_frame_images]
        resize_gt_image=np.expand_dims(resize_gt_image,0)
        
        resize_gt_image=(resize_gt_image!=0).astype(np.uint8)
        return resize_frame_images,resize_gt_image