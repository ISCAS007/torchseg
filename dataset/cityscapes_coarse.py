# -*- coding: utf-8 -*-
"""
get image path from txt file, then do it!
➜  ADEChallengeData2016 tree -L 2 .
.
├── annotations
│   ├── training
│   ├── train.txt
│   ├── validation
│   └── val.txt
├── images
│   ├── training
│   ├── train.txt
│   ├── validation
│   └── val.txt
├── objectInfo150.txt
└── sceneCategories.txt

"""
import torch
import torch.utils.data as TD
import os
import glob
import cv2
import numpy as np
from easydict import EasyDict as edict
import random
import matplotlib.pyplot as plt

from utils.augmentor import Augmentations

def get_dataset_generalize_config(config,dataset_name):
    support_datasets=['ADEChallengeData2016','ADE20K_2016_07_26','VOC2007','VOC2012','Kitti2015','Cityscapes']
    assert dataset_name in support_datasets,'unknown dataset %s, not in support dataset %s'%(dataset_name,str(support_datasets))
    if dataset_name=='ADEChallengeData2016':
        # train + val, no test
        config.root_path='/media/sdb/CVDataset/ObjectSegmentation/ADEChallengeData2016'
        config.image_txt_path=os.path.join(config.root_path,'images')
        config.annotation_txt_path=os.path.join(config.root_path,'annotations')
        config.foreground_class_ids=[i for i in range(1,151)]
        config.background_class_ids=[0]
        config.ignore_index=0
        
        return config
    else:
        assert False,'Not Implement for dataset %s'%dataset_name
    
class dataset_generalize(TD.Dataset):
    def __init__(self, config, augmentations=None, split=None, bchw=True):
        self.config=config
        self.augmentations=augmentations
        self.bchw=bchw
        self.split=split
            
        assert self.split in ['train','val','test'],'unexcepted split %s for dataset, must be one of [train,val,test]'%self.split
        
        if hasattr(self.config,'txt_path'):
            txt_file=os.path.join(config.txt_path,self.split+'.txt')
            self.image_files,self.annotation_files = self.get_files_from_txt(txt_file,self.config.root_path)
            assert len(self.image_files)>0, 'No files found in %s with %s'%(self.config.root_path,txt_file)
            assert len(self.annotation_files)>0, 'No files found in %s with %s'%(self.config.root_path,txt_file)
        else:
            assert hasattr(self.config,'image_txt_path'),'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            assert hasattr(self.config,'annotation_txt_path'),'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            image_txt_file=os.path.join(config.image_txt_path,self.split+'.txt')
            annotation_txt_file=os.path.join(config.annotation_txt_path,self.split+'.txt')
            self.image_files=self.get_files_from_txt(image_txt_file,self.config.root_path)
            self.annotation_files=self.get_files_from_txt(annotation_txt_file,self.config.root_path)
            assert len(self.image_files)>0, 'No files found in %s with %s'%(self.config.root_path,image_txt_file)
            assert len(self.annotation_files)>0, 'No files found in %s with %s'%(self.config.root_path,annotation_txt_file)
        
        
        self.foreground_class_ids=self.config.foreground_class_ids
        self.n_classes = len(self.foreground_class_ids)+1
        self.background_class_ids=self.config.background_class_ids
        if hasattr(self.config,'ignore_index'):
            self.ignore_index=self.config.ignore_index
        else:
            self.ignore_index = 0
        
        if hasattr(self.config,'norm'):
            self.norm=True
        else:
            self.norm=False
        
        print("Found %d image files, %d annotation files" % (len(self.image_files),len(self.annotation_files)))
        assert len(self.image_files)==len(self.annotation_files)
    
    @staticmethod
    def get_files_from_txt(txt_file,root_path):
        with open(txt_file,'r') as f:
            files=[i.strip() for i in f.readlines()]
            if ' ' in files[0]:
                image_files=[]
                annotation_files=[]
                for line in files:
                    strs=line.split(' ')
                    image_files.append(os.path.join(root_path,strs[0]))
                    annotation_files.append(os.path.join(root_path,strs[1]))
                
                return image_files,annotation_files
            else:
                files = [os.path.join(root_path,file) for file in files]
                return files
    def __len__(self):
        """__len__"""
        return len(self.image_files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # eg root_path/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png
        img_path = self.image_files[index]
        # eg root_path/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png
        lbl_path = self.annotation_files[index]
        
        if hasattr(self.config,'print_path'):
            if self.config.print_path:
                print('image path:',img_path)
                print('label path:',lbl_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        lbl = cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE)
        
        ann = np.zeros_like(lbl)
        
#        lbl_ids = np.unique(lbl)
#        print('label image ids',lbl_ids)
        if self.ignore_index==0:
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx+1
        else:
            assert self.ignore_index not in self.foreground_class_ids,'ignore_index cannot in foregournd_class_ids if not 0'
            assert self.ignore_index >= self.n_classes,'ignore_index %d cannot less than class number %d'%(self.ignore_index,self.n_classes)
            for class_id in self.background_class_ids:
                ann[lbl == class_id] = self.ignore_index
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx
        
        if self.augmentations is not None and self.split=='train':
            img = self.augmentations.transform(img)
            img, ann = self.augmentations.transform(img, ann)
#            print('augmentation',img.shape,ann.shape)
            
            assert hasattr(self.config,'resize_shape'),'augmentations may change image to random size by random crop'
            
        if hasattr(self.config,'resize_shape'):
            assert len(self.config.resize_shape)==2, 'resize_shape should with len of 2 but %d'%len(self.config.resize_shape)
            img=cv2.resize(src=img,dsize=tuple(self.config.resize_shape),interpolation=cv2.INTER_LINEAR)
            ann=cv2.resize(src=ann,dsize=tuple(self.config.resize_shape),interpolation=cv2.INTER_NEAREST)
        
        if self.norm:
            img=2*img/255-1.0
        
        if self.bchw:
            # convert image from (height,width,channel) to (channel,height,width)
            img=img.transpose((2,0,1))
            
        if hasattr(self.config,'with_edge'):
            if self.config.with_edge:
                edge=self.get_edge(ann_img=ann,edge_width=self.config.edge_width)
                return img,ann,edge
        if hasattr(self.config,'with_path'):
            return {'image':(img,ann),'filename':(img_path,lbl_path)}
        
        return img, ann
    
    @staticmethod
    def get_edge(ann_img,edge_width=5):
        kernel = np.ones((edge_width,edge_width),np.uint8)
        ann_edge=cv2.Canny(ann_img,0,1)
        ann_dilation=cv2.dilate(ann_edge,kernel,iterations=1)
        ann_dilation=(ann_dilation>0).astype(np.uint8)
        return ann_dilation
            
            
if __name__ == '__main__':
    config=edict()
    dataset_name='ADEChallengeData2016'
    config=get_dataset_generalize_config(config,dataset_name)
    
    dataset=dataset_generalize(config,split='train')
    for idx,image_file in enumerate(dataset.image_files):
        print(idx,image_file)
        img=cv2.imread(image_file,cv2.IMREAD_COLOR)
        plt.imshow(img)
        plt.show()
        if idx>5:
            break
    for idx,annotation_file in enumerate(dataset.annotation_files):
#        print(idx,annotation_file)
        ann=cv2.imread(annotation_file,cv2.IMREAD_GRAYSCALE)
        plt.imshow(ann)
        plt.show()
        print(np.unique(ann))
        if idx>5:
            break
    