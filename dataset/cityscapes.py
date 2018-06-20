# -*- coding: utf-8 -*-
import torch
import torch.utils.data as TD
import os
import glob
import cv2
import numpy as np
from easydict import EasyDict as edict
import random

class cityscapes(TD.Dataset):
    def __init__(self, config, augmentations=None, split=None):
        """
        root_path
        ├── gtFine_trainvaltest
        │   └── gtFine
        │       ├── test
        │       ├── train
        │       └── val
        └── leftImg8bit_trainvaltest
            ├── gtFine_trainvaltest -> ../gtFine_trainvaltest
            └── leftImg8bit
                ├── test
                ├── train
                └── val

        """
        self.config=config
        self.augmentations=augmentations
        if split is not None:
            self.split=split
        else:
            self.split=self.config.cityscapes_split
        
        assert self.split in ['train','val','test'],'unexcepted split %s for cityscapes, must be one of [train,val,test]'%self.split

        self.images_base = os.path.join(self.config.root_path,'leftImg8bit_trainvaltest','leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.config.root_path, 'gtFine_trainvaltest', 'gtFine', self.split)
        
        self.image_files = glob.glob(os.path.join(self.images_base,'*','*.png'))
        self.annotation_files = glob.glob(os.path.join(self.annotations_base,'*','*.png'))
        assert len(self.image_files)>0, 'No files found in %s'%(self.images_base)
        assert len(self.annotation_files)>0, 'No files found in %s'%(self.annotations_base)
        
        self.foreground_class_ids=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.n_classes = len(self.foreground_class_ids)+1
        self.background_class_ids=[0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.ignore_index = 0
        
        
        print("Found %d image files, %d annotation files" % (len(self.image_files),len(self.annotation_files)))
        
    def __len__(self):
        """__len__"""
        return len(self.image_files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        # eg root_path/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png
        img_path = self.image_files[index]
        # eg root_path/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_color.png
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2], 
                                os.path.basename(img_path).replace('leftImg8bit.png','gtFine_color.png'))
#        lbl_path = self.annotation_files[index]
        
        img_path_split=img_path.split(os.sep)
        lbl_path_split=lbl_path.split(os.sep)
        
        # assert the same split: eg test
        assert img_path_split[-3]==lbl_path_split[-3],'unmatched label path and image path: \n%s \n%s'%(img_path,lbl_path)
        # assert the same video: eg berlin
        assert img_path_split[-2]==lbl_path_split[-2],'unmatched label path and image path: \n%s \n%s'%(img_path,lbl_path)
        basename_img_path_split=img_path_split[-1].split('_')
        basename_lbl_path_split=lbl_path_split[-1].split('_')
        
        # the version of cityscapes format change with time
        if len(basename_img_path_split)==4:
            assert basename_img_path_split[0]==basename_lbl_path_split[0] and basename_img_path_split[1]==basename_lbl_path_split[1] and basename_img_path_split[2]==basename_lbl_path_split[2],'unmatched label path and image path: \n%s \n%s'%(img_path,lbl_path)
        
        if hasattr(self.config,'print_path'):
            if self.config.print_path:
                print(img_path)
                print(lbl_path)
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        lbl = cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE)
        
        ann = np.zeros_like(lbl)
        if self.ignore_index==0:
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx+1
        else:
            assert self.ignore_index not in self.foreground_class_ids,'ignore_index cannot in foregournd_class_ids if not 0'
            for class_id in self.background_class_ids:
                ann[lbl == class_id] = self.ignore_index
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx
        
        if hasattr(self.config,'resize_shape'):
            assert len(self.config.resize_shape)==2, 'resize_shape should with len of 2 but %d'%len(self.config.resize_shape)
            img=cv2.resize(src=img,dsize=tuple(self.config.resize_shape),interpolation=cv2.INTER_LINEAR)
            ann=cv2.resize(src=ann,dsize=tuple(self.config.resize_shape),interpolation=cv2.INTER_NEAREST)
        
        if self.augmentations is not None:
            img, ann = self.augmentations(img, ann)

        return img, ann
    
if __name__ == '__main__':
    config=edict()
    config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.cityscapes_split=random.choice(['test','val','train'])
    config.print_path=True
    
    dataset=cityscapes(config)
    loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=True,drop_last=False)
    for i, data in enumerate(loader):
        imgs, labels = data
        print(i,imgs.shape,labels.shape)
        
        if i>=3:
            break