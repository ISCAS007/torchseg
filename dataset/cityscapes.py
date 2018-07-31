# -*- coding: utf-8 -*-
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

class cityscapes(TD.Dataset):
    """
    ➜  ls root_path/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_
    berlin_000000_000019_gtFine_color.png      
    berlin_000000_000019_gtFine_instanceIds.png
    berlin_000000_000019_gtFine_labelIds.png   
    berlin_000000_000019_gtFine_polygons.json  
    """
    def __init__(self, config, augmentations=None, split=None, bchw=True):
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
        self.bchw=bchw
        if split is not None:
            self.split=split
        else:
            self.split=self.config.cityscapes_split
        
        assert self.split in ['train','val','test'],'unexcepted split %s for cityscapes, must be one of [train,val,test]'%self.split

        self.images_base = os.path.join(self.config.root_path,'leftImg8bit_trainvaltest','leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.config.root_path, 'gtFine_trainvaltest', 'gtFine', self.split)
        
        self.image_files = glob.glob(os.path.join(self.images_base,'*','*leftImg8bit.png'))
        self.annotation_files = glob.glob(os.path.join(self.annotations_base,'*','*labelIds.png'))
        assert len(self.image_files)>0, 'No files found in %s'%(self.images_base)
        assert len(self.annotation_files)>0, 'No files found in %s'%(self.annotations_base)
        
        self.foreground_class_ids=[7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.n_classes = len(self.foreground_class_ids)+1
        self.background_class_ids=[0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        if hasattr(self.config,'ignore_index'):
            self.ignore_index=self.config.ignore_index
        else:
            self.ignore_index = 0
        
        if hasattr(self.config,'norm'):
            self.norm=True
        else:
            self.norm=False
        
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
        # eg root_path/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2], 
                                os.path.basename(img_path).replace('leftImg8bit.png','gtFine_labelIds.png'))
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
    
    def get_benchmarkable_predict(self,img):
        """
        convert the id to origin image
        """
        new_img=np.zeros_like(img)
        img_ids=np.unique(img)
        if self.ignore_index==0:
            for idx in img_ids:
                if idx != self.ignore_index:
                    new_img[img==idx]=self.foreground_class_ids[idx-1]
        else:
            for idx in img_ids:
                if idx != self.ignore_index:
                    new_img[img==idx]=self.foreground_class_ids[idx]
        
        new_img=cv2.resize(new_img,dsize=(2048, 1024),interpolation=cv2.INTER_NEAREST)
        return new_img
        
    
if __name__ == '__main__':
    config=edict()
    config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.cityscapes_split=random.choice(['test','val','train'])
    config.print_path=True
    config.resize_shape=(224,224)
    
    augmentations=Augmentations()
    dataset=cityscapes(config,split='train',augmentations=augmentations)
    loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=True,drop_last=False)
    for i, data in enumerate(loader):
        imgs, labels = data
        print(i,imgs.shape,labels.shape)
        plt.imshow(imgs[0,0])
        plt.show()
        plt.imshow(labels[0])
        plt.show()
        if i>=3:
            break
        
    print('validation dataset'+'*'*50)
    val_dataset=cityscapes(config,split='val',augmentations=augmentations)
    val_loader=TD.DataLoader(dataset=val_dataset,batch_size=2, shuffle=True,drop_last=False)
    for i, data in enumerate(val_loader):
        imgs, labels = data
        print(i,imgs.shape,labels.shape)
        
        if i>=3:
            break