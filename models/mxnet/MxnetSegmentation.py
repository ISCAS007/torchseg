# -*- coding: utf-8 -*-
from gluoncv.data.segbase import SegmentationDataset
from easydict import EasyDict as edict
from PIL import Image
import mxnet.ndarray as F
import numpy as np
from mxnet import cpu
import os

from dataset.dataset_generalize import get_dataset_generalize_config

class MxnetSegmentation(SegmentationDataset):
    def __init__(self,root=None,split='train',mode=None,transform=None,config=edict(),name='Cityscapes'):
        super().__init__(root,split,mode,transform)
        self.config=get_dataset_generalize_config(config,name)
        self.NUM_CLASS = len(self.config.foreground_class_ids)
        if root is None:
            root=self.config.root_path
        
        splits=['train','val','test','train_extra']
        assert self.split in splits,'unexcepted split %s for dataset, must be one of %s'%(self.split,str(splits))
        
        if hasattr(self.config,'txt_note'):
            self.split=self.config.txt_note+'_'+self.split
            
        if hasattr(self.config,'txt_path'):
            txt_file=os.path.join(self.config.txt_path,self.split+'.txt')
            self.image_files,self.annotation_files = self.get_files_from_txt(txt_file,self.config.root_path)
            assert len(self.image_files)>0, 'No files found in %s with %s'%(self.config.root_path,txt_file)
            assert len(self.annotation_files)>0, 'No files found in %s with %s'%(self.config.root_path,txt_file)
        else:
            assert hasattr(self.config,'image_txt_path'),'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            assert hasattr(self.config,'annotation_txt_path'),'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            image_txt_file=os.path.join(self.config.image_txt_path,self.split+'.txt')
            annotation_txt_file=os.path.join(self.config.annotation_txt_path,self.split+'.txt')
            self.image_files=self.get_files_from_txt(image_txt_file,self.config.root_path)
            self.annotation_files=self.get_files_from_txt(annotation_txt_file,self.config.root_path)
            assert len(self.image_files)>0, 'No files found in %s with %s'%(self.config.root_path,image_txt_file)
            assert len(self.annotation_files)>0, 'No files found in %s with %s'%(self.config.root_path,annotation_txt_file)
            
        self.images=[os.path.join(root,p) for p in self.image_files]
        self.masks=[os.path.join(root,p) for p in self.annotation_files]
    
    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('Unknown dataset mode %s'%self.mode)
            
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask
        
    def __len__(self):
        return len(self.images)
    
    def _mask_transform(self,mask):
        target = np.array(mask).astype('int32')
        ann=np.zeros_like(target)
        self.ignore_index=-1
        for class_id in self.background_class_ids:
            ann[target == class_id] = self.ignore_index
        for idx, class_id in enumerate(self.foreground_class_ids):
            ann[target == class_id] = idx
        return F.array(ann, cpu(0))
    
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