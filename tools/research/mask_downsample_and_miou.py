# -*- coding: utf-8 -*-

from torchseg.dataset.dataset_generalize import dataset_generalize, \
    get_dataset_generalize_config

from torchseg.utils.metrics import runningScore
import cv2
import numpy as np
from easydict import EasyDict as edict
if __name__ == '__main__':
    config=edict()
    config.with_edge=False
    config=get_dataset_generalize_config(config,'HuaWei')
    
    val_dataset = dataset_generalize(config,
                                     split='val',
                                     augmentations=None,
                                     normalizations=None)
    
    N=len(val_dataset)
    
    if config.ignore_index == 0:
        config.class_number = len(config.foreground_class_ids)+1
    else:
        config.class_number = len(config.foreground_class_ids)
        
    metric=runningScore(config.class_number)
    print('| scale | miou |')
    print('| -\t | -\t |')
    for scale in range(2,9):
        metric.reset()
        for i in range(N):
            img,ann=val_dataset.__getitem__(i)
            
            
            h,w=ann.shape
            ann_downsample=cv2.resize(ann,dsize=(0,0),fx=1.0/scale,fy=1.0/scale,interpolation=cv2.INTER_NEAREST)
            ann_upsample=cv2.resize(ann_downsample,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
            
            ann_upsample[ann_upsample==config.ignore_index]=0
            metric.update(ann,ann_upsample)
            
            if i>10:
                break
        scores,cls_iu=metric.get_scores()
        miou=scores['Mean IoU : \t']
        
        print('| {} | {:.4f} |'.format(scale,miou))
    
    
    ### test for common size
    print('| size | miou |')
    print('| -\t | -\t |')
    for size in [(224,224),(256,256),(256,512),(512,1024)]:
        metric.reset()
        for i in range(N):
            img,ann=val_dataset.__getitem__(i)
            
            ann[ann==config.ignore_index]=0
            h,w=ann.shape
            ann_downsample=cv2.resize(ann,dsize=size,interpolation=cv2.INTER_NEAREST)
            ann_upsample=cv2.resize(ann_downsample,dsize=(w,h),interpolation=cv2.INTER_NEAREST)
            
            # print(ann.shape,ann_downsample.shape,ann_upsample.shape)
            # print(ann.dtype,ann_downsample.dtype,ann_upsample.dtype)
            # print(np.unique(ann),np.unique(ann_downsample),np.unique(ann_upsample))
            ann_upsample[ann_upsample==config.ignore_index]=0
            metric.update(ann,ann_upsample)
            
            if i>10:
                break
        scores,cls_iu=metric.get_scores()
        miou=scores['Mean IoU : \t']
        
        print('| {} | {:.4f} |'.format(size,miou))
    