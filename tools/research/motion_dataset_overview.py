# -*- coding: utf-8 -*-

import os
from torchseg.utils.configs.motionseg_config import get_default_config
from torchseg.models.motionseg.motion_utils import get_dataset
from torchseg.utils.disc_tools import show_images 
from tqdm import trange
from PIL import Image
import cv2
import numpy as np
from pprint import pprint

def get_category(dataset_name,gt_path):
    if dataset_name.lower() == 'cdnet2014':
        ### cdnet2014/PTZ/intermittentPan/groundtruth/gt000001.png
        path_splits=gt_path.split(os.path.sep)
        return path_splits[-4]
    else:
        raise NotImplementedError

def get_label(dataset_name,gt_path):
    if dataset_name.lower() == 'cdnet2014':
        ### cdnet2014/PTZ/intermittentPan/groundtruth/gt000001.png
        gt=cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
        def convert_label(img):
            ### background ==> 0
            labels=np.zeros_like(img)
            ### out of roi/unknown ==> 255
            labels[img==85]=255
            labels[img==170]=255
            
            ### foreground ==> 1
            labels[img==255]=1
            labels=labels.astype(np.uint8)
            return labels
        
        return convert_label(gt)
    else:
        raise NotImplementedError
    
def dataset_overview(dataset_name,split='train'):
    config=get_default_config()
    config.use_part_number=3000
    xxx_dataset=get_dataset(config,split)
    N=len(xxx_dataset)
    
    max_foregournd_in_category={}
    for i in trange(N):
        main,aux,main_gt,aux_gt=xxx_dataset.__get_path__(i)
        category=get_category(dataset_name,main_gt)
        
        label=get_label(dataset_name,main_gt)
        gt_area=np.count_nonzero(label==1)
        if category in max_foregournd_in_category.keys():
            if gt_area > max_foregournd_in_category[category]['area']:
                max_foregournd_in_category[category]={'area':gt_area,'path':[main,main_gt]}
        else:
            max_foregournd_in_category[category]={'area':gt_area,'path':[main,main_gt]}
    
    pprint(max_foregournd_in_category)
    
    return max_foregournd_in_category

def resize_img(img,target_height=200):
    h,w=img.shape[0:2]
    
    target_width=round((target_height/h)*w)
    
    if len(img.shape)==3:
        inter=cv2.INTER_LINEAR
    else:
        inter=cv2.INTER_NEAREST
    
    return cv2.resize(img,dsize=(target_width,target_height),interpolation=inter)
    
if __name__ == '__main__':
    dataset_name='cdnet2014'
    img_paths=dataset_overview(dataset_name)
    
    imgs=[None]*2*len(img_paths)
    titles=[None]*2*len(img_paths)
    idx=0
    for key,value in img_paths.items():
        main_path,gt_path=value['path']
        img=cv2.imread(main_path,cv2.IMREAD_COLOR)
        gt=cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
        
        
        imgs[idx]=resize_img(img)
        imgs[idx+1]=resize_img(gt)
        
        gt_path_splits=gt_path.split(os.path.sep)

        #titles[idx]=f'{gt_path_splits[-4]}/{gt_path_splits[-3]}'
        #titles[idx+1]=f'{gt_path_splits[-1]}'
        if key == 'intermittentObjectMotion':
            title='IOM'
        elif key == 'dynamicBackground':
            title='DB'
        else:
            title=key
            
        titles[idx]=title
        titles[idx+1]=''
        idx+=2
        
    fig=show_images(imgs,titles,row=4)
    fig.savefig(f'{dataset_name}_overview.eps',format='eps')
    fig.savefig(f'{dataset_name}_overview.pdf')