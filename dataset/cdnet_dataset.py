# -*- coding: utf-8 -*-

import os
import torch.utils.data as td
import random
import numpy as np
import cv2
import sys

class cdnet_dataset(td.Dataset):
    def __init__(self,config,split='train',normalizations=None):
        self.config=config
        self.split=split
        self.normalizations=normalizations
        
        self.img_path_pairs=self.get_img_path_pairs(self.config['root_path'])
        if self.config['use_part_number'] > 0:
            n=len(self.img_path_pairs)
            l=[i for i in range(n)]
            random.shuffle(l)
            part_pairs=[self.img_path_pairs[x] for x in l[0:self.config['use_part_number']]]
            self.img_path_pairs=part_pairs
            print('total dataset image %d, use %d'%(n,self.config['use_part_number']))
        
    def __len__(self):
        return len(self.img_path_pairs)
    
    def get_image_path(self, root_path, category, sub_category, data_type, frame_num):
        """
        root_path: root_path for dataset
        category: {'2012':baseline dynamicBackground shadow cameraJitter intermittentObjectMotion thermal,'2014':...}
        sub_category: {'2012::baseline':highway  office  pedestrians  PETS2006 ...}
        data_type: input or groundtruth
        frame_num: 1,2,3,... start from 1.
        return: root_path/baseline/highway/input/in000001.jpg
        """

        current_path = os.path.join(root_path, category, sub_category)
        if data_type == 'input' or data_type == 'in' or data_type == 'x':
            current_path = os.path.join(current_path, 'input', 'in%06d.jpg' % frame_num)
        elif data_type == 'groundtruth' or data_type == 'gt' or data_type == 'y':
            current_path = os.path.join(current_path, 'groundtruth', 'gt%06d.png' % frame_num)
        else:
            assert False,'unknown data_type %s'%data_type

        return current_path
    
    def get_img_path_pair(self,root_path,category,sub_category,frame_number):
        """
        return (main_img_path,aux_img_path,gt_img_path)
        """
        main_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number)
        
        gt_img_path=self.get_image_path(root_path,category,sub_category,'gt',frame_number)
        
        frame_gap=self.config['frame_gap']
        x=random.random()
        if x>0.5:
            aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number+frame_gap)
            if not os.path.exists(aux_img_path):
                aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number-frame_gap)
        else:
            aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number-frame_gap)
            if not os.path.exists(aux_img_path):
                aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number+frame_gap)
        
            
        assert os.path.exists(main_img_path),'main path not exists %s'%main_img_path
        assert os.path.exists(aux_img_path),'aux path not exists %s'%aux_img_path
        assert os.path.exists(gt_img_path),'gt path not exists %s'%gt_img_path
        
        return (main_img_path,aux_img_path,gt_img_path)
    
    def generate_img_path_pair(self,main_img_path):
        """
        return (main_img_path,aux_img_path,gt_img_path)
        """
        basename = os.path.basename(main_img_path)
        frame_str = basename.split('.')[0]
        frame_str = frame_str.replace('in', '')
        frame_number=int(frame_str)
        
        root_path=os.path.dirname(main_img_path)
        category=sub_category=''
        
        return self.get_img_path_pair(root_path,category,sub_category,frame_number)
        
    def get_img_path_pairs(self,root_path):
        img_path_pairs=[]        
        for category in os.listdir(root_path):
            if not os.path.isdir(os.path.join(root_path,category)):
                continue
            
            sub_category_list=[]
            for sub_category in os.listdir(os.path.join(root_path,category)):
                if os.path.isdir(os.path.join(root_path,category,sub_category)):
                    sub_category_list.append(sub_category)
            
            if self.split=='train':
                sub_category_list=sub_category_list[:-2]
            elif self.split=='val':
                sub_category_list=sub_category_list[-2:]
            elif self.split=='test':
                pass
            else:
                assert False
            
            for sub_category in sub_category_list:
                roi_img_path = os.path.join(root_path , category, sub_category, 'ROI.bmp')
                roi_txt_path = os.path.join(root_path , category, sub_category, 'temporalROI.txt')
                if not os.path.exists(roi_img_path):
                    assert False,'%s not exists'%roi_img_path
                
                if not os.path.exists(roi_txt_path):
                    assert False,'%s not exists'%roi_txt_path
                    
                f=open(roi_txt_path,'r')
                line=f.readline()
                number_list=line.split(' ')
                number_list=[int(n) for n in number_list]
                first_frame, last_frame=tuple(number_list)
                
                if self.split in ['train','val']:
                    half_gt_categories = ['badWeather', 'lowFramerate', 'PTZ', 'nightVideos', 'turbulence']
                    if category in half_gt_categories:
                        last_frame = (first_frame + last_frame) // 2 - 1
                        print('category %s  subcategory %s with groundtruth image, roi is %d to %d' % (
                            category, sub_category, first_frame, last_frame))
                    
                    img_path_pairs+=[self.get_img_path_pair(root_path,category,sub_category,frame_number) 
                                        for frame_number in range(first_frame,last_frame+1)]
                elif self.split=='test':
                    input_root_path=os.path.join(root_path,category,sub_category,'input')
                    main_img_paths=[os.path.join(input_root_path,image_file) for image_file in os.listdir(input_root_path)
                                        if image_file.lower().endswith('jpg','png','jpeg','bmp')]
                    
                    
                    img_path_pairs+=[self.generate_img_path_pair(p) for p in main_img_paths]
                else:
                    assert False
                    
        print('%s size = %d'%(self.split,len(img_path_pairs)))
        
        return img_path_pairs
    
    def __getitem__(self,index):
        main_img_path,aux_img_path,gt_img_path=self.img_path_pairs[index]
        
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_img_path,aux_img_path]]
        gt_image=cv2.imread(gt_img_path,cv2.IMREAD_GRAYSCALE)
        
        # resize image
        resize_frame_images=[cv2.resize(img,(224,224),interpolation=cv2.INTER_LINEAR) for img in frame_images]
        resize_gt_image=cv2.resize(gt_image,(224,224),interpolation=cv2.INTER_NEAREST)
        
        # normalize image
        if self.normalizations is not None:
            resize_frame_images = [self.normalizations.forward(img) for img in resize_frame_images]
        
        # bchw
        resize_frame_images=[img.transpose((2,0,1)) for img in resize_frame_images]
        resize_gt_image=np.expand_dims(resize_gt_image,0)
        
        # for groundtruth image: outside roi=85,unknown=170,motion=255,hard shadow=50,static=0
        labels=np.zeros_like(resize_gt_image)
        labels[resize_gt_image==85]=255
        labels[resize_gt_image==170]=1
        labels[resize_gt_image==255]=1
        labels=labels.astype(np.uint8)
        return resize_frame_images,labels
                
                