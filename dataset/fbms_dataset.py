# -*- coding: utf-8 -*-

import os
import glob
import torch.utils.data as td
import random
import numpy as np
import cv2
from dataset.segtrackv2_dataset import main2flow

class fbms_dataset(td.Dataset):
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        self.config=config
        self.split=split
        self.normalizations=normalizations
        self.augmentations=augmentations
        self.input_shape=tuple(config.input_shape)
        self.use_optical_flow=config.use_optical_flow
        if split=='train':
            self.gt_files=glob.glob(os.path.join(self.config['train_path'],'*','GroundTruth','*.png'),recursive=True)
        else:
            self.gt_files=glob.glob(os.path.join(self.config['val_path'],'*','GroundTruth','*.png'),recursive=True)
        
        print('%s dataset size %d'%(split,len(self.gt_files)))
        self.gt_files.sort()
        if self.split in ['train','val']:
            n=len(self.gt_files)
            if n > self.config['use_part_number'] > 0:
                gap=n//self.config['use_part_number']
                self.gt_files=self.gt_files[::gap]
                print('total dataset image %d, use %d'%(n,len(self.gt_files)))
        
    def __len__(self):
        return len(self.gt_files)
    
    def get_frames(self,gt_file):
        def get_frame_index_bound(base_path,video_name):
            """
            in images, not in groundtruth
            """
            frames=glob.glob(os.path.join(base_path,'*.jpg'))
            frames.sort()
            target_frames=[frames[0],frames[-1]]
            if video_name!='tennis':
                bound=[int(f.split(os.path.sep)[-1].split('_')[1].split('.')[0]) for f in target_frames]
            else:
                bound=[int(f.split(os.path.sep)[-1].split('.')[0].replace(video_name,'')) for f in target_frames]
            
            assert bound[0]<bound[1]
            return bound
            
        def get_frame_path(base_path,video_name,frame_index):
            bound=get_frame_index_bound(base_path,video_name)
            if frame_index<bound[0]:
                #print('change frame index from {} to {} for {}'.format(frame_index,bound[0],base_path))
                frame_index=bound[0]
            elif frame_index>bound[1]:
                #print('change frame index from {} to {} for {}'.format(frame_index,bound[1],base_path))
                frame_index=bound[1]
                
            if video_name!='tennis':
                path=os.path.join(base_path,video_name+'_'+'%02d'%frame_index)+'.jpg'
                if not os.path.exists(path):
                    path=os.path.join(base_path,video_name+'_'+'%03d'%frame_index)+'.jpg'
                if not os.path.exists(path):
                    path=os.path.join(base_path,video_name+'_'+'%04d'%frame_index)+'.jpg'
            else:
                path=os.path.join(base_path,video_name+'%03d'%frame_index)+'.jpg'
            
            assert os.path.exists(path),'path={},base_path={},frame_index={}'.format(path,base_path,frame_index)
            return path
        
        # gt_file=dataset/FBMS/Trainingset/bear01/GroundTruth/001_gt.png
        path_strings=gt_file.split(os.path.sep)
        
        index_string=path_strings[-1].split('_')[0]
        frame_index=int(index_string)
        video_name=path_strings[-3]
        
        base_path=os.path.sep.join(path_strings[0:-2])
        main_frame=get_frame_path(base_path,video_name,frame_index)
        assert os.path.exists(main_frame),'main_frame:{},gt_file:{}'.format(main_frame,gt_file)
        
        x=random.random()
        if x>0.5:
            aux_frame=get_frame_path(base_path,video_name,frame_index+self.config['frame_gap'])
        else:
            aux_frame=get_frame_path(base_path,video_name,frame_index-self.config['frame_gap'])
    
        assert os.path.exists(aux_frame),'aux_frame:{},gt_file:{}'.format(aux_frame,gt_file)
        return [main_frame,aux_frame]
    
    def __get_path__(self,index):
        frames=self.get_frames(self.gt_files[index])
        return frames[0],frames[1],self.gt_files[index]
    
    def __getitem__(self,index):
        frames=self.get_frames(self.gt_files[index])
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in frames]
        gt_image=cv2.imread(self.gt_files[index],cv2.IMREAD_GRAYSCALE)
        
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

        if self.use_optical_flow:
            flow_path=main2flow(frames[0])
            flow_file=open(flow_path,'r')
            a=np.fromfile(flow_file,np.uint8,count=4)
            b=np.fromfile(flow_file,np.int32,count=2)
            flow=np.fromfile(flow_file,np.float32).reshape((b[1],b[0],2))
            flow=np.clip(flow,a_min=-50,a_max=50)/50.0
            optical_flow=cv2.resize(flow,self.input_shape,interpolation=cv2.INTER_LINEAR).transpose((2,0,1))
            return [resize_frame_images[0],optical_flow],resize_gt_image
        else:
            return resize_frame_images,resize_gt_image