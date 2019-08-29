# -*- coding: utf-8 -*-

import os
import glob
import torch.utils.data as td
import random
import numpy as np
import cv2
from dataset.segtrackv2_dataset import main2flow,motionseg_dataset

class fbms_dataset(motionseg_dataset):
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        super().__init__(config,split,normalizations,augmentations)
        
        if split=='train':
            self.gt_files=glob.glob(os.path.join(self.config['root_path'],
                                                 'Trainingset',
                                                 '*',
                                                 'GroundTruth',
                                                 '*.png'),recursive=True)
        else:
            self.gt_files=glob.glob(os.path.join(self.config['root_path'],
                                                 'Testset',
                                                 '*',
                                                 'GroundTruth',
                                                 '*.png'),recursive=True)

        print('%s dataset size %d'%(split,len(self.gt_files)))
        self.gt_files.sort()
        if self.split in ['train','val','val_path']:
            n=len(self.gt_files)
            if n > self.config['use_part_number'] > 0:
                gap=n//self.config['use_part_number']
                self.gt_files=self.gt_files[::gap]
                print('total dataset image %d, use %d'%(n,len(self.gt_files)))
        elif self.split =='test':
            pass
        else:
            assert False

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

        if self.frame_gap==0:
            frame_gap=random.randint(1,10)
        else:
            frame_gap=self.frame_gap
        x=random.random()
        if x>0.5:
            aux_frame=get_frame_path(base_path,video_name,frame_index+frame_gap)
        else:
            aux_frame=get_frame_path(base_path,video_name,frame_index-frame_gap)

        assert os.path.exists(aux_frame),'aux_frame:{},gt_file:{}'.format(aux_frame,gt_file)
        return [main_frame,aux_frame]

    def __get_path__(self,index):
        frames=self.get_frames(self.gt_files[index])
        return frames[0],frames[1],self.gt_files[index]
    
    def __get_image__(self,index):
        main_file,aux_file,gt_file=self.__get_path__(index)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        gt_image=cv2.imread(self.gt_files[index],cv2.IMREAD_GRAYSCALE)
        return frame_images,gt_image,main_file,aux_file,gt_file