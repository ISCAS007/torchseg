# -*- coding: utf-8 -*-

import os
import torch.utils.data as td
import random
import numpy as np
import cv2
import sys
from dataset.segtrackv2_dataset import motionseg_dataset,main2flow

class cdnet_dataset(motionseg_dataset):
    def __init__(self,config,split='train',normalizations=None, augmentations=None):
        super().__init__(config,split,normalizations,augmentations)

        self.train_set=set()
        self.val_set=set()
        self.main_files=self.get_main_files(self.config.root_path)


        if self.split in ['train','val','val_path']:
            n=len(self.main_files)
            if n > self.config.use_part_number > 0:
                gap=n//self.config.use_part_number
                self.main_files=self.main_files[::gap]
                print('total dataset image %d, use %d'%(n,len(self.main_files)))
        elif self.split=='test':
            assert False
        else:
            assert False
        # random part
#            if self.config['use_part_number'] > 0:
#                n=len(self.img_path_pairs)
#                l=[i for i in range(n)]
#                random.shuffle(l)
#                part_pairs=[self.img_path_pairs[x] for x in l[0:self.config['use_part_number']]]
#                self.img_path_pairs=part_pairs
#                print('total dataset image %d, use %d'%(n,self.config['use_part_number']))

    def __len__(self):
        return len(self.main_files)

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

        if self.frame_gap==0:
            frame_gap=random.randint(1,10)
        else:
            frame_gap=self.frame_gap
        x=random.random()
        if x>0.5:
            aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number+frame_gap)
            aux_gt_path=self.get_image_path(root_path,category,sub_category,'gt',frame_number+frame_gap)
            if not os.path.exists(aux_img_path):
                aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number-frame_gap)
                aux_gt_path=self.get_image_path(root_path,category,sub_category,'gt',frame_number-frame_gap)
        else:
            aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number-frame_gap)
            aux_gt_path=self.get_image_path(root_path,category,sub_category,'gt',frame_number-frame_gap)
            if not os.path.exists(aux_img_path):
                aux_img_path=self.get_image_path(root_path,category,sub_category,'in',frame_number+frame_gap)
                aux_gt_path=self.get_image_path(root_path,category,sub_category,'gt',frame_number+frame_gap)


        assert os.path.exists(main_img_path),'main path not exists %s'%main_img_path
        assert os.path.exists(aux_img_path),'aux path not exists %s'%aux_img_path
        assert os.path.exists(gt_img_path),'gt path not exists %s'%gt_img_path
        assert os.path.exists(aux_gt_path),'aux gt path not exists %s'%aux_gt_path

        return (main_img_path,aux_img_path,gt_img_path,aux_gt_path)

    def generate_img_path_pair(self,main_img_path):
        """
        return (main_img_path,aux_img_path,gt_img_path)
        """
        basename = os.path.basename(main_img_path)
        frame_str = basename.split('.')[0]
        frame_str = frame_str[2:]
        frame_number=int(frame_str)

        root_path=os.path.dirname(os.path.dirname(main_img_path))
        category=sub_category=''

        return self.get_img_path_pair(root_path,category,sub_category,frame_number)

    def get_main_files(self,root_path):
        main_files=[]
        for category in os.listdir(root_path):
            if not os.path.isdir(os.path.join(root_path,category)):
                continue

            sub_category_list=[]
            for sub_category in os.listdir(os.path.join(root_path,category)):
                if os.path.isdir(os.path.join(root_path,category,sub_category)):
                    sub_category_list.append(sub_category)

            sub_category_list.sort()
            if self.split=='train':
                sub_category_list=sub_category_list[:-2]
                self.train_set.update(set(sub_category_list))
            elif self.split in ['val','val_path']:
                sub_category_list=sub_category_list[-2:]
                self.val_set.update(set(sub_category_list))
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

                with open(roi_txt_path,'r') as f:
                    line=f.readline()

                number_list=line.split(' ')
                number_list=[int(n) for n in number_list]
                first_frame, last_frame=tuple(number_list)

                if self.split in ['train','val','val_path']:
                    half_gt_categories = ['badWeather', 'lowFramerate', 'PTZ', 'nightVideos', 'turbulence']
                    if category in half_gt_categories:
                        last_frame = (first_frame + last_frame) // 2 - 1
#                        print('category %s  subcategory %s with groundtruth image, roi is %d to %d' % (
#                            category, sub_category, first_frame, last_frame))

                    main_files+=[self.get_image_path(root_path,category,sub_category,'in',frame_number)
                                        for frame_number in range(first_frame,last_frame+1)]
                elif self.split=='test':
                    input_root_path=os.path.join(root_path,category,sub_category,'input')
                    main_files=[os.path.join(input_root_path,image_file) for image_file in os.listdir(input_root_path)
                                        if image_file.lower().endswith(('jpg','png','jpeg','bmp'))]
                else:
                    assert False

        print('%s size = %d'%(self.split,len(main_files)))

        main_files.sort()
        return main_files

    def __get_path__(self,index):
        main_file=self.main_files[index]
        return self.generate_img_path_pair(main_file)

    def __get_image__(self,index):
        main_img_path,aux_img_path,gt_img_path,aux_gt_path=self.__get_path__(index)

        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_img_path,aux_img_path]]
        gt_images=[cv2.imread(p,cv2.IMREAD_GRAYSCALE) for p in [gt_img_path,aux_gt_path]]

        def convert_label(img):
            labels=np.zeros_like(img)
            labels[img==85]=255
            labels[img==170]=1
            labels[img==255]=1
            labels=labels.astype(np.uint8)
            return labels

        labels=[convert_label(img) for img in gt_images]
        return frame_images,labels,main_img_path,aux_img_path,gt_img_path
