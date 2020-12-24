# -*- coding: utf-8 -*-

import os
import glob
import random
import numpy as np
import cv2
import netpbmfile as pbm
from .segtrackv2_dataset import motionseg_dataset
import warnings

class fbms_dataset(motionseg_dataset):
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        super().__init__(config,split,normalizations,augmentations)

        #for fbms-3d, the first label may be invalid
        self.remove_first_empty_gt=True

        if config.root_path.lower().find('fbms-3d')>=0:
            self.gt_format='png'
        else:
            self.gt_format='ppm'

        if split=='train':
            split_dir='Trainingset'
        else:
            split_dir='Testset'

        if self.gt_format=='ppm':
            self.gt_files=[]

            clips_dir=os.listdir(os.path.join(self.config['root_path'],split_dir))
            for d in clips_dir:

                ppm_files=glob.glob(os.path.join(self.config['root_path'],
                                                 split_dir,
                                                 d,
                                                 'GroundTruth',
                                                 '*.'+self.gt_format),recursive=True)
                pgm_files=glob.glob(os.path.join(self.config['root_path'],
                                                 split_dir,
                                                 d,
                                                 'GroundTruth',
                                                 '*.pgm'),recursive=True)
                if len(ppm_files)==0:
                    files=pgm_files
                else:
                    files=[f for f in ppm_files if f.find('PROB_gt.ppm')==-1]

                files.sort()
                if self.remove_first_empty_gt:
                    gt=self.imread(files[0])
                    if np.sum(gt)==0:
                        warnings.warn('remove invalid label',files[0])
                        files=files[1:]

                self.gt_files+=files
        else:
            self.gt_files=glob.glob(os.path.join(self.config['root_path'],
                                             split_dir,
                                             '*',
                                             'GroundTruth',
                                             '*.'+self.gt_format),recursive=True)


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

        # image file
        self.img_files=[self.get_frames(gt_file)[0] for gt_file in self.gt_files]

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
        # or gt_file=dataset/FBMS/Trainingset/bear01/GroundTruth/bear01_0001_gt.ppm
        path_strings=gt_file.split(os.path.sep)
        video_name=path_strings[-3]

        if self.gt_format=='png':
            index_string=path_strings[-1].split('_')[0]
        else:
            if video_name=='tennis':
                index_string=path_strings[-1].replace(video_name,"")
            else:
                index_string=path_strings[-1].split('_')[1]
            index_string=index_string.replace(".pgm","")
        frame_index=int(index_string)


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

    def imread(self,file):
        if self.gt_format=='png':
            gt_image=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        else:
            img_rgb=pbm.imread(file)
            if len(img_rgb.shape)==3:
                gt_image=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
                gt_image[gt_image==255]=0
            else:
                gt_image=img_rgb

        return gt_image

    def __get_image__(self,index):
        main_file,aux_file,gt_file=self.__get_path__(index)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        gt_image=self.imread(self.gt_files[index])

        # find aux_gt_file
        if aux_file in self.img_files:
            aux_index=self.img_files.index(aux_file)
            aux_gt_image=self.imread(self.gt_files[aux_index])
        else:
            aux_gt_image=np.zeros_like(gt_image)

        labels=[]
        for gt in [gt_image,aux_gt_image]:
            label=np.zeros_like(gt)
            label[gt>0]=1
            labels.append(label)
        return frame_images,labels,main_file,aux_file,gt_file