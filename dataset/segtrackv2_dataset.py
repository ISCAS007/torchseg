# -*- coding: utf-8 -*-

import os
import glob
import torch.utils.data as td
import numpy as np
import random
import cv2

def main2flow(main_path,
              dataset_root_dir=os.path.expanduser('~/cvdataset'),
              flow_root_dir=os.path.expanduser('~/cvdataset/optical_flow')):
    """
    convert main path to flow path

    # path to save optical flow results
    flow_root_dir=os.path.expanduser('~/cvdataset/optical_flow')
    # root path for all dataset
    dataset_root_dir=os.path.expanduser('~/cvdataset')
    """

    assert main_path.find(dataset_root_dir)>=0,'suppose is wrong for {} and {} !!! cannot generate out_path'.format(main_path,dataset_root_dir)
    out_path=main_path.replace(dataset_root_dir,flow_root_dir)
    suffix=os.path.splitext(out_path)[1]
    out_path=out_path.replace(suffix,'.flo')

    return out_path

class motionseg_dataset(td.Dataset):
    """
    base class dataset
    """
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        self.config=config
        self.split=split
        self.normalizations=normalizations
        self.augmentations=augmentations
        self.input_shape=tuple(config.input_shape)
        self.root_path=config.root_path
        self.frame_gap=config.frame_gap
        self.use_optical_flow=config.use_optical_flow
        self.ignore_pad_area=config.ignore_pad_area
        
    def __get_image__(self,index):
        """
        return frame_images,gt_image,main_path,aux_path,gt_path
        """
        assert False

    def __getitem__(self,index):
        frame_images,gt_image,main_path,aux_path,gt_path=self.__get_image__(index)

        # augmentation dataset
        if self.split=='train' and self.augmentations is not None:
            frame_images=[self.augmentations.transform(img) for img in frame_images]

        # resize image
        resize_frame_images=[cv2.resize(img,self.input_shape,interpolation=cv2.INTER_LINEAR) for img in frame_images]
    
        if self.split=='train':
            resize_gt_image=cv2.resize(gt_image,self.input_shape,interpolation=cv2.INTER_NEAREST)
        else:
            resize_gt_image=gt_image
        
        # ignore_area, 0 for not ignore, 255 for ignore
        ignore_area=np.zeros_like(resize_gt_image)
        if self.split=='train':
            if self.ignore_pad_area>0:
                ignore_area[0:self.ignore_pad_area,:]=255
                ignore_area[-self.ignore_pad_area:,:]=255
                ignore_area[:,0:self.ignore_pad_area]=255
                ignore_area[:,-self.ignore_pad_area:]=255
        
        # only in cdnet2014, 255(85 in gt image) stands for ignore add
        if self.config.dataset=='cdnet2014':
            ignore_area[resize_gt_image==255]=255
            
        # normalize image
        if self.normalizations is not None:
            resize_frame_images = [self.normalizations.forward(img) for img in resize_frame_images]

        # bchw
        resize_frame_images=[img.transpose((2,0,1)) for img in resize_frame_images]
        
        resize_gt_image=(resize_gt_image!=0).astype(np.uint8)
        resize_gt_image[ignore_area==255]=255
        resize_gt_image=np.expand_dims(resize_gt_image,0)
        
        
        if self.use_optical_flow:
            flow_path=main2flow(main_path)
            flow_file=open(flow_path,'r')
            a=np.fromfile(flow_file,np.uint8,count=4)
            b=np.fromfile(flow_file,np.int32,count=2)
            flow=np.fromfile(flow_file,np.float32).reshape((b[1],b[0],2))
            flow=np.clip(flow,a_min=-50,a_max=50)/50.0
            optical_flow=cv2.resize(flow,self.input_shape,interpolation=cv2.INTER_LINEAR).transpose((2,0,1))
            if self.split=='val_path':
                return {'images':[resize_frame_images[0],optical_flow],
                        'gt':resize_gt_image,
                        'gt_path':gt_path,
                        'shape':frame_images[0].shape}
            else:
                return [resize_frame_images[0],optical_flow],resize_gt_image
        else:
            if self.split=='val_path':
                return {'images':resize_frame_images,
                        'gt':resize_gt_image,
                        'gt_path':gt_path,
                        'shape':frame_images[0].shape}
            else:
                return resize_frame_images,resize_gt_image

class segtrackv2_dataset(motionseg_dataset):
    """
    multi instances tracking dataset
    """
    def __init__(self,config,split='train',normalizations=None,augmentations=None):
        super().__init__(config,split,normalizations,augmentations)
        
        self.main_files=self.get_main_files()

        print('dataset size = {}',len(self.main_files))
        n=len(self.main_files)
        if n > self.config.use_part_number > 0:
            gap=n//self.config.use_part_number
            self.main_files=self.main_files[::gap]
            print('total dataset image %d, use %d'%(n,len(self.main_files)))

    def __len__(self):
        return len(self.main_files)

    def get_gt_files(self,main_file):
        """
        GroundTruth
        ├── birdfall
        ├── bird_of_paradise
        ├── bmx
        │   ├── 1
        │   └── 2
        ...
        """
        path_strings=main_file.split(os.path.sep)
        basename=path_strings[-1]
        basename=basename.split('.')[0]
        video_name=path_strings[-2]

        gt_files=[]
        for suffix in ['bmp','png']:
            gt_path=os.path.join(self.root_path,'GroundTruth',video_name,'**',basename+'.'+suffix)
            gt_files+=glob.glob(gt_path,recursive=True)
        assert len(gt_files)>0,'gt_path={}'.format(gt_path)
        return gt_files

    def get_aux_file(self,main_file):
        path_strings=main_file.split(os.path.sep)
        basename=path_strings[-1]
        video_name=path_strings[-2]
        suffix=basename.split('.')[1]

        frame_path=os.path.join(self.root_path,'JPEGImages')
        video_path=os.path.join(frame_path,video_name,'*.'+suffix)
        video_files=glob.glob(video_path)
        video_files.sort()

        main_index=video_files.index(main_file)
        video_files.remove(main_file)
        n=len(video_files)
        assert n>0,'main_file={}'.format(main_file)
        assert main_index>=0,'main_file={}'.format(main_file)

        if self.frame_gap==0:
            frame_gap=random.randint(1,10)
        else:
            frame_gap=self.frame_gap

        x=random.random()
        if x>0.5:
            aux_index=main_index+frame_gap
            aux_index=aux_index if aux_index<n else n-1
        else:
            aux_index=main_index-frame_gap
            aux_index=aux_index if aux_index>=0 else 0

        return video_files[aux_index]

    def get_main_files(self):
        frame_path=os.path.join(self.root_path,'JPEGImages')
        videos=[d for d in os.listdir(frame_path) if os.path.isdir(os.path.join(frame_path,d))]
        videos.sort()
        videos.remove('penguin')
        assert len(videos)>0,'frame_path={}'.format(frame_path)
        n=len(videos)
        train_size=int(n*0.7)
        if self.split=='train':
            videos=videos[:train_size]
        else:
            videos=videos[train_size:]

        frame_files=[]
        for v in videos:
            video_files=[]
            for suffix in ['bmp','png']:
                video_path=os.path.join(frame_path,v,'*.'+suffix)
                video_files+=glob.glob(video_path)
            assert len(video_files)>0,'video_path={}'.format(video_path)
            frame_files+=video_files

        frame_files.sort()
        assert len(frame_files)>0,'videos={}'.format(videos)
        return frame_files

    def __get_path__(self,index):
        main_file=self.main_files[index]
        aux_file=self.get_aux_file(main_file)
        gt_files=self.get_gt_files(main_file)

        return main_file,aux_file,gt_files
    
    def __get_image__(self,index):
        main_file,aux_file,gt_files=self.__get_path__(index)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        gt_images=[cv2.imread(f,cv2.IMREAD_GRAYSCALE) for f in gt_files]
        gt_image=np.zeros_like(gt_images[0])
        for gt in gt_images:
            gt_image+=gt
        
        return frame_images,gt_image,main_file,aux_file,gt_files[0]