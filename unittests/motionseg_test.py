# -*- coding: utf-8 -*-
import os
from torchseg.models.motionseg.motion_utils import get_dataset,get_model
from torchseg.utils.configs.motionseg_config import get_default_config
from torchseg.utils.configs.semanticseg_config import load_config
from torchseg.dataset.motionseg_dataset_factory import motionseg_show_images
from torchseg.dataset.motionseg_dataset_factory import prepare_input_output
from torchseg.utils.disc_tools import show_images
import torch.utils.data as td
import unittest
import cv2
from tqdm import trange
import numpy as np
import torch
import glob
import netpbmfile as pbm

class Test(unittest.TestCase):
    config=get_default_config()
    config.use_part_number=0

    def test_dataset(self):
        def test_img(p):
            try:
                img=cv2.imread(p)
            except Exception as e:
                print(dataset,p,e)
            else:
                self.assertIsNotNone(img,p)

        # 'FBMS','cdnet2014','segtrackv2', 'BMCnet'
        for dataset in ['FBMS','FBMS-3D']:
            self.config.dataset=dataset
            for split in ['train','val']:
                xxx_dataset=get_dataset(self.config,split)
#                N=min(10,len(xxx_dataset))
                N=len(xxx_dataset)
                for i in trange(N):
                    main,aux,gt=xxx_dataset.__get_path__(i)
                    for p in [main,aux,gt]:
                        if isinstance(p,str):
                            test_img(p)
                        else:
                            for x in p:
                                test_img(x)

    def test_ignore_pixel(self):
        def test_img(p,pixels):
            img=cv2.imread(p)
            new_pixels=np.unique(img)
            pixels=pixels.union(new_pixels)
            return pixels

        # 'FBMS','cdnet2014','segtrackv2', 'BMCnet', 'DAVIS2016', 'DAVIS2017'
        for dataset in ['DAVIS2016','DAVIS2017']:
            self.config.dataset=dataset
            for split in ['train','val']:
                pixels=set()
                xxx_dataset=get_dataset(self.config,split)
#                N=min(100,len(xxx_dataset))
                N=len(xxx_dataset)
                for i in trange(N):
                    main,aux,gt=xxx_dataset.__get_path__(i)
                    if isinstance(gt,str):
                        pixels=test_img(gt,pixels)
                    else:
                        for x in gt:
                            pixels=test_img(x,pixels)

                print(dataset,split,pixels)
        self.assertTrue(True)

    def test_motion_diff(self):
        """
        load model and show model output
        """
        #config_txt=os.path.expanduser('~/tmp/logs/motion/motion_diff/cdnet2014/test/2020-09-25___19-16-21/config.txt')
        config_txt=os.path.expanduser('~/tmp/logs/motion/motion_diff/FBMS/test/2020-09-25___18-52-18/config.txt')
        config=load_config(config_txt)

        model=get_model(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        for split in ['train','val']:
            xxx_loader=get_dataset(config,split)
            dataset_loader=td.DataLoader(dataset=xxx_loader,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=2)
            for data in dataset_loader:

                images,origin_labels,resize_labels=prepare_input_output(data=data,device=device,config=config)
                motionseg_show_images(images,origin_labels,[])

                outputs=model(images)
                predict=outputs['masks'][0]
                motionseg_show_images([],[],predict)
                break
        self.assertTrue(True)

    def test_fbms_pgm(self):
        for dataset in ['FBMS']:
            self.config.dataset=dataset
            xxx_dataset=get_dataset(self.config,'train')
            root=xxx_dataset.config.root_path



            ppm_files=glob.glob(os.path.join(root,
                                                 'Trainingset',
                                                 '*',
                                                 'GroundTruth',
                                                 '*.ppm'),recursive=True)

            pgm_files=glob.glob(os.path.join(root,
                                                 'Trainingset',
                                                 '*',
                                                 'GroundTruth',
                                                 '*.pgm'),recursive=True)

            in_count=noin_count=0
            for pgm in pgm_files:
                ppm=pgm.replace('.pgm','.ppm')
                if ppm in ppm_files:
                    in_count+=1
                else:
                    if noin_count==0:
                        print(pgm,ppm)
                    noin_count+=1

            print(root)
            print('in_count={}, noin_count={}, pgm={}, ppm={}'.format(in_count,noin_count,len(pgm_files),len(ppm_files)))
            self.assertTrue(True)

    def test_duration(self):
        def get_shape(img_file):
            if img_file.endswith(('ppm','pgm')):
                img=pbm.imread(img_file)
            else:
                img=cv2.imread(img_file)

            return img.shape[:2]

        def statistic(imgs,v,min_duration,max_duration,min_shape,max_shape):
            duration=len(imgs)
            if duration==0:
                print(v,duration)
                return min_duration,max_duration,min_shape,max_shape

            shape=get_shape(imgs[0])

            if min_duration is None:
                max_duration=min_duration=duration
                min_shape=max_shape=shape
            else:
                min_duration=min(min_duration,duration)
                max_duration=max(max_duration,duration)

                if shape[0]*shape[1]>max_shape[0]*max_shape[1]:
                    max_shape=shape

                if shape[0]*shape[1]<min_shape[0]*min_shape[1]:
                    min_shape=shape

            print(v,duration,shape)

            return min_duration,max_duration,min_shape,max_shape

        for dataset in ['FBMS','FBMS-3D']:
            self.config.dataset=dataset
            xxx_dataset=get_dataset(self.config,'train')
            root=xxx_dataset.config.root_path
            min_duration=None
            max_duration=None
            min_shape=max_shape=None

            clips=glob.glob(os.path.join(root,'*','*'),recursive=False)
            for v in clips:
                imgs=glob.glob(os.path.join(v,'*.jpg'))
                min_duration,max_duration,min_shape,max_shape=statistic(imgs,v,min_duration,max_duration,min_shape,max_shape)

            print(dataset,min_duration,max_duration,min_shape,max_shape)

        for dataset in ['DAVIS2016','DAVIS2017']:
            self.config.dataset=dataset
            xxx_dataset=get_dataset(self.config,'train')
            root=xxx_dataset.config.root_path
            min_duration=None
            max_duration=None
            min_shape=max_shape=None

            clips=[]
            for txt_file in glob.glob(os.path.join(root,'ImageSets','201*','*.txt')):
                f=open(txt_file,'r')
                clips+=f.readlines()
                f.close()

            clips=[c.strip() for c in clips]

            for v in clips:
                imgs=glob.glob(os.path.join(root,'JPEGImages','480p',v,'*.jpg'))
                min_duration,max_duration,min_shape,max_shape=statistic(imgs,v,min_duration,max_duration,min_shape,max_shape)

            print(dataset,min_duration,max_duration,min_shape,max_shape)

        for dataset in ['segtrackv2']:
            self.config.dataset=dataset
            xxx_dataset=get_dataset(self.config,'train')
            root=xxx_dataset.config.root_path
            min_duration=None
            max_duration=None

            clips=glob.glob(os.path.join(root,'JPEGImages','*'),recursive=False)
            print(clips)
            for v in clips:
                imgs=[]
                for fmt in ['png','bmp']:
                    imgs+=glob.glob(os.path.join(v,'*.'+fmt))

                min_duration,max_duration,min_shape,max_shape=statistic(imgs,v,min_duration,max_duration,min_shape,max_shape)
            print(dataset,min_duration,max_duration,min_shape,max_shape)

    def compare_fbms_3dmotion(self):
        def filter_gt_files(gt_files):
            new_gt_files=[]
            for f in gt_files:
                finded=False
                for new_f in new_gt_files:
                    if os.path.dirname(f)==os.path.dirname(new_f):
                        finded=True
                        break
                else:
                    assert finded==False
                    new_gt_files.append(f)

            return new_gt_files

        images={}
        files={}
        for dataset in ['FBMS','FBMS-3D']:
            for split in ['train','val']:
                self.config.dataset=dataset
                xxx_dataset=get_dataset(self.config,split)
                files[dataset+'/'+split]=xxx_dataset.gt_files=filter_gt_files(xxx_dataset.gt_files)
                xxx_dataset.img_files=[]
                images[dataset+'/'+split]=[]
                assert len(xxx_dataset)<=30
                for i in range(len(xxx_dataset)):
                    frame_images,labels,main_file,aux_file,gt_file=xxx_dataset.__get_image__(i)
                    images[dataset+'/'+split].append(frame_images[0])
                    images[dataset+'/'+split].append(labels[0])


        for split in ['train','val']:
            if split=='train':
                N=58
            else:
                N=60

            assert N==len(images['FBMS/'+split]),'len {} is {}'.format(split,len(images['FBMS/'+split]))
            for i in range(0,N,2):
                x1=images['FBMS/'+split][i]
                y1=images['FBMS/'+split][i+1]
                x2=images['FBMS-3D/'+split][i]
                y2=images['FBMS-3D/'+split][i+1]
                if np.sum(x1-x2)!=0 or np.sum(y1-y2)!=0:
                    print(files['FBMS/'+split][i//2])
                    print(files['FBMS-3D/'+split][i//2])
                    show_images(images=[x1,y1,x2,y2],titles=['x1','y1','x2','y2'])

if __name__ == '__main__':
    unittest.main()
