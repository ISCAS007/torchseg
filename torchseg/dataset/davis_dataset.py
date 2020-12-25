# -*- coding: utf-8 -*-

import numpy as np
import os
import random
import glob
from easydict import EasyDict as edict
import cv2
from .segtrackv2_dataset import motionseg_dataset

class davis_dataset(motionseg_dataset):
    split_set=['train','val','test-dev','test-challenge']
    split_with_gt_set=['train','val']
    def __init__(self,config,split='train',normalizations=None,augmentations=None,task='unsupervised'):
        super().__init__(config,split,normalizations,augmentations)
        if split not in self.split_set:
            assert False,'unknown split {} for davis'.format(split)

        self.split=split
        if config.dataset.upper()=='DAVIS2017':
            self.year=2017
        elif config.dataset.upper()=='DAVIS2016':
            self.year=2016
        else:
            assert False

        self.neighbor_type = 'gap'
        self.neighbor_gap = self.config.frame_gap
        self.annotation_suffix = '.png'
        self.image_suffix = '.jpg'
        self.imageset_suffix = '.txt'

        self.resolution = '480p'
        self.image_folder = 'JPEGImages'
        self.imageset_folder = 'ImageSets'

        if self.year==2017:
            self.annotation_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        else:
            self.annotation_folder = 'Annotations'

        self.root_path=config.root_path
        self.main_input_path_list=self.get_main_input_path_list(split)

        print('%s dataset size %d'%(split,len(self.main_input_path_list)))
        self.main_input_path_list.sort()
        if self.split in ['train','val']:
            n=len(self.main_input_path_list)
            if n > self.config['use_part_number'] > 0:
                gap=n//self.config['use_part_number']
                self.main_input_path_list=self.main_input_path_list[::gap]
                print('total dataset image %d, use %d'%(n,len(self.main_input_path_list)))

    def __len__(self):
        return len(self.main_input_path_list)

    def get_neighbor_path(self, main_input_path, neighbor_type=None):
        path_info = self.path_parse(main_input_path)
        neighbor_root_path = os.path.join(path_info.root,
                                          path_info.folder,
                                          path_info.resolution,
                                          path_info.category)

        # suffix='.jpg'
        suffix = path_info.suffix
        files = glob.glob(os.path.join(neighbor_root_path, '*'+suffix))
        files.sort()

        if neighbor_type is None:
            neighbor_type=self.neighbor_type

        if neighbor_type == 'gap':
            gap = self.neighbor_gap
            n = len(files)
            assert n > 0
            assert files[0].find('00000'+suffix) >= 0
            assert gap > 0
            filename = files[(path_info.number+gap) % n]
        elif neighbor_type == 'random':
            files.remove(main_input_path)
            filename = random.choice(files)
        elif neighbor_type == 'first':
            # need filter for when main input path is first.
            if main_input_path == files[0]:
                print('warning main input path is first for',main_input_path)

            return files[0]
        else:
            print('unknown dataset neighbor type',
                  neighbor_type)
            assert False

        return filename

    def get_annotation_path(self, main_input_path):
        path_info = self.path_parse(main_input_path)
        annotation_root_path = os.path.join(path_info.root,
                                            self.annotation_folder,  # 'Annotations',
                                            path_info.resolution,
                                            path_info.category)

        # suffix='.png'
        suffix = self.annotation_suffix
        files = glob.glob(os.path.join(annotation_root_path, '*'+suffix))
        files.sort()

        n = len(files)
        assert n > 0
        assert files[0].find('00000'+suffix) >= 0
        filename = files[path_info.number]

        return filename

    def get_result_path(self, save_root, main_input_path):
        path_info = self.path_parse(main_input_path)
        # suffix='.png'
        suffix = self.annotation_suffix
        result_root_path = os.path.join(save_root,
                                   path_info.category)

        os.makedirs(result_root_path,exist_ok=True)
        assert os.path.exists(result_root_path)
        result_path=os.path.join(result_root_path,
                                 path_info.numstr+suffix)
        return result_path

    def get_main_input_path_list(self, train_or_val=None, use_first_frame=True):
        if train_or_val is None:
            train_or_val=self.split
        else:
            assert train_or_val in self.split_set

        year = self.year
        root_path = self.root_path
        txt_file = os.path.join(root_path,
                                self.imageset_folder,  # 'ImageSets'
                                str(year),
                                train_or_val+self.imageset_suffix)

        f = open(txt_file, 'r')
        lines = f.readlines()
        f.close()

        # suffix=.jpg
        suffix = self.image_suffix
        main_input_path_list = []
        for line in lines:
            category = line.strip()
            category_root_path = os.path.join(root_path,
                                              self.image_folder,  # 'JPEGImages'
                                              self.resolution,  # 480p
                                              category)
            category_path_list = glob.glob(
                os.path.join(category_root_path, '*'+suffix))
            assert len(category_path_list) > 0
            if not use_first_frame:
                category_path_list.sort()
                category_path_list.remove(category_path_list[0])

            main_input_path_list.extend(category_path_list)

        return main_input_path_list

    @staticmethod
    def path_parse(main_input_path):
        """
        JPEGImages/480p/bear/00000.jpg
        suffix=.jpg
        """
        sep = os.sep
        path_list = main_input_path.split(sep)
        filename = path_list[-1]
        numstr, suffix = os.path.splitext(filename)
        number = int(numstr)
        path_info = edict({'root': sep.join(path_list[:-4]),
                           # JPEGImages or Annotations
                           'folder': path_list[-4],
                           'resolution': path_list[-3],  # 480p
                           'category': path_list[-2],  # bear,...
                           'filename': path_list[-1],  # 00000.jpg or 00000.png
                           'numstr': numstr,  # 00000
                           'number': number,  # 0
                           'suffix': suffix  # .jpg
                           })

        return path_info

    def __get_path__(self,index):
        main_path=self.main_input_path_list[index]
        aux_path=self.get_neighbor_path(main_path)

        if self.split in self.split_with_gt_set:
            main_gt_path=self.get_annotation_path(main_path)
            return main_path,aux_path,main_gt_path
        else:
            return main_path,aux_path,main_path

    def __get_image__(self,index):
        main_file,aux_file,gt_file=self.__get_path__(index)
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in [main_file,aux_file]]
        if self.split in self.split_with_gt_set:
            gt_image=cv2.imread(gt_file,cv2.IMREAD_GRAYSCALE)

            # find aux_gt_file
            aux_gt_file=self.get_annotation_path(aux_file)
            if os.path.exists(aux_gt_file):
                aux_gt_image=cv2.imread(aux_gt_file,cv2.IMREAD_GRAYSCALE)
            else:
                aux_gt_image=np.zeros_like(gt_image)
        else:
            height,width,_=frame_images[0].shape
            aux_gt_image=gt_image=np.zeros((height,width),dtype=np.uint8)

        labels=[]
        for gt in [gt_image,aux_gt_image]:
            label=np.zeros_like(gt)
            label[gt>0]=1
            labels.append(label)
        return frame_images,labels,main_file,aux_file,gt_file


if __name__ == '__main__':
    cfg = edict()
    cfg.input_shape=(224,224)
    cfg.use_part_number=0
    cfg.frame_gap=5
    cfg.root_path = os.path.expanduser('~/cvdataset/DAVIS')
    cfg.dataset='DAVIS2017'
    cfg.note='test'
    split='test-challenge'
    d = davis_dataset(cfg,split=split,task='unsupervised')

    main_input_path = os.path.expanduser('~/cvdataset/DAVIS/JPEGImages/480p/bear/00000.jpg')
    neighbor_path = d.get_neighbor_path(main_input_path)
    print(neighbor_path)
    annotation_path = d.get_annotation_path(main_input_path)
    print(annotation_path)

    save_dir=os.path.join(os.path.expanduser('~/tmp/result'),cfg.dataset,split,cfg.note)
    result_path=d.get_result_path(save_dir,main_input_path)
    print(result_path)

    main_input_path_list = d.get_main_input_path_list()
    print(len(main_input_path_list))

    for data in d:
        print(data)
        break
