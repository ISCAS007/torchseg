# -*- coding: utf-8 -*-
"""
get image path from txt file, then do it!
➜  ADEChallengeData2016 tree -L 2 .
.
├── annotations
│   ├── training
│   ├── train.txt
│   ├── validation
│   └── val.txt
├── images
│   ├── training
│   ├── train.txt
│   ├── validation
│   └── val.txt
├── objectInfo150.txt
└── sceneCategories.txt

cityscapes:
    num_classes=19
    ignore_index=255
pascal
    num_classes=21
    ignore_index=255
ade20k
    num_classes=151
    ignore_index=0
"""
import torch.utils.data as TD
import os
import cv2
import numpy as np
from easydict import EasyDict as edict
from PIL import Image


def get_dataset_generalize_config(config, dataset_name):
    support_datasets = ['ADEChallengeData2016', 'VOC2012', 'Kitti2015',
                        'Cityscapes', 'Cityscapes_Fine', 'Cityscapes_Coarse', 'ADE20K']
    assert dataset_name in support_datasets, 'unknown dataset %s, not in support dataset %s' % (
        dataset_name, str(support_datasets))
    if dataset_name == 'ADEChallengeData2016':
        # train + val, no test
        config.root_path = '/media/sdb/CVDataset/ObjectSegmentation/ADEChallengeData2016'
        assert os.path.exists(config.root_path),'dataset path %s not exist!'%config.root_path
        config.image_txt_path = os.path.join(config.root_path, 'images')
        config.annotation_txt_path = os.path.join(
            config.root_path, 'annotations')
        config.foreground_class_ids = [i for i in range(1, 151)]
        config.ignore_index = 0
    elif dataset_name == 'ADE20K':
        assert False, 'the ADE20K dataset luck some of detail'
        # train + val
        config.root_path = '/media/sdb/CVDataset/ObjectSegmentation/ADE20K_2016_07_26/images'
        assert os.path.exists(config.root_path),'dataset path %s not exist!'%config.root_path
        config.txt_note = 'ade20k'
        config.txt_path=os.path.join(os.getcwd(),'dataset','txt')
        assert os.path.exists(config.txt_path),'txt path %s not exist!'%config.txt_path
#        config.foreground_class_ids=[i for i in range(20)]
#        config.ignore_index=255
    elif dataset_name == 'VOC2012':
        # train + val, no test
#        config.root_path = '/media/sdb/CVDataset/VOC'
        config.root_path = os.path.expanduser('~/.cv/datasets/VOC')
        assert os.path.exists(config.root_path),'dataset path %s not exist!'%config.root_path
        config.txt_note = 'voc2012'
        config.txt_path=os.path.join(os.getcwd(),'dataset','txt')
        assert os.path.exists(config.txt_path),'txt path %s not exist!'%config.txt_path
#        config.txt_path = '/home/yzbx/git/torchseg/dataset/txt'
        config.foreground_class_ids = [i for i in range(21)]
        config.ignore_index = 255
    elif dataset_name in ['Cityscapes', 'Cityscapes_Fine']:
        # train + val + test
        config.root_path = os.path.expanduser('~/.cv/datasets/Cityscapes')
        assert os.path.exists(config.root_path),'dataset path %s not exist!'%config.root_path
        config.txt_note = 'cityscapes_fine'
        config.txt_path=os.path.join(os.getcwd(),'dataset','txt')
        assert os.path.exists(config.txt_path),'txt path %s not exist!'%config.txt_path
        config.foreground_class_ids = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        config.ignore_index = 255
    elif dataset_name == 'Cityscapes_Coarse':
        # train + val + train_extra
        config.root_path = '/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
        assert os.path.exists(config.root_path),'dataset path %s not exist!'%config.root_path
        config.txt_note = 'cityscapes_coarse'
        config.txt_path=os.path.join(os.getcwd(),'dataset','txt')
        assert os.path.exists(config.txt_path),'txt path %s not exist!'%config.txt_path
        config.foreground_class_ids = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        config.ignore_index = 255
    else:
        assert False, 'Not Implement for dataset %s' % dataset_name

    return config


class dataset_generalize(TD.Dataset):
    def __init__(self, config, augmentations=None, split=None, bchw=True, normalizations=None):
        self.config = config
        self.augmentations = augmentations
        self.bchw = bchw
        self.split = split
        self.normalizations = normalizations
        self.step = 0

        splits = ['train', 'val', 'test', 'train_extra']
        assert self.split in splits, 'unexcepted split %s for dataset, must be one of %s' % (
            self.split, str(splits))

        if hasattr(self.config, 'txt_note'):
            self.split = self.config.txt_note+'_'+self.split

        if hasattr(self.config, 'txt_path'):
            txt_file = os.path.join(config.txt_path, self.split+'.txt')
            self.image_files, self.annotation_files = self.get_files_from_txt(
                txt_file, self.config.root_path)
            assert len(self.image_files) > 0, 'No files found in %s with %s' % (
                self.config.root_path, txt_file)
            assert len(self.annotation_files) > 0, 'No files found in %s with %s' % (
                self.config.root_path, txt_file)
        else:
            assert hasattr(
                self.config, 'image_txt_path'), 'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            assert hasattr(
                self.config, 'annotation_txt_path'), 'image_txt_path and annotation_txt_path needed when txt_path not offered!'
            image_txt_file = os.path.join(
                config.image_txt_path, self.split+'.txt')
            annotation_txt_file = os.path.join(
                config.annotation_txt_path, self.split+'.txt')
            self.image_files = self.get_files_from_txt(
                image_txt_file, self.config.root_path)
            self.annotation_files = self.get_files_from_txt(
                annotation_txt_file, self.config.root_path)
            assert len(self.image_files) > 0, 'No files found in %s with %s' % (
                self.config.root_path, image_txt_file)
            assert len(self.annotation_files) > 0, 'No files found in %s with %s' % (
                self.config.root_path, annotation_txt_file)

        self.foreground_class_ids = self.config.foreground_class_ids
        self.n_classes = len(self.foreground_class_ids)+1
        if hasattr(self.config, 'ignore_index'):
            self.ignore_index = self.config.ignore_index
        else:
            self.ignore_index = 0

        print("Found %d image files, %d annotation files" %
              (len(self.image_files), len(self.annotation_files)))
        assert len(self.image_files) == len(self.annotation_files)

    @staticmethod
    def get_files_from_txt(txt_file, root_path):
        with open(txt_file, 'r') as f:
            files = [i.strip() for i in f.readlines()]
            if ' ' in files[0]:
                image_files = []
                annotation_files = []
                for line in files:
                    strs = line.split(' ')
                    image_files.append(os.path.join(root_path, strs[0]))
                    annotation_files.append(os.path.join(root_path, strs[1]))

                return image_files, annotation_files
            else:
                files = [os.path.join(root_path, file) for file in files]
                return files

    def __len__(self):
        """__len__"""
        return len(self.image_files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        self.step = self.step+1
        # eg root_path/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png
        img_path = self.image_files[index]
        # eg root_path/gtFine_trainvaltest/gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png
        lbl_path = self.annotation_files[index]

        if hasattr(self.config, 'print_path'):
            if self.config.print_path:
                print('image path:', img_path)
                print('label path:', lbl_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#        lbl = cv2.imread(lbl_path,cv2.IMREAD_GRAYSCALE)
        lbl_pil = Image.open(lbl_path)
        lbl = np.array(lbl_pil, dtype=np.uint8)
        assert img is not None, 'empty image for path %s' % img_path

        ann = np.zeros_like(lbl)+self.ignore_index

#        lbl_ids = np.unique(lbl)
#        print('label image ids',lbl_ids)
        if self.ignore_index == 0:
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx+1
        else:
            assert self.ignore_index not in self.foreground_class_ids, 'ignore_index cannot in foregournd_class_ids if not 0'
            for idx, class_id in enumerate(self.foreground_class_ids):
                ann[lbl == class_id] = idx

        if self.augmentations is not None and self.split == 'train':
            if hasattr(self.config, 'augmentations_blur'):
                if self.config.augmentations_blur:
                    img = self.augmentations.transform(img)
            else:
                img = self.augmentations.transform(img)

            img, ann = self.augmentations.transform(img, ann)
#            print('augmentation',img.shape,ann.shape)

            assert hasattr(
                self.config, 'resize_shape'), 'augmentations may change image to random size by random crop'

        if hasattr(self.config, 'resize_shape'):
            assert len(self.config.resize_shape) == 2, 'resize_shape should with len of 2 but %d' % len(
                self.config.resize_shape)
            img = cv2.resize(src=img, dsize=tuple(
                self.config.resize_shape), interpolation=cv2.INTER_LINEAR)
            ann = cv2.resize(src=ann, dsize=tuple(
                self.config.resize_shape), interpolation=cv2.INTER_NEAREST)

#        print('bgr',np.min(img),np.max(img),np.mean(img),np.std(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        print('rgb',np.min(img),np.max(img),np.mean(img),np.std(img))
        if self.config.with_edge:
            edge_img=None
            if hasattr(self.config,'edge_with_gray'):
                if self.config.edge_with_gray:
                    edge_img=img
            edge = self.get_edge(
                ann_img=ann, edge_width=self.config.edge_width, img=edge_img)
    
        if self.normalizations is not None:
            img = self.normalizations.forward(img)
        
        if self.bchw:
            # convert image from (height,width,channel) to (channel,height,width)
            img = img.transpose((2, 0, 1))
        
        if self.config.with_edge:
            return img, ann, edge
            
        if hasattr(self.config, 'with_path'):
            return {'image': (img, ann), 'filename': (img_path, lbl_path)}

        return img, ann

    def get_edge(self, ann_img, edge_width=5, img=None):
        if hasattr(self.config, 'edge_class_num'):
            edge_class_num = self.config.edge_class_num
        else:
            edge_class_num = 2
        
        assert edge_class_num>=2,'edge class number %d must > 2'%edge_class_num
        kernel = np.ones((edge_width, edge_width), np.uint8)
        if img is None:
            ann_edge = cv2.Canny(ann_img, 0, 1)
        else:
            ann_edge = cv2.Canny(ann_img, 0, 1)
#            print(type(img),img.shape)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            edge=cv2.Canny(gray,100,200)
            ann_edge=edge+ann_edge
        # remove ignore area in ann_img
        ann_edge[ann_img==self.ignore_index]=0
        
        ann_dilation = cv2.dilate(ann_edge, kernel, iterations=1)
        if edge_class_num == 2:
            # fg=0, bg=1
            edge_label = (ann_dilation == 0).astype(np.uint8)
        else:
            # fg=0, bg=1,2,...edge_class_num-1
            edge_label = np.zeros_like(ann_img)+edge_class_num-1
            for class_num in range(edge_class_num-1):
                edge_label[np.logical_and(ann_dilation>0,edge_label==(edge_class_num-1))]=class_num
                ann_dilation = cv2.dilate(ann_dilation, kernel, iterations=1)
                
        # remove ignore area in ann_img
        edge_label[ann_img==self.ignore_index]=self.ignore_index
        return edge_label


class image_normalizations():
    def __init__(self, ways='caffe'):
        if ways == 'caffe(255-mean)' or ways == 'caffe' or ways.lower() in ['voc', 'voc2012']:
            scale = 1.0
            mean_rgb = [123.68, 116.779, 103.939]
            std_rgb = [1.0, 1.0, 1.0]
        elif ways.lower() in ['cityscapes', 'cityscape']:
            scale = 1.0
            mean_rgb = [72.39239876, 82.90891754, 73.15835921]
            std_rgb = [1.0, 1.0, 1.0]
        elif ways == 'pytorch(1.0-mean)/std' or ways == 'pytorch':
            scale = 255.0
            mean_rgb = [0.485, 0.456, 0.406]
            std_rgb = [0.229, 0.224, 0.225]
        elif ways == 'common(-1,1)' or ways == 'common':
            scale = 255.0
            mean_rgb = [0.5, 0.5, 0.5]
            std_rgb = [0.5, 0.5, 0.5]
        else:
            assert False, 'unknown ways %s' % ways

        self.mean_rgb = mean_rgb
        self.std_rgb = std_rgb
        self.scale = scale

    def forward(self, img_rgb):
        x = img_rgb/self.scale
        for i in range(3):
            x[:, :, i] = (x[:, :, i]-self.mean_rgb[i])/self.std_rgb[i]

        return x

    def backward(self, x_rgb):
        x = np.zeros_like(x_rgb)
        if x.ndim == 3:
            for i in range(3):
                x[:, :, i] = x_rgb[:, :, i]*self.std_rgb[i]+self.mean_rgb[i]

            x = x*self.scale
            return x
        elif x.ndim == 4:
            for i in range(3):
                x[:, :, :, i] = x_rgb[:, :, :, i] * \
                    self.std_rgb[i]+self.mean_rgb[i]

            x = x*self.scale
            return x
        else:
            assert False, 'unexpected input dim %d' % x.ndim


if __name__ == '__main__':
    config = edict()
    config.norm = True
#    dataset_name='ADEChallengeData2016'
#    dataset_name='VOC2012'
    dataset_name = 'Cityscapes_Fine'
    config = get_dataset_generalize_config(config, dataset_name)

    dataset = dataset_generalize(config, split='val')
#    for idx,image_file in enumerate(dataset.image_files):
#        print(idx,image_file)
#        img=cv2.imread(image_file,cv2.IMREAD_COLOR)
#        plt.imshow(img)
#        plt.show()
#        if idx>5:
#            break

#    for idx,annotation_file in enumerate(dataset.annotation_files):
# print(idx,annotation_file)
# ann=cv2.imread(annotation_file,cv2.IMREAD_GRAYSCALE)
#        ann=Image.open(annotation_file)
#        ann=np.array(ann, dtype=np.uint8)
#        print(idx,ann.shape)
#        print(np.unique(ann))
# plt.imshow(ann)
# plt.show()
#        if idx>5:
#            break

    print('test norm'+'*'*50)
    for i in range(50):
        img, ann = dataset.__getitem__(i)
        print(i, np.min(img), np.max(img), np.mean(img), np.std(img))
        print(i, np.unique(ann))
