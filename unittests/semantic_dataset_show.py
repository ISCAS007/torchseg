# -*- coding: utf-8 -*-
"""
python -m pytest -q test/dataset_test.py
"""

import argparse
from torchseg.dataset.dataset_generalize import (dataset_generalize,
                                                 read_ann_file,
                                                 get_dataset_generalize_config,
                                                 support_datasets)
from torchseg.utils.augmentor import get_default_augmentor_config,Augmentations
from torchseg.utils.disc_tools import (show_tensor_list,
                                       show_images,
                                       add_color_map)
from easydict import EasyDict as edict
import torch.utils.data as TD
import numpy as np
import matplotlib.pyplot as plt
import cv2
    
def test_dataset_loader(dataset_name):
    config=edict()
    config.dataset_name=dataset_name
    config.print_path=True
    config.input_shape=(224,224)
    config.ignore_index=255
    config.with_edge=False
    config.batch_size=2
    config=get_dataset_generalize_config(config,config.dataset_name)
    
    config=get_default_augmentor_config(config)
    augmentations=Augmentations(config)
    dataset=dataset_generalize(config,split='train',augmentations=augmentations)
    loader=TD.DataLoader(dataset=dataset,batch_size=config.batch_size, shuffle=True,drop_last=False)
    plt.ion()
    for i, data in enumerate(loader):
        imgs, labels = data
        print(i,imgs.shape,labels.shape)
        
        show_tensor_list([imgs],['img'])
        show_tensor_list([labels],['labels'])
        
        np_labels=labels.data.cpu().numpy()
        print('label id: ',np.unique(np_labels))
        
        image_list=np.split(np_labels,config.batch_size)
        image_list=[np.squeeze(img) for img in image_list]
        image_list=[add_color_map(img) for img in image_list]
        show_images(image_list,['label']*config.batch_size)
        if i>1:
            break
    
    plt.ioff()
    plt.show()
    
def test_dataset(dataset_name):
    config=edict()
    config.dataset_name=dataset_name
    config.with_path=True
    config.input_shape=(512,1024)
    config.max_crop_size=(1024,2048)
    config.aug_library='imgaug'
    
    config=get_dataset_generalize_config(config,config.dataset_name)
    
    config=get_default_augmentor_config(config)
    augmentations=Augmentations(config)
    dataset=dataset_generalize(config,split='train',augmentations=augmentations)
    
    N=len(dataset)
    
    idx=np.random.randint(0,N)
    
    data=dataset.__getitem__(idx)
    img,ann=data['image']
    img=img.transpose((1,2,0))
    ann=add_color_map(ann)
    img_path,ann_path=data['filename']
    
    ori_img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    ori_img=cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    
    ori_ann=read_ann_file(ann_path)
    ori_ann=add_color_map(ori_ann)
    show_images([img,ann,ori_img,ori_ann],['img','ann','origin img','origin ann'])
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser('semantic dataset show')
    parser.add_argument('--dataset_name',
                        choices=support_datasets,
                        default='HuaWei')
    
    parser.add_argument('--test_loader',
                        action='store_true',
                        dest='test_loader',
                        default=False)
    
    args=parser.parse_args()
    if args.test_loader:
        test_dataset_loader(args.dataset_name)
    else:
        test_dataset(args.dataset_name)