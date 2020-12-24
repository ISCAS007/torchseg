# -*- coding: utf-8 -*-
"""
design patter: factory
"""
import torch
import torch.nn.functional as F
import torch.utils.data as td
import numpy as np

from .fbms_dataset import fbms_dataset
from .cdnet_dataset import cdnet_dataset
from .segtrackv2_dataset import segtrackv2_dataset
from .bmcnet_dataset import bmcnet_dataset
from .davis_dataset import davis_dataset
from .dataset_generalize import image_normalizations
from ..utils.augmentor import Augmentations
from ..utils.configs.motionseg_config import get_default_config, dataset_root_dict
from ..utils.disc_tools import show_images

def get_motionseg_dataset(config,split):
    dataset_dict={"fbms":fbms_dataset,
                  "fbms-3d":fbms_dataset,
                  "cdnet2014":cdnet_dataset,
                  "segtrackv2":segtrackv2_dataset,
                  "bmcnet":bmcnet_dataset,
                  "davis2016":davis_dataset,
                  'davis2017':davis_dataset}

    normer=image_normalizations(ways='-1,1')
    augmentations = Augmentations()
    key=config.dataset.lower()
    if key in dataset_dict.keys():
        config.root_path=dataset_root_dict[key]
        xxx_dataset=dataset_dict[key](config,split,normalizations=normer,augmentations=augmentations)
    elif key == "all":
        dataset_set=[]
        for d in ['FBMS','cdnet2014','segtrackv2','BMCnet','DAVIS2017','DAVIS2016']:
            config.dataset=d
            dataset_set.append(get_motionseg_dataset(config,split))
        xxx_dataset=td.ConcatDataset(dataset_set)
    else:
        assert False,'dataset must in {} or all'.format(dataset_dict.keys())

    return xxx_dataset

def prepare_input_output(config,data,device):
    frames=data['images']
    images = [torch.autograd.Variable(img.to(device).float()) for img in frames]
    origin_labels=[torch.autograd.Variable(gt.to(device).long()) for gt in data['labels']]
    resize_labels=[F.interpolate(gt.float(),size=config.input_shape,mode='nearest').long() for gt in origin_labels]

    aux_input = []
    for c in config.input_format:
        if c.lower()=='b':
            assert False
        elif c.lower()=='g':
            origin_aux_gt=origin_labels[1]
            #print(origin_aux_gt.shape)
            ignore_index=255
            origin_aux_gt[origin_aux_gt==ignore_index]=0
            resize_aux_gt=F.interpolate(origin_aux_gt.float(),size=config.input_shape,mode='nearest').float()
            aux_input.append(resize_aux_gt)
        elif c.lower()=='n':
            aux_input.append(images[1])
        elif c.lower()=='o':
            aux_input.append(torch.autograd.Variable(data['optical_flow'].to(device).float()))
        elif c.lower()=='-':
            pass
        else:
            assert False

    if len(aux_input)>0:
        images[1]=torch.autograd.Variable(torch.cat(aux_input,dim=1).to(device).float())

    if config.use_sync_bn:
        images=[img.cuda(config.gpu,non_blocking=True) for img in images]
        origin_labels=[label.cuda(config.gpu,non_blocking=True) for label in origin_labels]
        resize_labels=[label.cuda(config.gpu,non_blocking=True) for label in resize_labels]
    return images,origin_labels,resize_labels

def motionseg_show_images(imgs,labels=[],predict=[]):
    normer=image_normalizations(ways='-1,1')
    for img in imgs:
        print("image shape {}, range in [{},{}]".format(img.shape,np.min(img),np.max(img)))

    for img in labels:
        print("label shape {}, range in [{},{}]".format(img.shape,np.min(img),np.max(img)))

    for img in predict:
        print("predict shape {}, range in [{},{}]".format(img.shape,np.min(img),np.max(img)))

    imgs=[img.transpose((1,2,0)) for img in imgs]
    imgs=[normer.backward(img).astype(np.uint8) for img in imgs]
    labels=[label.squezze() for label in labels]

    show_images(imgs+labels+predict)

if __name__ == '__main__':
    config=get_default_config()
    #keys=dataset_root_dict.keys()
    keys=['fbms','fbms-3d']
    for key in keys:
        for split in ['train','val']:
            config.dataset=key
            d=get_motionseg_dataset(config,split)
            N=len(d)
            idx=np.random.randint(N)
            data=d.__getitem__(idx)
            imgs=[img.transpose((1,2,0)) for img in data['images']]
            gts=data['labels']

            print(imgs[0].shape, gts[0].shape)
            gts=[gt.squeeze() for gt in gts]
            show_images(imgs+gts)
            print('dataset={}, split={}'.format(key,split) + '*'*30)
            print(data['main_path'],data['aux_path'],data['gt_path'])
            for img in imgs:
                print('img range in [{},{}]'.format(np.min(img),np.max(img)))

            for gt in gts:
                print('gt range in {}'.format(np.unique(gt)))
