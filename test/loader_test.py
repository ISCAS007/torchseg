# -*- coding: utf-8 -*-

import torch
from torch.nn import Module,Conv2d
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as TD
import random
from dataset.cityscapes import cityscapes
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

config=edict()
config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
config.cityscapes_split=random.choice(['test','val','train'])
config.resize_shape=(224,224)
config.print_path=True
config.with_path=True


dataset=cityscapes(config)
loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=True,drop_last=False)

for i, data in enumerate(loader):
    print(len(data))
    imgs=data['image']
    paths=data['filename']
    imgs, labels = imgs
    img_path,lbl_path = paths
    print(img_path,lbl_path)
    print(i,imgs.shape,labels.shape)
    plt.imshow(imgs[0,:,:,:].permute([1,2,0]))
    plt.show()
    plt.imshow(labels[0,:,:])
    plt.show()
    if i>=3:
        break