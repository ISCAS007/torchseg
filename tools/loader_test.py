# -*- coding: utf-8 -*-

import torch.utils.data as TD
import random
from ..dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config
from easydict import EasyDict as edict
import matplotlib.pyplot as plt

config=edict()
config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
config.cityscapes_split=random.choice(['test','val','train'])
config.resize_shape=(224,224)
config.print_path=True
config.with_path=True
config=get_dataset_generalize_config(config,'Cityscapes')

dataset=dataset_generalize(config)
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