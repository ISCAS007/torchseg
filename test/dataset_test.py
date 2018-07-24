# -*- coding: utf-8 -*-
"""
python -m pytest -q test/dataset_test.py
"""
from dataset.cityscapes import cityscapes
from easydict import EasyDict as edict
import torch.utils.data as TD
import matplotlib.pyplot as plt
import random
import numpy as np

def test_ignore_index():
    config=edict()
    config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.cityscapes_split=random.choice(['test','val','train'])
    config.print_path=True
    config.resize_shape=(224,224)
    config.ignore_index=255
    
    fg_ids=[i for i in range(19)]
    
    augmentations=None
    dataset=cityscapes(config,split='train',augmentations=augmentations)
    loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=True,drop_last=False)
    for i, data in enumerate(loader):
        imgs, labels = data
#        print(i,imgs.shape,labels.shape)
        labels_ids=np.unique(labels)
        print(labels_ids)
        for id in labels_ids:
            assert id == config.ignore_index or id in fg_ids,'bad id=%d'%id
#        plt.imshow(imgs[0,0])
#        plt.show()
#        plt.imshow(labels[0])
#        plt.show()
        if i>=10:
            break

if __name__ == '__main__':
    test_ignore_index()