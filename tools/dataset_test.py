# -*- coding: utf-8 -*-
"""
python -m pytest -q test/dataset_test.py
"""
from torchseg.dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from torchseg.utils.disc_tools import show_tensor_list,show_images,add_color_map
from easydict import EasyDict as edict
from PIL import Image
import torch.utils.data as TD
import numpy as np
import matplotlib.pyplot as plt
    
def test_dataset():
    config=edict()
    config.dataset_name='Cityscapes_Category'
    config.print_path=True
    config.resize_shape=(224,224)
    config.ignore_index=255
    config.with_edge=False
    config.batch_size=2
    config=get_dataset_generalize_config(config,config.dataset_name)
    
    augmentations=None
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
if __name__ == '__main__':
    test_dataset()