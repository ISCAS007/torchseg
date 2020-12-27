# -*- coding: utf-8 -*-

"""
training without information loss
    - not resize the mask with cv2.INTER_NEAREST
"""

from tqdm import trange
import sys

if '.' not in sys.path:
    sys.path.append('.')
from torchseg.dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from torchseg.utils.survey import dataset_mean, dataset_std
 
def static_dataset(dataset_name,split='train'):
    config=get_dataset_generalize_config(None,dataset_name)
    dataset=dataset_generalize(config,split=split,bchw=False)
    
    N=len(dataset)
    MEAN=dataset_mean()
    for idx in trange(N):
        img,ann=dataset.__getitem__(idx)
        MEAN.update(img)
        
    mean=MEAN.summary()
    
    STD=dataset_std(mean)
    for idx in trange(N):
        img,ann=dataset.__getitem__(idx)
        STD.update(img)
    
    std=STD.summary()
    
    return mean,std

if __name__ == '__main__':
    for dataset_name in ['Cityscapes','HuaWei']:
        for split in ['train','val']:
            mean,std=static_dataset(dataset_name,split)
            
            print(f'{dataset_name} {split} RGB mean={mean} RGB std={std}')