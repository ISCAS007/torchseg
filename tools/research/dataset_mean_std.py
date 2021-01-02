# -*- coding: utf-8 -*-

"""
training without information loss
    - not resize the mask with cv2.INTER_NEAREST
"""

from tqdm import trange
import sys
from pprint import pprint

if '.' not in sys.path:
    sys.path.append('.')
from torchseg.dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from torchseg.utils.survey import dataset_mean, dataset_std, dataset_class_count
 
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

def class_weight_count(dataset_name,split='train'):
    config=get_dataset_generalize_config(None,dataset_name)
    dataset=dataset_generalize(config,split=split,bchw=False)
    
    N=len(dataset)
    class_number=len(config.foreground_class_ids)
    COUNT=dataset_class_count(class_number)
    for idx in trange(N):
        img,ann=dataset.__getitem__(idx)
        COUNT.update(ann)
        
    count=COUNT.summary()
    return count

if __name__ == '__main__':
    for dataset_name in ['HuaWei']:
        for split in ['train','val']:
            #mean,std=static_dataset(dataset_name,split)
            #print(f'{dataset_name} {split} RGB mean={mean} RGB std={std}')
            
            count=class_weight_count(dataset_name,split)
            print(f'{dataset_name} {split} class count \n')
            pprint(count)