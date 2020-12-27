# -*- coding: utf-8 -*-
import unittest

from torchseg.dataset.dataset_generalize import dataset_generalize, \
    get_dataset_generalize_config
from torchseg.utils.disc_tools import show_images,add_color_map
from torchseg.dataset.labels_cityscapes import id2catId
from PIL import Image
import numpy as np

class Test(unittest.TestCase):
    
    def test_cityscapes_category(self):
        """
        the cityscapes category id != huawei class index, need remap them
        
        and the cityscape dataset may ignore some object like parking
        (class id=9, category=flat ...)
        
        Returns
        -------
        None.

        """
        config=get_dataset_generalize_config(None,'Cityscapes_Category')
        config.with_path=True
        cat_val_dataset = dataset_generalize(config,
                                     split='val',
                                     augmentations=None,
                                     normalizations=None)
        
        config=get_dataset_generalize_config(None,'Cityscapes_Fine')
        config.with_path=True
        class_val_dataset = dataset_generalize(config,
                                     split='val',
                                     augmentations=None,
                                     normalizations=None)
        
        assert len(cat_val_dataset)==len(class_val_dataset)
        N=len(cat_val_dataset)
        
        for key,value in id2catId.items():
            print('key={}, value={}'.format(key,value))
            
        for idx in range(min(N,3)):
            cat_data=cat_val_dataset.__getitem__(idx)
            cat_img,cat_ann=cat_data['image']
            class_data=class_val_dataset.__getitem__(idx)
            class_img,class_ann=class_data['image']
            
            assert cat_data['filename'][1]==class_data['filename'][1]
            origin_ann=Image.open(class_data['filename'][1])
            origin_ann=np.array(origin_ann,np.uint8)
            
            class2cat=origin_ann.copy()
            for key,value in id2catId.items():
                if key>=0:
                    class2cat[origin_ann==key]=value
                    
            for class_id in config.foreground_class_ids:
                print('class_id={},catId={}'.format(class_id,id2catId[class_id]))
            
            class2cat[origin_ann==0]=255
            class2cat[origin_ann!=255]-=1
            
            diff=(class2cat!=cat_ann)
            negative=(origin_ann==255)
            print(np.unique(origin_ann[diff]))
            cat_img=cat_img.transpose((1,2,0))
            class_img=class_img.transpose((1,2,0))
            cat_ann=add_color_map(cat_ann)
            class_ann=add_color_map(class_ann)
            class2cat=add_color_map(class2cat)
            show_images([cat_img,cat_ann,class_img,class_ann,class2cat,diff,negative],['cat','cat','class','class','class2cat','diff','negative'])
    
if __name__ == '__main__':
    unittest.main()