# -*- coding: utf-8 -*-

"""
➜  Cityscapes_archives tree -d -L 3.
.
├── gtCoarse
│   └── gtCoarse *
│       ├── train 
│       ├── train_extra
│       └── val
├── gtFine_trainvaltest
│   ├── evaluationResults
│   └── gtFine
│       ├── test
│       ├── train
│       └── val
├── leftImg8bit_trainextra
│   └── leftImg8bit
│       └── train_extra *
└── leftImg8bit_trainvaltest
    ├── gtFine_trainvaltest -> ../gtFine_trainvaltest
    └── leftImg8bit
        ├── test
        ├── train *
        └── val *

"""

import os 
import glob

root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
image_path='leftImg8bit_trainvaltest/leftImg8bit'
annotation_path='gtCoarse/gtCoarse'
splits=['train','val','train_extra']

for split in splits:
    if split == 'train_extra':
        image_path='leftImg8bit_trainextra/leftImg8bit'
        annotation_path='gtCoarse/gtCoarse'
    
    #glob_images rootpath/leftImg8bit_trainextra/leftImg8bit/train_extra/augsburg/augsburg_000000_000001_leftImg8bit.png
    #glob_annotations rootpath/gtCoarse/gtCoarse/train_extra/augsburg/augsburg_000000_000001_gtCoarse_labelIds.png
    glob_images=glob.glob(os.path.join(root_path,image_path,split,'*','*leftImg8bit.png'))
    glob_annotations=glob.glob(os.path.join(root_path,annotation_path,split,'*','*labelIds.png'))
    glob_images.sort()
    glob_annotations.sort()
    print('%s glob images'%split,len(glob_images))
    print('%s glob annotations'%split,len(glob_annotations))
    assert len(glob_images)==len(glob_annotations),'image number %d != annotations number %d'%(len(glob_images),len(glob_annotations))
    
    write_file=open('dataset/txt/cityscapes_coarse_'+split+'.txt','w')
    for g_img,g_ann in zip(glob_images,glob_annotations):
        img_p=g_img.replace(root_path+'/','')
        ann_p=g_ann.replace(root_path+'/','')
        img_basename=os.path.basename(img_p)
        ann_basename=os.path.basename(ann_p)
        assert img_basename.replace('leftImg8bit.png','gtCoarse_labelIds.png')==ann_basename,'%s not correpond to %s'%(img_p,ann_p)
        
        write_file.write(img_p+' '+ann_p+'\n')
    write_file.close()