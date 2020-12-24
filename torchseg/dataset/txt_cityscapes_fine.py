# -*- coding: utf-8 -*-
"""
➜  Cityscapes_archives tree -d -L 3 .
.
├── gtCoarse
│   └── gtCoarse
│       ├── train
│       ├── train_extra
│       └── val
├── gtFine_trainvaltest
│   ├── evaluationResults
│   └── gtFine *
│       ├── test
│       ├── train
│       └── val
└── leftImg8bit_trainvaltest
    ├── gtFine_trainvaltest -> ../gtFine_trainvaltest
    └── leftImg8bit *
        ├── test
        ├── train
        └── val
"""
import os
import glob
root_path=os.path.expanduser('~/cvdataset/Cityscapes')
image_path='leftImg8bit_trainvaltest/leftImg8bit'
annotation_path='gtFine_trainvaltest/gtFine'
splits=['train','val','test']

#train glob images 2975
#train glob annotations 8925
#val glob images 500
#val glob annotations 1500
#test glob images 1525
#test glob annotations 4575
for split in splits:
    glob_images=glob.glob(os.path.join(root_path,image_path,split,'*','*leftImg8bit.png'))
    glob_annotations=glob.glob(os.path.join(root_path,annotation_path,split,'*','*labelIds.png'))
    print('%s glob images'%split,len(glob_images))
    print('%s glob annotations'%split,len(glob_annotations))
    
    write_file=open('dataset/txt/cityscapes_fine_'+split+'.txt','w')
    for g_img in glob_images:
        #img_p: eg leftImg8bit_trainvaltest/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png
        #ann_p: eg gtFine_trainvaltest/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png
        img_p=g_img.replace(root_path+'/','')
        #replace will not change img_p
        ann_p=img_p.replace('leftImg8bit_trainvaltest/leftImg8bit','gtFine_trainvaltest/gtFine').replace('leftImg8bit.png','gtFine_labelIds.png')
        assert os.path.join(root_path,img_p) in glob_images,'%s not exist'%img_p
        assert os.path.join(root_path,ann_p) in glob_annotations,'%s not exist'%ann_p
        write_file.write(img_p+' '+ann_p+'\n')
    write_file.close()

