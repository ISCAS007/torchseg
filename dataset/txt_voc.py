# -*- coding: utf-8 -*-
"""
/media/sdb/CVDataset/VOC
├── benchmark_RELEASE
│   ├── benchmark_code_RELEASE
│   │   ├── cp_src
│   │   └── demo
│   │       ├── datadir
│   │       │   ├── cls
│   │       │   ├── img
│   │       │   └── inst
│   │       ├── indir
│   │       └── outdir
│   └── dataset
│       ├── cls
│       ├── img
│       └── inst
└── VOCdevkit
    ├── VOC2007
    │   ├── Annotations
    │   ├── ImageSets
    │   │   ├── Layout
    │   │   ├── Main
    │   │   └── Segmentation
    │   ├── JPEGImages
    │   ├── SegmentationClass
    │   └── SegmentationObject
    └── VOC2012
        ├── Annotations
        ├── ImageSets
        │   ├── Action
        │   ├── Layout
        │   ├── Main
        │   └── Segmentation *
        ├── JPEGImages *
        ├── SegmentationClass *
        │   └── pre_encoded
        └── SegmentationObject

"""
import os
import glob
root_path='/media/sdb/CVDataset/VOC'
txt_path=os.path.join(root_path,'VOCdevkit/VOC2012/ImageSets/Segmentation')
image_path='VOCdevkit/VOC2012/JPEGImages'
annotation_path='VOCdevkit/VOC2012/SegmentationClass'
txt_files=['train.txt','val.txt','test.txt']

glob_images=glob.glob(os.path.join(root_path,image_path,'*.jpg'))
glob_annotations=glob.glob(os.path.join(root_path,annotation_path,'*.png'))

for txt_file in txt_files:
    filename=os.path.join(txt_path,txt_file)
    with open(filename,'r') as f:
        write_file=open('dataset/txt/voc2012_'+txt_file,'w')
        for line in f.readlines():
            line=line.strip()
            if len(line)==0:
                continue
            img_p=os.path.join(image_path,line+'.jpg')
            ann_p=os.path.join(annotation_path,line+'.png')
            assert os.path.join(root_path,img_p) in glob_images,'%s not exist'%img_p
            if txt_file != 'test.txt':
                assert os.path.join(root_path,ann_p) in glob_annotations,'%s not exist'%ann_p
            write_file.write(img_p+' '+ann_p+'\n')
        write_file.close()