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
import os
import cv2

config=edict()
config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
#config.cityscapes_split=random.choice(['test','val','train'])
config.cityscapes_split='val'
config.resize_shape=(224,224)
config.print_path=False
config.with_path=True


dataset=cityscapes(config)
loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=False,drop_last=False)

save_root='/media/sdc/yzbx/benchmark_output/cityscapes'
os.makedirs(save_root,exist_ok=True)

for i, data in enumerate(loader):
    imgs=data['image']
    paths=data['filename']
    imgs, labels = imgs
    img_path,lbl_path = paths
    print(img_path,lbl_path)
    predicts=labels.data.cpu().numpy()
    for idx,p in enumerate(lbl_path):
        save_p=p.replace(config.root_path,save_root)
        print(save_p)
        save_dir=os.path.dirname(save_p)
        os.makedirs(save_dir,exist_ok=True)
        predict_img=predicts[idx]
        benchmarkable_predict_img=dataset.get_benchmarkable_predict(predict_img)
        flag=cv2.imwrite(save_p,benchmarkable_predict_img)
        assert flag,'cannot save image to %s'%save_p