# -*- coding: utf-8 -*-

import numpy as np
from utils.metrics import runningScore,get_scores
import glob
from PIL import Image
import os
from tqdm import tqdm

#class_number=20
#size=(20,224,224)
#trues=np.random.randint(low=0,high=class_number,size=size)
#preds=np.random.randint(low=0,high=class_number,size=size)
#
#confusion_matrix=np.zeros(shape=(class_number,class_number))
#for i in range(class_number):
#    for j in range(class_number):
#        nij=np.logical_and((preds==i).flatten(),(trues==j).flatten())
#        confusion_matrix[i,j]=np.sum(nij)
#        
##print(trues)
##print(preds)
#print(confusion_matrix)
#result=1
#for i in size:
#    result*=i
#assert np.sum(confusion_matrix)==result
#
#run_score=runningScore(class_number)
#run_score.update(trues,preds)
#print(run_score.confusion_matrix.transpose())
#
#hist=confusion_matrix
#iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#print(iu)
#miou=np.nanmean(iu)
#print(miou)
#
#score, class_iou = run_score.get_scores()
#for k, v in score.items():
#    print(k, v)
#
#score=get_scores(trues,preds)
#for k,v in score.items():
#    print(k,v)
#score=get_scores(preds,trues)
#for k,v in score.items():
#    print(k,v)
#    
#print(class_iou)

output_path='output/pspnet/voc2012/voc_val'
label_path='/home/yzbx/.cv/datasets/VOC/VOCdevkit/VOC2012/SegmentationClass'
val_output_files=glob.glob(os.path.join(output_path,'*.png'))
# all label files for train and validation
label_files=glob.glob(os.path.join(label_path,'*.png'))
val_label_files=[]

for f in val_output_files:
    basename=os.path.basename(f)
    val_label_file=os.path.join(label_path,basename)
    assert val_label_file in label_files,'%s %s %s'%(basename,val_label_file,label_files[0])
    val_label_files.append(val_label_file)
    
run_score=runningScore(21)
for output_file,label_file in tqdm(zip(val_output_files,val_label_files)):
    label_img_pil=Image.open(label_file)
    label_img = np.array(label_img_pil, dtype=np.uint8)
    
    output_img_pil=Image.open(output_file)
    output_img=np.array(output_img_pil,dtype=np.uint8)
    run_score.update(label_trues=label_img, label_preds=output_img)
#    run_score.update(label_trues=output_img, label_preds=label_img)

score, class_iou = run_score.get_scores()
for k, v in score.items():
    print(k, v)

labels=['background','aeroplane','bicycle','bird','boat',
                       'bottle','bus','car','cat','chair',
                       'cow','diningtable','dog','horse','motorbike',
                       'person','pottedplant','sheep','sofa','train','tvmonitor']
for key,value in class_iou.items():
    print(labels[key],value)
    
#Overall Acc:     0.7263556328790647
#Mean Acc :       0.05737974393166002
#FreqW Acc :      0.5384941540213117
#Mean IoU :       0.04429854010972365
#background 0.7284185438215062
#aeroplane 0.003393615837147475
#bicycle 0.0006607226193107853
#bird 0.0006303667373707968
#boat 0.0
#bottle 0.00045285143729870754
#bus 0.004160031021474166
#car 0.02181230602534112
#cat 0.028694991462340193
#chair 0.0
#cow 0.013632117832053415
#diningtable 0.005824017097625476
#dog 0.005280279355790445
#horse 0.003775059564897503
#motorbike 0.007420286934063259
#person 0.041363906883305324
#pottedplant 0.008778013951597044
#sheep 0.02580095477190837
#sofa 0.006256871454329566
#train 0.018677358777851087
#tvmonitor 0.005237046718985833