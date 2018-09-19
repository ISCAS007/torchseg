# -*- coding: utf-8 -*-

import numpy as np
from utils.metrics import runningScore,get_scores

class_number=20
size=(20,224,224)
trues=np.random.randint(low=0,high=class_number,size=size)
preds=np.random.randint(low=0,high=class_number,size=size)

confusion_matrix=np.zeros(shape=(class_number,class_number))
for i in range(class_number):
    for j in range(class_number):
        nij=np.logical_and((preds==i).flatten(),(trues==j).flatten())
        confusion_matrix[i,j]=np.sum(nij)
        
#print(trues)
#print(preds)
print(confusion_matrix)
result=1
for i in size:
    result*=i
assert np.sum(confusion_matrix)==result

run_score=runningScore(class_number)
run_score.update(trues,preds)
print(run_score.confusion_matrix.transpose())

hist=confusion_matrix
iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
print(iu)
miou=np.nanmean(iu)
print(miou)

score, class_iou = run_score.get_scores()
for k, v in score.items():
    print(k, v)

score=get_scores(trues,preds)
for k,v in score.items():
    print(k,v)
score=get_scores(preds,trues)
for k,v in score.items():
    print(k,v)
    
print(class_iou)