# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
B=30
N=40
C=5
x=np.random.rand(B,N,10,C)
x=np.argmax(x,-1)
print(x.shape)
x=torch.from_numpy(x).long()
print(x.shape)
x2=torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(x2.shape)
model = nn.Embedding(10, 3)
y=model(x)
print(y.shape)
y2=model(x2)
print(y2.shape)