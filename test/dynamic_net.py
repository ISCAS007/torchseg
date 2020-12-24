# -*- coding: utf-8 -*-
"""
modify pytorch model on basic model

example:
    change the channels in classifier to 10 for vgg11
"""
import torchsummary
import torch
import torchvision

net=torchvision.models.vgg11()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torchsummary.summary(net.to(device),(3,224,224))

#Sequential(
#  (0): Linear(in_features=25088, out_features=4096, bias=True)
#  (1): ReLU(inplace)
#  (2): Dropout(p=0.5)
#  (3): Linear(in_features=4096, out_features=4096, bias=True)
#  (4): ReLU(inplace)
#  (5): Dropout(p=0.5)
#  (6): Linear(in_features=4096, out_features=10, bias=True)
#)

in_features=None
out_features=10
for layer in net.classifier:
    if isinstance(layer,torch.nn.Linear):
        if in_features is None:
            in_features=layer.in_features
        else:
            in_features=out_features
            
        layer.__init__(in_features,out_features)
        
    if isinstance(layer,torch.nn.ReLU):
        pass
    
    if isinstance(layer,torch.nn.Dropout):
        layer.p=0.2

torchsummary.summary(net.to(device),(3,224,224))