# -*- coding: utf-8 -*-

"""
experiment on different imagenet model
"""

import timm
from pprint import pprint
import random
import torch

from torchseg.utils.torchsummary import summary
model_names = timm.list_models(pretrained=True)
pprint(model_names)

name=random.choice(model_names)
name='resnet50d'
m = timm.create_model(name, pretrained=False)
pprint(name)
pprint(m)

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
m.to(device)
m.eval()
summary(m,(3,224,224))
    
    