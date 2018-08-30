# -*- coding: utf-8 -*-

import torch
from models.psp_convert import psp_convert,CONFIG
import torchsummary
            
if __name__ == '__main__':
    # VOC2012 Cityscapes ADEChallengeData2016 ADE20K
    dataset_name='ADE20K'
    net=psp_convert(dataset_name,True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size=CONFIG[dataset_name]['input_size']
    torchsummary.summary(net.to(device),(3,input_size[0],input_size[1]))
    
    print('save params'+'*'*30)
    torch.save(net.state_dict(),'psp_convert_%s.pkl'%dataset_name)
    print('load params'+'*'*30)
    net.load_state_dict(torch.load('psp_convert_%s.pkl'%dataset_name))