# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
import torch.utils.data as TD
from easydict import EasyDict as edict


class simple_net(nn.Module):
    def __init__(self,class_number):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=3,out_channels=class_number,kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)

def test_loss_fn():
    class_number=5
    ignore_index=255
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    for i in range(100):
        print(i)
        predict=torch.randint(low=0,high=class_number,size=(2,class_number,5,5),requires_grad=True)
        label=torch.randint(low=0,high=class_number,size=(2,5,5))
        
        label_bad=torch.randint(low=0,high=class_number,size=(2,5,5))
        label_bad[1,1,2]=ignore_index
        
        loss=loss_fn(predict,label.long())
        loss.backward()
        loss_bad=loss_fn(predict,label_bad.long())
        loss_bad.backward()
    
    print(label_bad)
    
def test_network():
    class_number=19
    ignore_index=255
    net=simple_net(19)
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr = 0.0001)
    
    config=edict()
    config=get_dataset_generalize_config(config,'Cityscapes')
    config.resize_shape=(224,224)
    
    augmentations=None
    dataset=dataset_generalize(config,split='train',augmentations=augmentations)
    loader=TD.DataLoader(dataset=dataset,batch_size=2, shuffle=True,drop_last=False)
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.train()
    for i, (images, labels) in enumerate(loader):
        images = torch.autograd.Variable(images.to(device).float())
        labels = torch.autograd.Variable(labels.to(device).long())
        
        optimizer.zero_grad()
        outputs = net.forward(images)
#        print('images shape',images.shape)
#        print('output shape',outputs.shape)
#        print('labels shape',labels.shape)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == class_number
        loss = loss_fn(input=outputs, target=labels)
#                    loss=torch.nn.functional.cross_entropy(input=outputs,target=labels,ignore_index=255)

        loss.backward()
        optimizer.step()
        
        print('epoch=%s, loss=%.4f'%(i,loss.data))
        
if __name__ == '__main__':
    test_network()
