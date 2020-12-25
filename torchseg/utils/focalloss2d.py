####################################################
##### This is focal loss class for multi class #####
##### University of Tokyo Doi Kento            #####
####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
# I refered https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss2d(nn.Module):

    def __init__(self, alpha=1.0, gamma=0, weight=None,ignore_index=-100, size_average=True, with_grad=True):
        super(FocalLoss2d, self).__init__()
        assert gamma>=0.0 and gamma<=5.0,'gamma in [0,5] is okay, but %0.2f'%gamma
        assert alpha>0.0
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index=ignore_index
        self.with_grad=with_grad

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood
        logpt = -F.cross_entropy(input, target,
                                 weight=self.weight, 
                                 ignore_index=self.ignore_index,
                                 reduction='none')
        if self.with_grad:
            pt = torch.exp(logpt)
        else:
            with torch.no_grad():
                pt = torch.exp(logpt)
        # compute the loss
        loss = - self.alpha* ((1-pt)**self.gamma) * logpt
        
        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

if __name__ == '__main__':
    input=torch.rand(2,10,3,3).float()
    print(input.shape)
    target=torch.rand(2,10,3,3)
    target=torch.argmax(target,dim=1)
    loss_fn=FocalLoss2d()
    loss=loss_fn(input,target)
    print(loss)