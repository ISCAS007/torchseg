# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
def cos_loss(input,target):
    return 1-torch.mean(F.cosine_similarity(input,target))

class CenterLoss(nn.Module):
    def __init__(self,channels,num_classes,ignore_index=-1,loss_fn='cos_loss'):
        super().__init__()
        self.channels=channels
        self.num_classes=num_classes
        self.ignore_index=ignore_index
        
        if loss_fn=='cos_loss':
            self.loss_fn=cos_loss
        elif loss_fn=='l1_loss':
            self.loss_fn=F.l1_loss
        elif loss_fn=='l2_loss':
            self.loss_fn=F.l2_loss
        else:
            assert False,'unknown loss function {}'.format(loss_fn)
        
        # self.centers = nn.Parameter(torch.randn(channels,num_classes))
        self.centers = nn.Parameter(torch.randn(num_classes,channels))
        
    def forward(self,feature,label):
        """
        feature: [batch_size,channels,height,width]
        label: [batch_size,height*N,widht*N], N in {1,2,4,8,16,32}
        self.center: [num_classes,channels]
        
        1. use centered_feature to update self.center
        2. shrink label to feature map size
        """

        if len(label.shape)==3:
            label=label.unsqueeze(1)
        size=feature.shape[-2:]
        label=F.interpolate(label.float(),size=size,mode='nearest',align_corners=None).long()
        valid_indexs=torch.nonzero((label!=self.ignore_index).flatten()).squeeze()
        label=label.flatten()[valid_indexs]
        feature_2d=feature.permute([0,2,3,1]).reshape([-1,self.channels])        
        feature_2d=feature_2d.index_select(0,valid_indexs)
        
        # fast version for cos_loss only
        added_label=torch.unique(label)
        sum_feature_2d=torch.zeros_like(self.centers)
        sum_feature_2d.index_add_(0,label,feature_2d)
        sum_feature_2d=sum_feature_2d[added_label]
        
        count=torch.bincount(label)[added_label]
        count=count.reshape(-1,1).float()
        assert sum_feature_2d.size(0)==count.size(0),'sum_feature_2d.shape={},count.shape={}'.format(sum_feature_2d,count)
        sum_feature_2d.div_(count)
        center2d=self.centers[added_label]
        return self.loss_fn(sum_feature_2d,center2d)
        
        # slow version
        valid_center_2d=self.centers.index_select(0,label)
        return self.loss_fn(feature_2d,valid_center_2d)
        # cos_loss
#        cross_value=feature_2d.matmul(self.centers)
#        norm_value=feature_2d.norm(p=2,dim=1,keepdim=True).matmul(self.centers.norm(p=2,dim=0,keepdim=True))
#        assert cross_value.shape==norm_value.shape,'{} != {}'.format(cross_value.shape,norm_value.shape)
#        cos_value=1-cross_value/norm_value
#        
#        valid_indexs=label.flatten()[valid_mask]
#        valid_cos_value=cos_value[valid_mask][valid_indexs]
#        return torch.mean(valid_cos_value)