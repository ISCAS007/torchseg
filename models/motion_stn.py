# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class motion_stn(nn.Module):

    def __init__(self, nb_ref_imgs=1, output_exp=True):
        """
        nb_ref_imgs: the input image number
        output_exp: output the mask or not
        """
        super().__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4]*(1+self.nb_ref_imgs), conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        
        self.backbone=nn.Sequential(self.conv1,
                                    self.conv2,
                                    self.conv3,
                                    self.conv4,
                                    self.conv5)
        
        self.pose_pred = nn.Sequential(self.conv6,
                                       self.conv7,
                                       nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0))

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4]*(1+self.nb_ref_imgs), upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)
            
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, imgs):
        features=[self.backbone(img) for img in imgs]
        merge_features=torch.cat(features,dim=1)
        pose = self.pose_pred(merge_features)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), self.nb_ref_imgs, 6)
        
        stn_features=[features[0]]
        stn_images=[imgs[0]]
        n=len(features)
        for i in range(n-1):
            theta=pose[:,i,:].view(-1,2,3)
            grid=F.affine_grid(theta,features[i+1].size())
            aux_feature=F.grid_sample(features[i+1],grid)
            stn_features.append(aux_feature)
            
            grid_images=F.affine_grid(theta,imgs[i+1].size())
            aux_images=F.grid_sample(imgs[i+1],grid_images)
            stn_images.append(aux_images)
        
        merge_stn_features=torch.cat(stn_features,dim=1)
        b,_,h,w=imgs[0].shape
        if self.output_exp:
            out_upconv5 = self.upconv5(merge_stn_features)
            out_upconv4 = self.upconv4(out_upconv5)
            out_upconv3 = self.upconv3(out_upconv4)
            out_upconv2 = self.upconv2(out_upconv3)
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:h, 0:w]

            exp_mask4 = torch.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = torch.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = torch.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = torch.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        return {'masks':[exp_mask1,exp_mask2,exp_mask3,exp_mask4],
                'features':stn_features,
                'stn_images':stn_images,
                'pose':pose}
        
class motion_net(nn.Module):

    def __init__(self, nb_ref_imgs=1, output_exp=True):
        """
        nb_ref_imgs: the input image number
        output_exp: output the mask or not
        """
        super().__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(3, conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4]*(1+self.nb_ref_imgs), conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])
        
        self.backbone=nn.Sequential(self.conv1,
                                    self.conv2,
                                    self.conv3,
                                    self.conv4,
                                    self.conv5)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4]*(1+self.nb_ref_imgs), upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, imgs):
        features=[self.backbone(img) for img in imgs]
        merge_features=torch.cat(features,dim=1)
        
        b,_,h,w=imgs[0].shape
        if self.output_exp:
            out_upconv5 = self.upconv5(merge_features)
            out_upconv4 = self.upconv4(out_upconv5)
            out_upconv3 = self.upconv3(out_upconv4)
            out_upconv2 = self.upconv2(out_upconv3)
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:h, 0:w]

            exp_mask4 = torch.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = torch.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = torch.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = torch.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        return {'masks':[exp_mask1,exp_mask2,exp_mask3,exp_mask4],
                'features':features}
        
def stn_loss(features,motion,pose,pose_mask_reg=1.0):
    n=len(features)
    total_loss=0
    for i in range(n-1):
        theta=pose[:,i,:].view(-1,2,3)
        grid=F.affine_grid(theta,features[i+1].size())
        pose_mask=F.grid_sample(torch.ones_like(features[i+1]),grid)
            
#         loss=F.l1_loss(features[0],features[i+1],reduction='none')
        loss=torch.abs(features[0]-features[i+1])
        loss=torch.clamp(loss,min=0,max=2.0)
        total_loss+=torch.mean(loss*(1-motion)*pose_mask)+pose_mask_reg*torch.mean(1-pose_mask)
    return total_loss
