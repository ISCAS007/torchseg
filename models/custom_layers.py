# -*- coding: utf-8 -*-
"""
backbone.py: custom backbone
custom_layers.py: custom layers
upsample.py: custom layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from utils.disc_tools import lcm_list
from models.dual_attention import CAM_Module,PAM_Module,DUAL_Module

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, main_channel,reduction=16,attention_channel=None):
        super(CALayer, self).__init__()
        if attention_channel is None:
            attention_channel=main_channel
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_c, main_channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.avg_pool(attention_feature)
        y = self.conv(y)
        return main_feature * y

class UpsampleLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, size=self.size,
                          scale_factor=self.scale_factor,
                          mode=self.mode,
                          align_corners=self.align_corners)

        return x

## Spatial Attention (SA) Layer
class SALayer(nn.Module):
    def __init__(self,main_channel,reduction=16,attention_channel=None):
        super().__init__()
        if attention_channel is None:
            attention_channel=main_channel

        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                        nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, 1, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.conv(attention_feature)
        return main_feature * y

## Global Attention (GA) Layer
class GALayer(nn.Module):
    def __init__(self,main_channel,ga_channel=32,reduction=16,attention_channel=None):
        super().__init__()
        if attention_channel is None:
            attention_channel=main_channel

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                        nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, ga_channel, 1, padding=0, bias=True),
                        nn.Sigmoid()
                    )

        self.ga_conv = nn.Sequential(
                        nn.Conv2d(main_channel+ga_channel, main_channel, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature
        y = self.avg_pool(attention_feature)
        y = self.conv(y)
        b,c,h,w=main_feature.shape
        y = y.repeat(1,1,h,w)
        y = torch.cat([main_feature,y],dim=1)
        y = self.ga_conv(y)
        return y

## Global Attention Layer 2, like spatial attention + global attention
class HALayer(nn.Module):
    def __init__(self,main_channel,ga_channel=1,reduction=16,attention_channel=None):
        super().__init__()
        if attention_channel is None:
            attention_channel=main_channel

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        assert main_channel > ga_channel

        mid_c=max(8,main_channel//reduction)
        self.conv = nn.Sequential(
                        nn.Conv2d(main_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, main_channel-ga_channel, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True)
                    )

        mid_c=max(8,attention_channel//reduction)
        self.ga_conv = nn.Sequential(
                        nn.Conv2d(attention_channel, mid_c, 1, padding=0, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(mid_c, ga_channel, 1, padding=0, bias=True),
                        nn.Sigmoid()
                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, main_feature,attention_feature=None):
        if attention_feature is None:
            attention_feature=main_feature

        x1=self.conv(main_feature)
        x2=self.ga_conv(attention_feature)
        y = torch.cat([x1,x2],dim=1)
        return y

class AttentionLayer(nn.Module):
    def __init__(self,config,main_c,filter_c=None,is_lr_layer=False):
        super().__init__()
        self.config=config
        self.is_lr_layer=is_lr_layer

        if hasattr(config,'res_attention'):
            self.res_attention=config.res_attention
        else:
            warnings.warn('not use residual attention default')
            self.res_attention=False

        self.attention_type=config.attention_type
        if filter_c is None:
            filter_c=main_c

        if 's' in self.attention_type:
            self.spatial_filter=SALayer(main_channel=main_c,attention_channel=filter_c)

        if 'c' in self.attention_type:
            self.channel_filter=CALayer(main_channel=main_c,attention_channel=filter_c)

        if 'g' in self.attention_type:
            self.global_filter=GALayer(main_channel=main_c,attention_channel=filter_c)

        if 'h' in self.attention_type:
            self.global2_filter=HALayer(main_channel=main_c,attention_channel=filter_c)

        if 'd' in self.attention_type and self.is_lr_layer:
            self.dual_filter=DUAL_Module(main_c)

        if 'p' in self.attention_type and self.is_lr_layer:
            self.pam_filter=PAM_Module(main_c)

        if 'q' in self.attention_type and self.is_lr_layer:
            self.cam_filter=CAM_Module(main_c)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x,y=None):
        if y is None:
            y=x

        if self.res_attention:
            origin_x=x

        for c in self.attention_type:
            if c=='s':
                x=self.spatial_filter(x,y)
            elif c=='c':
                x=self.channel_filter(x,y)
            elif c=='g':
                x=self.global_filter(x,y)
            elif c=='h':
                x=self.global2_filter(x,y)
            elif (not self.is_lr_layer) and c in ['d','p','q']:
                pass
            elif c=='d':
                x=self.dual_filter(x)
            elif c=='p':
                x=self.pam_filter(x)
            elif c=='q':
                x=self.cam_filter(x)
            elif c=='n':
                pass
            else:
                assert False,'unknonw attention type {}'.format(self.attention_type)

        if self.res_attention and self.attention_type!='n':
            return origin_x+x
        else:
            return x

# apply attention on last layer, psp,...,
class LowResolutionLayer(nn.Module):
    def __init__(self,config,in_c,out_c):
        super().__init__()
        self.config=config

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        return x

def conv_1x1(in_c,out_c,use_bn=False,groups=1):
    if use_bn:
        return nn.Sequential(nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_c,
                                                   groups=groups,
                                                   kernel_size=1),
                                        nn.BatchNorm2d(out_c),
                                        nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_c,
                                                   groups=groups,
                                                   kernel_size=1),
                                        nn.ReLU(inplace=True))

def conv_nxn(in_c,out_c,kernel_size,use_bn=False,groups=1,dilation=1):
    if use_bn:
        return nn.Sequential(nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_c,
                                                   groups=groups,
                                                   dilation=dilation,
                                                   kernel_size=kernel_size,
                                                   padding=dilation*(kernel_size-1)//2),
                                        nn.BatchNorm2d(out_c),
                                        nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_c,
                                                   groups=groups,
                                                   dilation=dilation,
                                                   kernel_size=kernel_size,
                                                   padding=dilation*(kernel_size-1)//2),
                                        nn.ReLU(inplace=True))

# merge all layers
class MergeLayer(nn.Module):
    def __init__(self,config,in_channel_list,output_shape):
        super().__init__()
        self.config=config
        # height,width
        b,out_c,h,w=output_shape
        self.size=(h,w)

        if hasattr(self.config,'use_bn'):
            self.use_bn=self.config.use_bn
        else:
            self.use_bn=False

        # change feature map channel and size
        self.layers=[]
        total_c=0
        for in_c in in_channel_list:
            mid_c=min(in_c,out_c)
            total_c+=mid_c
            self.layers.append(conv_1x1(in_c,mid_c,self.use_bn))

        self.module_list=nn.ModuleList(self.layers)

        self.conv=conv_1x1(total_c,out_c,self.use_bn)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,in_list):
        assert len(in_list)==len(self.layers),'len(in_list)={},len(layers)={}'.format(len(in_list),len(self.layers))
        x=torch.cat([F.interpolate(conv(f),size=self.size,mode='nearest') for conv,f in zip(self.layers,in_list)],dim=1)
        return self.conv(x)

class PSPLayer(nn.Module):
    """x->4x[pool->conv->bn->relu->upsample]->concat
    input_shape[batch_size,channel,height,width]
    height:lcm(pool_sizes)*scale*(2**upsample_ratio)
    width:lcm(pool_sizes)*scale*(2**upsample_ratio)
    lcm: least common multiple, lcm([4,5])=20, lcm([2,3,6])=6

    for feature layer with output size (batch_size,channel,height=x,width=x)
    digraph G {
      "feature[x,x]" -> "pool6[x/6*scale,x/6*scale]" -> "conv_bn_relu6[x/6*scale,x/6*scale]" -> "interp6[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool3[x/3*scale,x/3*scale]" -> "conv_bn_relu3[x/3*scale,x/3*scale]" -> "interp3[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool2[x/2*scale,x/2*scale]" -> "conv_bn_relu2[x/2*scale,x/2*scale]" -> "interp2[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool1[x/1*scale,x/1*scale]" -> "conv_bn_relu1[x/1*scale,x/1*scale]" -> "interp1[x,x]" -> "concat[x,x]"
    }
    """

    def __init__(self, pool_sizes, scale, input_shape, out_channels, use_bn, additional_upsample=True):
        """
        pool_sizes = [1,2,3,6]
        scale = 5,10
        out_channels = the output channel for transform_psp
        out_size = ?
        """
        super().__init__()
        self.input_shape = input_shape
        self.use_bn=use_bn
        self.additional_upsample=additional_upsample
        b, in_channels, height, width = input_shape
        assert width>=height,'current support only width >= height'

        path_out_c_list = []
        N = len(pool_sizes)

        assert out_channels > in_channels, 'out_channels will concat inputh, so out_chanels=%d should >= in_chanels=%d' % (
            out_channels, in_channels)
        # the output channel for pool_paths
        pool_out_channels = out_channels-in_channels

        mean_c = pool_out_channels//N
        for i in range(N-1):
            path_out_c_list.append(mean_c)

        path_out_c_list.append(pool_out_channels+mean_c-mean_c*N)

        self.pool_sizes = pool_sizes
        self.scale = scale
        pool_paths = []

        self.min_input_size=lcm_list(pool_sizes)*scale

        # scale=1, pool_sizes=[15,5,3,1], deconv_layer=5
        # input_shape 480 --> feature 15(deconv_layer=5) --> psp --> [1,3,5,15] --> 15

        # scale=5 pool_sizes=[6,3,2,1] deconv_layer=3
        # input_shape 480 --> feature 60(deconv_layer=3) --> psp --> [6,3,2,1] -->60
        if not self.additional_upsample:
            assert height%self.min_input_size==0 and width%self.min_input_size==0,"height={},min input size={}".format(height,self.min_input_size)

        h_ratio=height//self.min_input_size
        w_ratio=width//self.min_input_size
        if self.additional_upsample:

            self.conv_before_psp=nn.Sequential(conv_1x1(in_channels,in_channels,self.use_bn),
                                               UpsampleLayer(size=(h_ratio*self.min_input_size,w_ratio*self.min_input_size),mode='bilinear',align_corners=True))


        for pool_size, out_c in zip(pool_sizes, path_out_c_list):
            pool_path = nn.Sequential(nn.AvgPool2d(kernel_size=[h_ratio*pool_size*scale,pool_size*scale*w_ratio],
                                                   stride=[pool_size*scale*h_ratio,pool_size*scale*w_ratio],
                                                   padding=0),
                                      conv_1x1(in_channels,out_c,self.use_bn),
                                      conv_1x1(out_c,out_c,self.use_bn),
                                      UpsampleLayer(size=(self.min_input_size*h_ratio, w_ratio*self.min_input_size), mode='nearest'))
            pool_paths.append(pool_path)

        self.pool_paths = nn.ModuleList(pool_paths)

        if self.additional_upsample:
            self.conv_after_psp=nn.Sequential(conv_1x1(pool_out_channels,pool_out_channels,self.use_bn),
                                          UpsampleLayer(size=(height,width),mode='bilinear',align_corners=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        try:
            if self.additional_upsample:
                in_psp_feature=self.conv_before_psp(x)
                out_psp_features=self.conv_after_psp(torch.cat([m(in_psp_feature) for m in self.pool_paths],dim=1))
            else:
                out_psp_features=torch.cat([m(x) for m in self.pool_paths],dim=1)
            x = torch.cat([x,out_psp_features], dim=1)
        except:
            warnings.warn('exception in PSPLayer')
            if self.additional_upsample:
                for m in self.pool_paths:
                    print(m(in_psp_feature).shape)
            else:
                for m in self.pool_paths:
                    print(m(x).shape)

            assert False
        return x

# merge layer cascade
class CascadeMergeLayer(nn.Module):
    def __init__(self,backbone,config):
        super().__init__()
        self.config=config

        self.deconv_layer=config.deconv_layer
        self.upsample_layer=config.upsample_layer
        self.use_none_layer=config.use_none_layer
        self.min_channel_number=config.min_channel_number
        self.max_channel_number=config.max_channel_number
        self.decode_main_layer=config.decode_main_layer
        self.use_bn=config.use_bn
        self.input_shape=config.input_shape

        self.layers=[None for i in range(self.deconv_layer+1)]
        out_c=0
        for idx in range(self.deconv_layer,-1,-1):
            if idx<self.upsample_layer:
                continue
            elif idx==self.deconv_layer:
                merge_c=0
                current_c=2*backbone.get_feature_map_channel(idx)
            else:
                merge_c=out_c
                current_c=backbone.get_feature_map_channel(idx)

            out_c=min(max(backbone.get_feature_map_channel(idx),
                      self.min_channel_number),self.max_channel_number)



            merge_c+=current_c

            current_layer=[conv_1x1(merge_c,out_c,self.use_bn)]

            if idx==0 or (self.use_none_layer and idx>3) or idx==self.upsample_layer:
                pass
            else:
                upsample_shape=backbone.get_output_shape(idx-1,self.input_shape)
                current_layer.append(UpsampleLayer(size=upsample_shape[2:],mode='bilinear',align_corners=True))

            # add more layers in merge module.
            for _ in range(self.decode_main_layer):
                current_layer.append(conv_1x1(out_c,out_c,self.use_bn))

            self.layers[idx]=nn.Sequential(*current_layer)

        # use the last layer (deconv_layer) for psp module
        midnet_out_channels=2*backbone.get_feature_map_channel(self.deconv_layer)
        midnet_input_shape=backbone.get_output_shape(self.deconv_layer,self.input_shape)
        self.psplayer=PSPLayer(config.midnet_pool_sizes,config.midnet_scale,midnet_input_shape,midnet_out_channels,self.use_bn,config.additional_upsample)
        self.model_layers=nn.ModuleList([layer for layer in self.layers if layer is not None])

    def forward(self,features):
        assert isinstance(features,list)
        feature=None
        for idx in range(self.deconv_layer,self.upsample_layer-1,-1):
            if feature is None:
                feature=self.psplayer(features[idx])
            else:
                feature=torch.cat([feature,features[idx]],dim=1)

            feature=self.layers[idx](feature)
        return feature