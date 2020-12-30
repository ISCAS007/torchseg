# -*- coding: utf-8 -*-
import torch.nn as TN
import torch.nn.functional as F
import torch
import os
from ..utils.disc_tools import str2bool
from .custom_layers import AttentionLayer
import warnings

class local_bn(TN.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.bn = TN.BatchNorm2d(num_features=num_features,
                                 eps=eps,
                                 momentum=momentum)

        if 'torchseg_use_bn' in os.environ.keys():
            warnings.warn('use bn')
            self.use_bn = str2bool(os.environ['torchseg_use_bn'])
        else:
            self.use_bn=True

        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_bn:
            return self.bn(x)
        else:
            return x


class local_dropout2d(TN.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.dropout = TN.Dropout2d(p=p)
        self.use_dropout = True

        if 'torchseg_use_dropout' in os.environ.keys():
            self.use_dropout = str2bool(os.environ['torchseg_use_dropout'])

    def forward(self, x):
        if self.use_dropout:
            return self.dropout(x)
        else:
            return x


class local_upsample(TN.Module):
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


class conv_bn_relu(TN.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=None,
                 padding_mode='zeros',
                 eps=1e-5,
                 momentum=0.1,
                 inplace=False):
        """
        out_channels: class number
        upsample_ratio: 2**upsample_layer
        """
        super().__init__()
        bias = str2bool(os.environ['torchseg_use_bias']) if bias is None else bias
        self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    stride=stride,
                                                    bias=bias,
                                                    groups=groups,
                                                    padding_mode=padding_mode,
                                                    dilation=dilation),
                                          local_bn(num_features=out_channels,
                                                   eps=eps,
                                                   momentum=momentum),
                                          TN.ReLU(inplace=inplace),
                                          local_dropout2d(p=0.1))
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x


class upsample_duc(TN.Module):
    def __init__(self, in_channels, out_channels, upsample_ratio, eps=1e-5, momentum=0.1):
        """
        out_channels: class number
        upsample_ratio: 2**upsample_layer
        """
        super().__init__()

        self.conv_bn_relu = conv_bn_relu(in_channels=in_channels,
                                         out_channels=out_channels*upsample_ratio*upsample_ratio,
                                         kernel_size=3,
                                         padding=1,
                                         stride=1,
                                         eps=eps,
                                         momentum=momentum)

        self.duc = TN.PixelShuffle(upsample_ratio)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.duc(x)
        
        return x


class upsample_bilinear(TN.Module):
    def __init__(self, in_channels, out_channels, output_shape, eps=1e-5, momentum=0.1):
        """
        out_channels: class number
        """
        super().__init__()
        self.output_shape = output_shape
        self.center_channels=min(512,in_channels)
        self.conv_bn_relu = conv_bn_relu(in_channels=in_channels,
                                         out_channels=self.center_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         eps=eps,
                                         momentum=momentum)

        # for single stream network, the last conv not need bias?
        bias=str2bool(os.environ['torchseg_use_bias'])
        self.conv = TN.Conv2d(in_channels=self.center_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=bias)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    # TODO upsampel feature is self.conv_bn_relu(x) or self.conv(x)
    def forward(self, x, need_upsample_feature=False,need_raw_result=False):
        self.center_feature = x = self.conv_bn_relu(x)
        raw_result = x = self.conv(x)
        x = F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)

        if need_upsample_feature:
            return self.center_feature, x
        elif need_raw_result:
            return raw_result,x
        else:
            return x

class upsample_subclass(TN.Module):
    def __init__(self,in_channels, out_channels, output_shape,use_sigmoid):
        """
        in_channels
        out_channels
        sub_class_number=in_channels//out_channels
        mid_channels=out_channels*sub_class_number
        """
        super().__init__()
        self.output_shape=output_shape
        self.class_number=out_channels
        self.sub_class_number=in_channels//out_channels
        assert self.sub_class_number>1
        self.midnet=conv_bn_relu(in_channels,
                                 self.class_number*self.sub_class_number)

        self.use_sigmoid=use_sigmoid
        self.sigmoid=TN.Sigmoid()

    def forward(self,x):
        x=self.midnet(x)
        b,c,h,w=x.shape
        x=x.permute(0,2,3,1).reshape(b,h,w,self.class_number,-1)
        if self.use_sigmoid:
            x=self.sigmoid(x)
        x=torch.sum(x,dim=-1).permute(0,3,1,2)
        x = F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)
        return x

class upsample_fcn(TN.Module):
    def __init__(self, in_channels, out_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        bias=str2bool(os.environ['torchseg_use_bias'])
        self.conv_seq = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                out_channels=4096,
                                                kernel_size=7,
                                                stride=1,
                                                padding=0,
                                                bias=bias),
                                      TN.ReLU(),
                                      local_dropout2d(p=0.5),
                                      TN.Conv2d(in_channels=4096,
                                                out_channels=4096,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=bias),
                                      TN.ReLU(),
                                      local_dropout2d(p=0.5)
                                      )
        self.conv = TN.Conv2d(in_channels=4096,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=bias)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x, need_upsample_feature=False):
        upsample_feature = x = self.conv_seq(x)
        x = self.conv(x)
        x = F.interpolate(x, size=self.output_shape,
                          mode='bilinear', align_corners=True)

        if need_upsample_feature:
            return upsample_feature, x
        else:
            return x
            
class transform_psp_caffe(TN.Module):
    """x->4x[pool->conv->bn->relu->upsample]->concat
    when input_shape is choose according to scale and pool_sizes
    transform_psp_caffe=transfrom_psp

    for feature layer with output size (batch_size,channel,height=x,width=x)
    digraph G {
      "feature[x,x]" -> "pool6[6,6]" -> "conv_bn_relu6[6,6]" -> "interp6[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool3[3,3]" -> "conv_bn_relu3[3,3]" -> "interp3[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool2[2,2]" -> "conv_bn_relu2[2,2]" -> "interp2[x,x]" -> "concat[x,x]"
      "feature[x,x]" -> "pool1[1,1]" -> "conv_bn_relu1[1,1]" -> "interp1[x,x]" -> "concat[x,x]"
    }
    """

    def __init__(self, pool_sizes, input_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        b, in_channels, height, width = input_shape
        out_c = in_channels//len(pool_sizes)

        pool_paths = []
        for pool_size in pool_sizes:
            pool_path = TN.Sequential(TN.AvgPool2d(kernel_size=(height//pool_size, width//pool_size),
                                                   stride=(
                                                       height//pool_size, width//pool_size),
                                                   padding=0),
                                      TN.Conv2d(in_channels=in_channels,
                                                out_channels=out_c,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False),
                                      local_bn(num_features=out_c,
                                               eps=eps,
                                               momentum=momentum),
                                      TN.ReLU(),
                                      local_upsample(size=(height, width), mode='bilinear', align_corners=True))
            pool_paths.append(pool_path)

        self.pool_paths = TN.ModuleList(pool_paths)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        output_slices = [x]
        for module in self.pool_paths:
            y = module(x)
            output_slices.append(y)

        x = torch.cat(output_slices, dim=1)
        return x


class transform_psp(TN.Module):
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

    def __init__(self, pool_sizes, scale, input_shape, out_channels, eps=1e-5, momentum=0.1):
        """
        pool_sizes = [1,2,3,6]
        scale = 5,10
        out_channels = the output channel for transform_psp
        out_size = ?
        """
        super(transform_psp, self).__init__()
        self.input_shape = input_shape
        b, in_channels, height, width = input_shape

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
        bias=str2bool(os.environ['torchseg_use_bias'])
        for pool_size, out_c in zip(pool_sizes, path_out_c_list):
            pool_path = TN.Sequential(TN.AvgPool2d(kernel_size=pool_size*scale,
                                                   stride=pool_size*scale,
                                                   padding=0),
                                      TN.Conv2d(in_channels=in_channels,
                                                out_channels=out_c,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=bias),
                                      local_bn(num_features=out_c,
                                               eps=eps,
                                               momentum=momentum),
                                      TN.ReLU(),
                                      local_dropout2d(p=0.1),
                                      local_upsample(size=(height, width), mode='bilinear', align_corners=True))
            pool_paths.append(pool_path)

        self.pool_paths = TN.ModuleList(pool_paths)

        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        output_slices = [x]
        min_input_size = max(self.pool_sizes)*self.scale
        in_size = x.shape
        assert in_size[2] >= min_input_size, 'psp in size %d should >= %d' % (
            in_size[2], min_input_size)

        for module in self.pool_paths:
            y = module(x)
            output_slices.append(y)

        x = torch.cat(output_slices, dim=1)
        return x


class transform_global(TN.Module):
    def __init__(self, dilation_sizes, class_number, eps=1e-5, momentum=0.1):
        """
        in_channels=class_number
        out_channels=class_number
        backbone->mid_net->global_net->upsample
        """
        super(transform_global, self).__init__()
        dil_paths = []
        bias=str2bool(os.environ['torchseg_use_bias'])
        for dilation_size in dilation_sizes:
            # for stride=1, to keep size: 2*padding=dilation*(kernel_size-1)
            seq = TN.Sequential(TN.Conv2d(in_channels=class_number,
                                          out_channels=class_number,
                                          kernel_size=3,
                                          stride=dilation_size,
                                          padding=dilation_size,
                                          bias=bias),
                                local_bn(
                num_features=class_number,
                eps=eps,
                momentum=momentum),
                TN.ReLU(),
                local_dropout2d(p=0.1))

            dil_paths.append(seq)
        self.dil_paths = TN.ModuleList(dil_paths)
        self.conv = TN.Conv2d(in_channels=class_number*(1+len(dilation_sizes)),
                              out_channels=class_number,
                              kernel_size=1,
                              padding=0,
                              bias=bias)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        global_features = [x]
        for model in self.dil_paths:
            global_features.append(model(x))

        x = torch.cat(global_features, dim=1)
        x = self.conv(x)
        return x


class transform_dict(TN.Module):
    """
    input [b,c,h,w]
    output [b,dict_vertor_size,h,w]
    assert c in range(dict_vertor_number)
    """

    def __init__(self, dict_vector_number, dict_verctor_length):
        super().__init__()
        self.dict = TN.Embedding(num_embeddings=dict_vector_number,
                                 embedding_dim=dict_verctor_length)

    def forward(self, x):
        # [b,c,h,w] -> [b,h,w,c]
        x = x.permute([0, 2, 3, 1])
        x = x.argmax(-1)
        x = self.dict(x)
        # [b,h,w,c] -> [b,c,h,w]
        x = x.permute([0, 3, 1, 2])

        return x


class transform_fractal(TN.Module):
    def __init__(self, in_channels, depth, class_number, fusion_type='max', do_fusion=False, eps=1e-5, momentum=0.1):
        """
        input [b,in_channels,h,w]
        output [b,class_number,h,w]
        """
        super().__init__()
        self.depth = depth
        self.class_number = class_number
        self.fusion_type = fusion_type
        self.do_fusion = do_fusion
        if do_fusion:
            assert depth > 1, 'fusion can only do once on the top level'

        support_types = ['max', 'mean', 'route']
        assert fusion_type in support_types, 'fusion_type %s not in %s' % (
            fusion_type, support_types)
        assert depth >= 1, 'depth %d cannot less than 1' % depth

        fractal_paths = []
        if fusion_type == 'route':
            raise NotImplementedError

        bias=str2bool(os.environ['torchseg_use_bias'])
        if depth == 1:
            path = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                           out_channels=class_number,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=bias),
                                 local_bn(
                num_features=class_number,
                eps=eps,
                momentum=momentum),
                TN.ReLU())
            fractal_paths.append(path)
        else:
            for i in range(depth):
                if i == 0:
                    path = transform_fractal(
                        in_channels, i+1, class_number, fusion_type)
                else:
                    path = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                   out_channels=(
                                                       2**i)*class_number,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0,
                                                   bias=bias),
                                         local_bn(
                                             num_features=(2**i)*class_number),
                                         TN.ReLU(),
                                         transform_fractal((2**i)*class_number, i, class_number, fusion_type))

                fractal_paths.append(path)
        self.fractal_paths = TN.ModuleList(fractal_paths)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []
        for model in self.fractal_paths:
            y = model(x)
            if type(y) is list:
                outputs.extend(y)
            else:
                outputs.append(y)

        if self.do_fusion:
            if self.fusion_type == 'max':
                result = None
                for o in outputs:
                    if result is None:
                        result = o
                    else:
                        result = torch.max(result, o)
                return result
            elif self.fusion_type == 'mean':
                result = None
                for o in outputs:
                    if result is None:
                        result = o
                    else:
                        result = result+o
                return result/len(outputs)
            elif self.fusion_type == 'route':
                raise NotImplementedError
            else:
                raise Exception('This is unexcepted Error')

        return outputs


class transform_aspp(TN.Module):
    def __init__(self,
                 output_stride,
                 input_shape,
                 out_channels,
                 eps=1e-05,
                 momentum=0.1):
        """
        output_stride=2**upsample_layer
        input_shape=b,c,h,w
        output_shape=b,out_channels,h,w
        """
        super().__init__()
        b, in_channels, h, w = input_shape
        assert h == w, 'height=%d and width=%d must equal in input_shape' % (
            h, w)
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')

        atrous_rates = [1, 6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]
            atrous_rates[0] = 1

        out_c = out_channels
        self.out_channels = out_channels

        atrous_paths = []
        k_sizes = [1, 3, 3, 3]
        bias=str2bool(os.environ['torchseg_use_bias'])
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        for r, k in zip(atrous_rates, k_sizes):
            path = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                           out_channels=out_c,
                                           kernel_size=k,
                                           stride=1,
                                           padding=r*(k-1)//2,
                                           dilation=r,
                                           bias=bias
                                           ),
                                 local_bn(num_features=out_c,
                                          eps=eps,
                                          momentum=momentum),
                                 TN.ReLU())
            atrous_paths.append(path)

        self.image_level_path = TN.Sequential(TN.AvgPool2d(kernel_size=h),
                                              TN.Conv2d(in_channels=in_channels,
                                                        out_channels=out_c,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        bias=bias),
                                              local_bn(num_features=out_c,
                                                       eps=eps,
                                                       momentum=momentum),
                                              TN.ReLU(),
                                              local_upsample(size=(h, w),
                                                             mode='bilinear',
                                                             align_corners=True))

        self.final_conv = TN.Sequential(TN.Conv2d(in_channels=out_c*5,
                                                  out_channels=out_c,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0,
                                                  bias=bias),
                                        local_bn(num_features=out_c,
                                                 eps=eps,
                                                 momentum=momentum),
                                        TN.ReLU())

        self.atrous_paths = TN.ModuleList(atrous_paths)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)

    def forward(self, x):
        output_slices = []

        for module in self.atrous_paths:
            y = module(x)
            output_slices.append(y)

        y = self.image_level_path(x)
        output_slices.append(y)

        y = torch.cat(output_slices, dim=1)
        y = self.final_conv(y)
        return y

class transform_segnet(TN.Module):
    """
    segnet or unet as midnet
    """
    def __init__(self,backbone,config):
        super().__init__()
        self.config=config
        self.layers=[]

        self.concat_layers=[]
        if not hasattr(self.config,'merge_type'):
            self.merge_type='mean'
        else:
            self.merge_type=self.config.merge_type

        in_c=out_c=0
        for idx in range(6):
            if idx<self.config.upsample_layer:
                self.layers.append(None)
                self.concat_layers.append(None)
            elif idx==5:
                in_c=out_c=backbone.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)
                layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                 out_channels=out_c,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1),
                                    conv_bn_relu(in_channels=out_c,
                                                 out_channels=out_c,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0)
                                    )
                self.layers.append(layer)
                if self.merge_type=='concat':
                    self.concat_layers.append(conv_bn_relu(in_channels=2*out_c,
                                                    out_channels=in_c,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
            else:
                in_c=backbone.get_feature_map_channel(idx+1)
                out_c=backbone.get_feature_map_channel(idx)
#                print('idx,in_c,out_c',idx,in_c,out_c)

                if self.config.use_none_layer and idx>3:
                    layer=TN.Sequential(conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                        conv_bn_relu(in_channels=out_c,
                                                     out_channels=out_c,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0))
                else:
                    layer=TN.Sequential(TN.ConvTranspose2d(in_c,in_c,kernel_size=4,stride=2,padding=1,bias=False),
                                        conv_bn_relu(in_channels=in_c,
                                                     out_channels=out_c,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1),
                                        conv_bn_relu(in_channels=out_c,
                                                     out_channels=out_c,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0)
                                    )
                self.layers.append(layer)
                if self.merge_type=='concat':
                    self.concat_layers.append(conv_bn_relu(in_channels=in_c+2*out_c,
                                                    out_channels=in_c,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type

        self.model_layers=TN.ModuleList([layer for layer in self.layers if layer is not None])
        if self.merge_type=='concat':
            self.merge_layers=TN.ModuleList([layer for layer in self.concat_layers if layer is not None])
        else:
            assert self.merge_type=='mean','unknown merge type %s'%self.merge_type

    def single_forward(self,x):
        assert isinstance(x,(list,tuple)),'input for segnet should be list or tuple'
        assert len(x)==6

#        for idx in range(6):
#            print(idx,x[idx].shape)
        for idx in range(5,self.config.upsample_layer-1,-1):
            if idx==5:
                feature=x[idx]
                feature=self.layers[idx](feature)
            else:
                #print(self.layers[idx],feature.shape,x[idx].shape)
                feature=self.layers[idx](feature)
#                print(idx,feature.shape,x[idx].shape)
                feature+=x[idx]

        return feature

    def forward(self,main,aux=None):
        if aux is None:
            return self.single_forward(main)

        for x in [main,aux]:
            assert isinstance(x,(list,tuple)),'input for segnet should be list or tuple'
            assert len(x)==6

        for idx in range(5,self.config.upsample_layer-1,-1):
            if idx==5:
                if self.merge_type=='concat':
                    feature=torch.cat([main[idx],aux[idx]],dim=1)
                    feature=self.concat_layers[idx](feature)
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                    feature=main[idx]+aux[idx]
                feature=self.layers[idx](feature)
            else:
                if self.merge_type=='concat':
                    feature=torch.cat([feature,main[idx],aux[idx]],dim=1)
                    feature=self.concat_layers[idx](feature)
                else:
                    assert self.merge_type=='mean','unknown merge type %s'%self.merge_type
                    feature+=main[idx]+aux[idx]

                feature=self.layers[idx](feature)

        return feature

class GlobalAvgPool2d(TN.Module):
    """ Global Average pooling over last two spatial dimensions. """

    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, input):
        return input.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)

    def __repr__(self):
        return self.__class__.__name__ + '( )'


def get_midnet(config, midnet_input_shape, midnet_out_channels):
    if hasattr(config, 'midnet_name'):
        midnet_name = config.midnet_name
    else:
        midnet_name = 'psp'

    if hasattr(config, 'eps'):
        eps = config.eps
    else:
        eps = 1e-5

    if hasattr(config, 'momentum'):
        momentum = config.momentum
    else:
        momentum = 0.1

    os.environ['torchseg_use_bn'] = str(config.use_bn)
    os.environ['torchseg_use_dropout'] = str(config.use_dropout)
    os.environ['torchseg_use_bias'] = str(config.use_bias)

    print('use_bn',os.environ['torchseg_use_bn'])
    print('use_dropout',os.environ['torchseg_use_dropout'])
    print('use_bias',os.environ['torchseg_use_bias'])

    if midnet_name == 'psp':
        #        print('midnet is psp'+'*'*50)
        midnet_pool_sizes = config.midnet_pool_sizes
        midnet_scale = config.midnet_scale
        midnet = transform_psp(midnet_pool_sizes,
                               midnet_scale,
                               midnet_input_shape,
                               midnet_out_channels,
                               eps=eps,
                               momentum=momentum)
    elif midnet_name == 'aspp':
        #        print('midnet is aspp'+'*'*50)
        output_stride = 2**config.upsample_layer
        midnet = transform_aspp(output_stride=output_stride,
                                input_shape=midnet_input_shape,
                                out_channels=midnet_out_channels,
                                eps=eps,
                                momentum=momentum)

    attention=AttentionLayer(config,midnet_out_channels)
    final_net=TN.Sequential(midnet,attention)
    return final_net


def get_suffix_net(config, midnet_out_channels, class_number, aux=False):
    if aux:
        upsample_type = config.auxnet_type
        upsample_layer = config.auxnet_layer
    else:
        upsample_type = config.upsample_type
        upsample_layer = config.upsample_layer

    input_shape = config.input_shape
    if hasattr(config, 'eps'):
        eps = config.eps
    else:
        eps = 1e-5

    if hasattr(config, 'momentum'):
        momentum = config.momentum
    else:
        momentum = 0.1

    os.environ['torchseg_use_bn'] = str(config.use_bn)
    os.environ['torchseg_use_dropout'] = str(config.use_dropout)
    os.environ['torchseg_use_bias'] = str(config.use_bias)

    if upsample_type == 'duc':
        #        print('upsample is duc'+'*'*50)
        r = 2**3 if config.use_none_layer else 2**upsample_layer
        decoder = upsample_duc(midnet_out_channels,
                               class_number, r, eps=eps, momentum=momentum)
    elif upsample_type == 'bilinear':
        #        print('upsample is bilinear'+'*'*50)
        decoder = upsample_bilinear(
            midnet_out_channels, class_number, input_shape[0:2], eps=eps, momentum=momentum)
    elif upsample_type == 'fcn':
        decoder = upsample_fcn(midnet_out_channels,
                               class_number, input_shape[0:2])
    elif upsample_type == 'subclass':
        use_sigmoid=config.subclass_sigmoid
        decoder = upsample_subclass(midnet_out_channels,class_number,input_shape[0:2],use_sigmoid)
    elif upsample_type == 'lossless':
        # decoder = upsample_lossless(midnet_out_channels,class_number,input_shape[0:2],scale=4)
        # output_shape =[duc_ratio*x for x in input_shape]
        if hasattr(config,'duc_ratio'):
            duc_ratio=config.duc_ratio
        else:
            duc_ratio=4
        
        r = 2**3 if config.use_none_layer else 2**upsample_layer
        decoder = upsample_duc(midnet_out_channels,
                               class_number, r*duc_ratio, eps=eps, momentum=momentum)
    else:
        assert False, 'unknown upsample type %s' % upsample_type

    return decoder
