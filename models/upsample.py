# -*- coding: utf-8 -*-
import torchvision as TV
import torch.nn as TN
import numpy as np
import torch.nn.functional as F
import torch


class upsample_duc(TN.Module):
    def __init__(self, in_channels, out_channels, upsample_ratio, eps=1e-5, momentum=0.1):
        """
        out_channels: class number
        upsample_ratio: 2**upsample_layer
        """
        super(upsample_duc, self).__init__()

        self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels*upsample_ratio*upsample_ratio,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=1,
                                                    bias=False),
                                          TN.BatchNorm2d(
            num_features=out_channels*upsample_ratio*upsample_ratio,
            eps=eps,
            momentum=momentum),
            TN.ReLU(),
            TN.Dropout2d(p=0.1))
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
    def __init__(self, in_channels, out_channels, output_shape, eps=1e-5,momentum=0.1):
        """
        out_channels: class number
        """
        super().__init__()
        self.output_shape = output_shape
        self.conv_bn_relu = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                                    out_channels=512,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    bias=False),
                                          TN.BatchNorm2d(
                                              num_features=512,
                                              eps=eps,
                                              momentum=momentum),
                                          TN.ReLU(),
                                          TN.Dropout2d(p=0.1))

        self.conv = TN.Conv2d(in_channels=512,
                              out_channels=out_channels,
                              kernel_size=1,
                              padding=0,
                              stride=1,
                              bias=False)
        for m in self.modules():
            if isinstance(m, TN.Conv2d):
                TN.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, TN.BatchNorm2d):
                TN.init.constant_(m.weight, 1)
                TN.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv(x)
        x = F.upsample(x, size=self.output_shape, mode='bilinear',align_corners=False)
        return x

class transform_psp(TN.Module):
    def __init__(self, pool_sizes, scale, input_shape, out_channels, eps=1e-5, momentum=0.1):
        """
        pool_sizes = [1,2,3,6]
        scale = 5,10
        out_channels = the output channel for transform_psp
        out_size = ?
        """
        super(transform_psp, self).__init__()
        self.input_shape=input_shape
        b,in_channels,height,width=input_shape
        
        path_out_c_list = []
        N = len(pool_sizes)
        
        assert out_channels>in_channels, 'out_channels will concat inputh, so out_chanels=%d should >= in_chanels=%d'%(out_channels,in_channels)
        # the output channel for pool_paths
        pool_out_channels=out_channels-in_channels
        
        mean_c = pool_out_channels//N
        for i in range(N-1):
            path_out_c_list.append(mean_c)

        path_out_c_list.append(pool_out_channels+mean_c-mean_c*N)

        self.pool_sizes = pool_sizes
        self.scale = scale
        pool_paths = []
        for pool_size, out_c in zip(pool_sizes, path_out_c_list):
            pool_path = TN.Sequential(TN.AvgPool2d(kernel_size=pool_size*scale,
                                                   stride=pool_size*scale,
                                                   padding=0),
                                      TN.Conv2d(in_channels=in_channels,
                                                out_channels=out_c,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False),
                                      TN.BatchNorm2d(num_features=out_c,
                                                     eps=eps, 
                                                     momentum=momentum),
                                      TN.ReLU(),
                                      TN.Upsample(size=(height,width), mode='bilinear', align_corners=False))
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
        in_size=x.shape
        assert in_size[2]>=min_input_size,'psp in size %d should >= %d'%(in_size[2],min_input_size)

        for module in self.pool_paths:
            y = module(x)
            output_slices.append(y)

        x = torch.cat(output_slices, dim=1)
        return x


class transform_global(TN.Module):
    def __init__(self, dilation_sizes, class_number,eps=1e-5,momentum=0.1):
        """
        in_channels=class_number
        out_channels=class_number
        backbone->mid_net->global_net->upsample
        """
        super(transform_global, self).__init__()
        dil_paths = []
        for dilation_size in dilation_sizes:
            # for stride=1, to keep size: 2*padding=dilation*(kernel_size-1)
            seq=TN.Sequential(TN.Conv2d(in_channels=class_number,
                                           out_channels=class_number,
                                           kernel_size=3,
                                           stride=dilation_size,
                                           padding=dilation_size,
                                           bias=False),
                                 TN.BatchNorm2d(
                num_features=class_number,
                eps=eps,
                momentum=momentum),
                TN.ReLU(inplace=False))
                                 
            dil_paths.append(seq)
        self.dil_paths = TN.ModuleList(dil_paths)
        self.conv = TN.Conv2d(in_channels=class_number*(1+len(dilation_sizes)),
                              out_channels=class_number,
                              kernel_size=1,
                              padding=0,
                              bias=False)
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

        if depth == 1:
            path = TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                           out_channels=class_number,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False),
                                 TN.BatchNorm2d(
                num_features=class_number,
                eps=eps,
                momentum=momentum),
                TN.ReLU(inplace=False))
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
                                                   bias=False),
                                         TN.BatchNorm2d(
                                             num_features=(2**i)*class_number),
                                         TN.ReLU(inplace=False),
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
        b,in_channels,h,w=input_shape
        assert h==w,'height=%d and width=%d must equal in input_shape'%(h,w)
        if output_stride not in [8, 16]:
            raise ValueError('output_stride must be either 8 or 16.')
    
        atrous_rates = [1, 6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2*rate for rate in atrous_rates]
            atrous_rates[0] = 1
        
        out_c=out_channels
        self.out_channels=out_channels
        
        atrous_paths=[]
        k_sizes=[1,3,3,3]
        # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        for r,k in zip(atrous_rates,k_sizes):
            path=TN.Sequential(TN.Conv2d(in_channels=in_channels,
                                         out_channels=out_c,
                                         kernel_size=k,
                                         stride=1,
                                         padding=r*(k-1)//2,
                                         dilation=r,
                                         bias=False
                                         ),
                               TN.BatchNorm2d(num_features=out_c,
                                              eps=eps,
                                              momentum=momentum),
                               TN.ReLU())
            atrous_paths.append(path)
        
        self.image_level_path=TN.Sequential(TN.AvgPool2d(kernel_size=h),
                                       TN.Conv2d(in_channels=in_channels,
                                                 out_channels=out_c,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=False),
                                       TN.BatchNorm2d(num_features=out_c,
                                                      eps=eps,
                                                      momentum=momentum),
                                       TN.ReLU(),
                                       TN.Upsample(size=(h,w),
                                                   mode='bilinear',
                                                   align_corners=False))
                            
        self.final_conv=TN.Sequential(TN.Conv2d(in_channels=out_c*5,
                                                 out_channels=out_c,
                                                 kernel_size=1,
                                                 stride=1,
                                                 padding=0,
                                                 bias=False),
                               TN.BatchNorm2d(num_features=out_c,
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
                
    def forward(self,x):
        output_slices = []
        
        for module in self.atrous_paths:
            y = module(x)
            output_slices.append(y)
        
        y=self.image_level_path(x)
        output_slices.append(y)
        
        y=torch.cat(output_slices,dim=1)
        y=self.final_conv(y)
        return y
    
def get_midnet(config,midnet_input_shape,midnet_out_channels):
    if hasattr(config.model,'midnet_name'):
        midnet_name = config.model.midnet_name
    else:
        midnet_name = 'psp'
    
    if hasattr(config.model,'eps'):
        eps=config.model.eps
    else:
        eps=1e-5
    
    if hasattr(config.model,'momentum'):
        momentum=config.model.momentum
    else:
        momentum=0.1
    
    if midnet_name=='psp':
        print('midnet is psp'+'*'*50)
        midnet_pool_sizes=config.model.midnet_pool_sizes
        midnet_scale=config.model.midnet_scale
        midnet=transform_psp(midnet_pool_sizes,
                                  midnet_scale,
                                  midnet_input_shape,
                                  midnet_out_channels,
                                  eps=eps, 
                                  momentum=momentum)
    elif midnet_name=='aspp':
        print('midnet is aspp'+'*'*50)
        output_stride=2**config.model.upsample_layer
        midnet=transform_aspp(output_stride=output_stride,
                                   input_shape=midnet_input_shape,
                                   out_channels=midnet_out_channels,
                                   eps=eps,
                                   momentum=momentum)
    
    return midnet

def get_suffix_net(config,midnet_out_channels,class_number):
    upsample_type = config.model.upsample_type
    upsample_layer = config.model.upsample_layer
    input_shape = config.model.input_shape
    if hasattr(config.model,'eps'):
        eps=config.model.eps
    else:
        eps=1e-5
    
    if hasattr(config.model,'momentum'):
        momentum=config.model.momentum
    else:
        momentum=0.1
        
    if upsample_type=='duc':
        print('upsample is duc'+'*'*50)
        r=2**upsample_layer
        decoder=upsample_duc(midnet_out_channels,class_number,r,eps=eps,momentum=momentum)
    elif upsample_type=='bilinear':
        print('upsample is bilinear'+'*'*50)
        decoder=upsample_bilinear(midnet_out_channels,class_number,input_shape[0:2],eps=eps,momentum=momentum)
    else:
        assert False,'unknown upsample type %s'%upsample_type
        
    return decoder