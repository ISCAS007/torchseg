# -*- coding: utf-8 -*-
import torch
from utils import caffe_pb2
import torch.nn as nn
import numpy as np
# resnet with dilation on layer3,layer4
from models.psp_resnet import resnet50,resnet101
from models.upsample import transform_psp_caffe

CONFIG = {
    'VOC2012': 
    {
         'n_classes': 21,
         'input_size': (473, 473),
         'block_config': [3, 4, 23, 3],
         'weight':'/media/sdb/yzbx/weights/pspnet101_VOC2012.caffemodel',
         'params':'/media/sdb/yzbx/weights/psp_convert_VOC2012.pkl',
         'backbone': 'resnet101',
    },

    'Cityscapes': 
    {
         'n_classes': 19,
         'input_size': (713, 713),
         'block_config': [3, 4, 23, 3],
         'weight':'/media/sdb/yzbx/weights/pspnet101_cityscapes.caffemodel',
         'params':'/media/sdb/yzbx/weights/psp_convert_cityscapes.pkl',
         'backbone': 'resnet101',
    },

    'ADEChallengeData2016': 
    {
         'n_classes': 150,
         'input_size': (473, 473),
         'block_config': [3, 4, 6, 3],
         'weight':'/media/sdb/yzbx/weights/pspnet50_ADE20K.caffemodel',
         'params':'/media/sdb/yzbx/weights/psp_convert_ADE20K.pkl',
         'backbone': 'resnet50',
    },
    'ADE20K': 
    {
         'n_classes': 150,
         'input_size': (473, 473),
         'block_config': [3, 4, 6, 3],
         'weight':'/media/sdb/yzbx/weights/pspnet50_ADE20K.caffemodel',
         'params':'/media/sdb/yzbx/weights/psp_convert_ADE20K.pkl',
         'backbone': 'resnet50',
    },
}

class conv_bn_relu(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.seq=nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias),
                                 nn.BatchNorm2d(num_features=out_channels,
                                                eps=1e-05,
                                                momentum=0.1,
                                                affine=True)
                                 )
        self.relu=nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.seq(x)
        x = self.relu(x)
        return x
    
class psp_convert(torch.nn.Module):
    def __init__(self,dataset_name='cityscapes',load_caffe_weight=True):
        super().__init__()
        
        self.name = self.__class__.__name__
        self.block_config = CONFIG[dataset_name]['block_config']
        self.n_classes = CONFIG[dataset_name]['n_classes']
        self.input_size = CONFIG[dataset_name]['input_size']
        self.backbone_name = CONFIG[dataset_name]['backbone']
        
        backbone = globals()[self.backbone_name]()
        
        self.prefix_net=backbone.prefix_net
        # Vanilla Residual Blocks
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        
        # Dilated Residual Blocks
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        # Pyramid Pooling Module
        height_choices=[60,90]
        # find the closest height,widht for feature shape[input_size/8,input_size/8]
        if abs(self.input_size[0]/8-height_choices[0])<abs(self.input_size[0]/8-height_choices[1]):
            height,width=height_choices[0],height_choices[0]
        else:
            height,width=height_choices[1],height_choices[1]
            
        self.pyramid_pooling = transform_psp_caffe(pool_sizes=[6, 3, 2, 1],
                                                   input_shape=[None,2048,height,width])
       
        # Final conv layers
        self.cbr_final = conv_bn_relu(in_channels=4096, 
                                      out_channels=512,
                                      kernel_size=3, 
                                      stride=1, 
                                      dilation=1,
                                      padding=1,
                                      bias=False)
        self.dropout = nn.Dropout2d(p=0.1, inplace=True)
        self.classification = nn.Conv2d(in_channels=512, 
                                        out_channels=self.n_classes, 
                                        kernel_size=1, 
                                        stride=1, 
                                        padding=0,
                                        bias=True)
        
        self.upsample=torch.nn.Upsample(size=self.input_size,
                                        mode='bilinear',
                                        align_corners=True)

        # Auxiliary layers for training
        self.convbnrelu4_aux = conv_bn_relu(in_channels=1024, 
                                            out_channels=256, 
                                            kernel_size=3, 
                                            padding=1, 
                                            stride=1, 
                                            bias=False)
        
        self.aux_cls = nn.Conv2d(in_channels=256, 
                                 out_channels=self.n_classes, 
                                 kernel_size=1, 
                                 stride=1, 
                                 padding=0,
                                 bias=True)
        
        if load_caffe_weight:
            self.load_pretrained_model(CONFIG[dataset_name]['weight'])
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self,x):
        # x with shape H,W
        
        # (H,W) -> (H/4,W/4)
        x=self.prefix_net(x)
        
        # (H/4,W/4) -> (H/8,W/8)
        x=self.layer1(x)
        
        # (H/8,W/8) -> (H/8,W/8)
        x=self.layer2(x)
        x=self.layer3(x)
        
        # Auxiliary layers for training
        x_aux = self.convbnrelu4_aux(x)
        x_aux = self.dropout(x_aux)
        x_aux = self.aux_cls(x_aux)
        
        x=self.layer4(x)
        x=self.pyramid_pooling(x)
        x=self.cbr_final(x)
        x=self.dropout(x)
        x=self.classification(x)
        x=self.upsample(x)
        
        if self.training:
#            return x_aux,x
            return x
        else:
            return x
        
    def load_pretrained_model(self, model_path):
        """
        Load weights from caffemodel w/o caffe dependency
        and plug them in corresponding modules
        """
        # My eyes and my heart both hurt when writing this method

        # Only care about layer_types that have trainable parameters
        ltypes = ['BNData', 'ConvolutionData', 'HoleConvolutionData']

        def _get_layer_params(layer, ltype):

            if ltype == 'BNData':
                gamma = np.array(layer.blobs[0].data)
                beta = np.array(layer.blobs[1].data)
                mean = np.array(layer.blobs[2].data)
                var =  np.array(layer.blobs[3].data)
                return [mean, var, gamma, beta]

            elif ltype in ['ConvolutionData', 'HoleConvolutionData']:
                is_bias = layer.convolution_param.bias_term
                weights = np.array(layer.blobs[0].data)
                bias = []
                if is_bias:
                    bias = np.array(layer.blobs[1].data)
                return [weights, bias]
            
            elif ltype == 'InnerProduct':
                raise Exception("Fully connected layers {}, not supported".format(ltype))

            else:
                raise Exception("Unkown layer type {}".format(ltype))


        net = caffe_pb2.NetParameter()
        with open(model_path, 'rb') as model_file:
            net.MergeFromString(model_file.read())

        # dict formatted as ->  key:<layer_name> :: value:<layer_type>
        layer_types = {}
        # dict formatted as ->  key:<layer_name> :: value:[<list_of_params>]
        layer_params = {}

        for l in net.layer:
            lname = l.name
            ltype = l.type
            if ltype in ltypes:
                print("Processing layer {}".format(lname))
                layer_types[lname] = ltype
                layer_params[lname] = _get_layer_params(l, ltype)

        # Set affine=False for all batchnorm modules
        def _no_affine_bn(module=None):
            if isinstance(module, nn.BatchNorm2d):
                module.affine = False

            if len([m for m in module.children()]) > 0:
                for child in module.children():
                    _no_affine_bn(child)

        #_no_affine_bn(self)


        def _transfer_conv(layer_name, module):
            weights, bias = layer_params[layer_name]
            w_shape = np.array(module.weight.size())
            
            print("CONV {}: Original {} and trans weights {}".format(layer_name,
                                                                  w_shape,
                                                                  weights.shape))

            module.weight.data.copy_(torch.from_numpy(weights).view_as(module.weight))

            if len(bias) != 0:
                b_shape = np.array(module.bias.size())
                print("CONV {}: Original {} and trans bias {}".format(layer_name,
                                                                      b_shape,
                                                                      bias.shape))
                module.bias.data.copy_(torch.from_numpy(bias).view_as(module.bias))


        def _transfer_conv_bn(conv_layer_name, mother_module):
            conv_module = mother_module[0]
            bn_module = mother_module[1]
            
            _transfer_conv(conv_layer_name, conv_module)
            
            mean, var, gamma, beta = layer_params[conv_layer_name+'/bn']
            print("BN {}: Original {} and trans weights {}".format(conv_layer_name,
                                                                   bn_module.running_mean.size(),
                                                                   mean.shape))
            bn_module.running_mean.copy_(torch.from_numpy(mean).view_as(bn_module.running_mean))
            bn_module.running_var.copy_(torch.from_numpy(var).view_as(bn_module.running_var))
            bn_module.weight.data.copy_(torch.from_numpy(gamma).view_as(bn_module.weight))
            bn_module.bias.data.copy_(torch.from_numpy(beta).view_as(bn_module.bias))


        def _transfer_residual(prefix, block):
            block_module, n_layers = block[0], block[1]

            bottleneck = block_module[0]
            bottleneck_conv_bn_dic = {prefix + '_1_1x1_reduce': nn.Sequential(bottleneck.conv1,bottleneck.bn1),
                                      prefix + '_1_3x3': nn.Sequential(bottleneck.conv2,bottleneck.bn2),
                                      prefix + '_1_1x1_proj': bottleneck.downsample,
                                      prefix + '_1_1x1_increase': nn.Sequential(bottleneck.conv3,bottleneck.bn3),}

            for k, v in bottleneck_conv_bn_dic.items():
                _transfer_conv_bn(k, v)

            for layer_idx in range(2, n_layers+1):
                residual_layer = block_module[layer_idx-1]
                residual_conv_bn_dic = {'_'.join(map(str, [prefix, layer_idx, '1x1_reduce'])): nn.Sequential(residual_layer.conv1,residual_layer.bn1),
                                        '_'.join(map(str, [prefix, layer_idx, '3x3'])):  nn.Sequential(residual_layer.conv2,residual_layer.bn2),
                                        '_'.join(map(str, [prefix, layer_idx, '1x1_increase'])): nn.Sequential(residual_layer.conv3,residual_layer.bn3),} 
                
                for k, v in residual_conv_bn_dic.items():
                    _transfer_conv_bn(k, v)


        convbn_layer_mapping = {'conv1_1_3x3_s2': self.prefix_net[0],
                                'conv1_2_3x3': self.prefix_net[1],
                                'conv1_3_3x3': self.prefix_net[2],
                                'conv5_3_pool6_conv': nn.Sequential(self.pyramid_pooling.pool_paths[3][1],self.pyramid_pooling.pool_paths[3][2]), 
                                'conv5_3_pool3_conv': nn.Sequential(self.pyramid_pooling.pool_paths[2][1],self.pyramid_pooling.pool_paths[2][2]),
                                'conv5_3_pool2_conv': nn.Sequential(self.pyramid_pooling.pool_paths[1][1],self.pyramid_pooling.pool_paths[1][2]),
                                'conv5_3_pool1_conv': nn.Sequential(self.pyramid_pooling.pool_paths[0][1],self.pyramid_pooling.pool_paths[0][2]),
                                'conv5_4': self.cbr_final.seq,
                                'conv4_' + str(self.block_config[2]+1): self.convbnrelu4_aux.seq,} # Auxiliary layers for training

        residual_layers = {'conv2': [self.layer1, self.block_config[0]],
                           'conv3': [self.layer2, self.block_config[1]],
                           'conv4': [self.layer3, self.block_config[2]],
                           'conv5': [self.layer4, self.block_config[3]],}
        
        # Transfer weights for all non-residual conv+bn layers
        print('process convbn_layer_mapping','*'*50)
        for k, v in convbn_layer_mapping.items():
            _transfer_conv_bn(k, v)

        # Transfer weights for final non-bn conv layer
        print('process conv6, conv6_1','*'*50)
        _transfer_conv('conv6', self.classification)
        _transfer_conv('conv6_1', self.aux_cls)

        # Transfer weights for all residual layers
        print('process residual_layers','*'*50)
        for k, v in residual_layers.items():
            _transfer_residual(k, v)