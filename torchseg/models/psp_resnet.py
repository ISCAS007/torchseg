"""modify version resnet for psp
the detial change can see in get_backbone()
1. the `layer0` from 7x7 to 3x`3x3`
2. the BatchNorm2d momentum for 0.1 to 0.95
3. the dilation in layer3, layer4 change to 2,4
"""
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.utils.model_zoo as model_zoo
import os
from ..utils.disc_tools import str2bool
from typing import Type, Any, Callable, Union, List, Optional
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import warnings

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        momentum: int = 0.1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, momentum=momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, momentum=0.1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        if dilation == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                   padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, modify_resnet_head=None, momentum=0.1,upsample_layer=5,in_channels=3,use_none_layer=True):
        self.inplanes = 64
        self.momentum=momentum
        self.in_channels=in_channels
        self.use_none_layer=use_none_layer
        super(ResNet, self).__init__()

        # for pspnet, the layer1_in_channels=128
        # but from checkpoint, the layer1_in_channels=64, make layer1 unchanged!!!
        self.layer1_in_channels=64
        self.upsample_layer=upsample_layer
        # must set the environment variable before!!!
        if modify_resnet_head is None:
            warnings.warn('use config from os.environ[modify_resnet_head]')
            self.modify_resnet_head=str2bool(os.environ['modify_resnet_head'])
        else:
            self.modify_resnet_head=modify_resnet_head

        if self.modify_resnet_head:
            self.prefix_net = nn.Sequential(self.conv_bn_relu(in_channels=in_channels,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=2,
                                                              padding=1),
                                            self.conv_bn_relu(in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1),
                                            self.conv_bn_relu(in_channels=64,
                                                              out_channels=self.layer1_in_channels,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1),
                                            nn.MaxPool2d(kernel_size=3,
                                                         stride=2,
                                                         padding=1))
        else:
            self.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.bn1=nn.BatchNorm2d(64, momentum=momentum)
            self.relu=nn.ReLU(inplace=True)
            self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.prefix_net = nn.Sequential(self.conv1,
                                            self.bn1,
                                            self.relu,
                                            self.maxpool)

        self.layer1 = self._make_layer(
            block, 64, layers[0], index=1, momentum=momentum)
        self.layer2 = self._make_layer(
            block, 128, layers[1], index=2, stride=2, momentum=momentum)
        self.layer3 = self._make_layer(
            block, 256, layers[2], index=3, stride=2, momentum=momentum)
        self.layer4 = self._make_layer(
            block, 512, layers[3], index=4, stride=2, momentum=momentum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self,state_dict):
        model_dict = self.state_dict()
#        print('model',len(model_dict))
#        print('checkpoint',len(state_dict))
#        for k,v in model_dict.items():
#            if k.startswith('layer1.0'):
#                print(k,v.shape)
#        print('*'*30)
#        for k,v in state_dict.items():
#            if k.startswith('layer1.0'):
#                print(k,v.shape)
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        if not self.modify_resnet_head:
            assert 'conv1.weight' in pretrained_dict
            assert 'bn1.weight' in pretrained_dict

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        super().load_state_dict(model_dict)

    def conv_bn_relu(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        seq = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      bias=bias),
                            nn.BatchNorm2d(num_features=out_channels,
                                           momentum=self.momentum),
                            nn.ReLU(inplace=True))

        return seq

    def _make_layer(self, block, planes, blocks, index=1, stride=1, momentum=0.1):
        if self.use_none_layer is False:
            dilation = 1
        elif index == 1:
            dilation = 1
        elif index == 2:
            dilation = 1
        elif index == 3:
            dilation = 2
        elif index == 4:
            dilation = 4
        else:
            assert False, 'unexpected index=%d' % index

        if index == 1 and self.modify_resnet_head:
            in_channels = self.layer1_in_channels
        else:
            in_channels = self.inplanes

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if dilation != 1 and self.use_none_layer:
                downsample_stride = 1
            else:
                downsample_stride = stride

            downsample = nn.Sequential(
                nn.Conv2d(in_channels, planes * block.expansion,
                          kernel_size=1, stride=downsample_stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=momentum),
            )

        layers = []
        layers.append(block(in_channels, planes, stride,
                            downsample, momentum=momentum, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                momentum=momentum, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x, upsample_layer=None):
        if upsample_layer is None:
            upsample_layer=self.upsample_layer

        x = self.prefix_net(x)
        if upsample_layer==1:
            return x
        x = self.layer1(x)
        if upsample_layer==2:
            return x
        x = self.layer2(x)
        if upsample_layer==3:
            return x
        x = self.layer3(x)
        if upsample_layer==4:
            return x
        x = self.layer4(x)
        if upsample_layer==5:
            return x

        assert False,'upsample_layer=%d, not in [1-5]'%self.upsample_layer
        return x

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    momentum: float,
    upsample_layer: int,
    in_channels: int,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool = False, progress: bool = True,
             momentum: float = 0.1, upsample_layer: int = 5, in_channels: int = 3,
             **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True,
             momentum: float = 0.1, upsample_layer: int = 5, in_channels: int = 3,
             **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels,
                   **kwargs)


def resnet50(pretrained=True,momentum=0.1,upsample_layer=5,in_channels=3,**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels,**kwargs)
    if pretrained and in_channels==3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=True,momentum=0.1,upsample_layer=5,in_channels=3,**kwargs):
    """Constructs a ResNet-101 model.

    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels,**kwargs)
    if pretrained and in_channels==3:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(momentum=0.1,upsample_layer=5,in_channels=3,**kwargs):
    """Constructs a ResNet-152 model.

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], momentum=momentum,upsample_layer=upsample_layer,in_channels=in_channels,**kwargs)
    return model

def get_backbone(momentum):
    res101 = torchvision.models.resnet101()
    layer1, layer2, layer3, layer4 = res101.layer1, res101.layer2, res101.layer3, res101.layer4

    # modify res101 for pspnet
    for n, m in layer1.named_modules():
        if '0.conv1' == n:
            m.in_channels = 128
        elif '0.downsample.0' == n:
            m.in_channels = 128

    # modify BN and ReLU config
    for layer in [layer1, layer2, layer3, layer4]:
        for n, m in layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = momentum

    for n, m in layer3.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)
    for n, m in layer4.named_modules():
        if 'conv2' in n:
            m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        elif 'downsample.0' in n:
            m.stride = (1, 1)

    seq = nn.Sequential(layer1,
                        layer2,
                        layer3,
                        layer4)

    return seq
if __name__ == '__main__':
    net = resnet101(momentum=0.5)
    fack_net=get_backbone(momentum=0.5)

    seq_true=nn.Sequential(net.layer1,net.layer2,net.layer3,net.layer4)
    for a,b in zip(seq_true.modules(),fack_net.modules()):
#        assert type(a)==type(b),'type not equal %s!=%s'%(type(a),type(b))

        if isinstance(a, nn.BatchNorm2d):
            assert a.momentum==b.momentum,'momentum not equal'
        elif isinstance(a,nn.Conv2d):
            assert a.stride==b.stride,'stride not equal'
            assert a.bias==b.bias,'bias not equal'
            assert a.dilation==b.dilation,'dilation not equal'
            assert a.kernel_size==b.kernel_size,'kernel size not equal'
            assert a.padding==b.padding,'padding not equal'
            assert a.in_channels==b.in_channels,'in_channels not equal'
            assert a.out_channels==b.out_channels,'out_channels not equal'
        elif isinstance(a,nn.ReLU):
            assert a.inplace==b.inplace,'inplace not equal'
        elif isinstance(a,nn.Sequential) or isinstance(a,Bottleneck):
            pass
        else:
            print(type(a),a)
