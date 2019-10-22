import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from utils.disc_tools import str2bool
import os
import warnings
from models.custom_layers import get_batchnorm
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class NoneLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        else:
            # init for bias and batch norm, kernel weight will load from checkpoint weights
            # wish all modules will be overwrite by checkpoint weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
#                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def load_state_dict(self,state_dict):
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        super().load_state_dict(model_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False,group_norm=False,eps=1e-5,momentum=0.1,use_none_layer=None,in_channels=3):
    layers = []

    BatchNorm=get_batchnorm()
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            if use_none_layer is None:
                if 'use_none_layer' in os.environ.keys():
                    use_none_layer = str2bool(os.environ['use_none_layer'])
                else:
                    warnings.warn('use default value for use_none_layer')
                    use_none_layer = False

            if use_none_layer:
                layers += [NoneLayer()]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if group_norm:
                assert not batch_norm,'group norm will overwrite batch_norm'
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=False)
                layers += [conv2d,
                           nn.GroupNorm(num_groups=32,num_channels=v,eps=eps),
                           nn.ReLU(inplace=True)]
            elif batch_norm:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=False)
                layers += [conv2d, BatchNorm(v,eps=eps,momentum=momentum), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=True)
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# when use momentum, remove last two pooling
#cfg = {
#    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
#}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512,'N', 512, 512, 'N'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'N', 512, 512, 'N'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N', 512, 512],
}

def vgg(cfg_key,url_key,pretrained=True,group_norm=False,eps=1e-5,momentum=0.1,in_channels=3,**kwargs):
    if pretrained and in_channels==3:
        kwargs['init_weights'] = False

    if group_norm is False and cfg_key.find('_bn')>=0:
        batch_norm=True
    else:
        batch_norm=False
    model = VGG(make_layers(cfg[cfg_key],batch_norm=batch_norm,
                            group_norm=group_norm,eps=eps,
                            momentum=momentum,in_channels=in_channels), **kwargs)
    if pretrained and in_channels==3:
        model.load_state_dict(model_zoo.load_url(model_urls[url_key]))
    elif pretrained:
        warnings.warn('not pretrain vgg model')
    return model

def vgg11(**kwargs):
    return vgg(cfg_key='A',url_key='vgg11', **kwargs)

def vgg11_bn(**kwargs):
    return vgg(cfg_key='A',url_key='vgg11_bn',**kwargs)

def vgg13(**kwargs):
    return vgg(cfg_key='B',url_key='vgg13',**kwargs)

def vgg13_bn(**kwargs):
    return vgg(cfg_key='B',url_key='vgg13_bn',**kwargs)

def vgg16(**kwargs):
    return vgg(cfg_key='D',url_key='vgg16',**kwargs)

def vgg16_bn(**kwargs):
    return vgg(cfg_key='D',url_key='vgg16_bn',**kwargs)

def vgg16_gn(**kwargs):
    return vgg(cfg_key='D',url_key='vgg16_bn',group_norm=True,**kwargs)

def vgg19(**kwargs):
    return vgg(cfg_key='E',url_key='vgg19',**kwargs)

def vgg19_bn(**kwargs):
    return vgg(cfg_key='E',url_key='vgg19_bn',**kwargs)

def vgg19_gn(**kwargs):
    return vgg(cfg_key='E',url_key='vgg19_bn',group_norm=True,**kwargs)

def vgg21(**kwargs):
    return vgg(cfg_key='F',url_key='vgg19',**kwargs)

def vgg21_bn(**kwargs):
    return vgg(cfg_key='F',url_key='vgg19_bn',**kwargs)

