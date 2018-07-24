import torch.nn as nn


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

    def __init__(self, block, layers, psp_mode=True, momentum=0.1):
        self.inplanes = 64
        self.momentum=momentum
        self.psp_mode=psp_mode
        super(ResNet, self).__init__()

        if psp_mode:
            self.prefix_net = nn.Sequential(self.conv_bn_relu(in_channels=3,
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
                                                              out_channels=128,
                                                              kernel_size=3,
                                                              stride=1,
                                                              padding=1),
                                            nn.MaxPool2d(kernel_size=3,
                                                         stride=2,
                                                         padding=1))
        else:
            self.prefix_net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                                      bias=False),
                                            nn.BatchNorm2d(
                                                64, momentum=momentum),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

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
        if index == 1:
            dilation = 1
        elif index == 2:
            dilation = 1
        elif index == 3:
            dilation = 2
        elif index == 4:
            dilation = 4
        else:
            assert False, 'unexpected index=%d' % index

        if index == 1 and self.psp_mode:
            in_channels = 128
        else:
            in_channels = self.inplanes

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            if dilation != 1:
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

    def forward(self, x):
        x = self.prefix_net(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(momentum=0.1):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], momentum=momentum)
    return model


def resnet101(momentum=0.1):
    """Constructs a ResNet-101 model.

    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], momentum=momentum)
    return model


def resnet152(momentum=0.1):
    """Constructs a ResNet-152 model.

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], momentum=momentum)
    return model


if __name__ == '__main__':
    net = resnet101(momentum=0.5)
    print(net)
