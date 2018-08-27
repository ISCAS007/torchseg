# -*- coding: utf-8 -*-

import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_suffix_net
import torch.nn.functional as F

class fcn(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        
        if hasattr(self.config.model,'use_momentum'):
            use_momentum=self.config.model.use_momentum
        else:
            use_momentum=False
        
        self.backbone=backbone(config.model,use_momentum=use_momentum)
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.midnet_out_channels=self.midnet_input_shape[1]
        self.decoder=get_suffix_net(config,
                                    self.midnet_out_channels,
                                    self.class_number)
        
    def forward(self,x):
        x=self.backbone.forward(x,self.upsample_layer)
        x=self.decoder(x)
        return x

# FCN32s
class fcn32s(TN.Module):

    def __init__(self,config):
        super(fcn32s, self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.learned_billinear = False
        self.n_classes = self.config.model.class_number
        self.class_number = self.config.model.class_number
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index

        self.conv_block1 = TN.Sequential(
            TN.Conv2d(3, 64, 3, padding=100),
            TN.ReLU(inplace=True),
            TN.Conv2d(64, 64, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = TN.Sequential(
            TN.Conv2d(64, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(128, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = TN.Sequential(
            TN.Conv2d(128, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = TN.Sequential(
            TN.Conv2d(256, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = TN.Sequential(
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = TN.Sequential(
            TN.Conv2d(512, 4096, 7),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, 4096, 1),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, self.n_classes, 1),)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = TN.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)

        out = F.upsample_bilinear(score, x.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, TN.Conv2d) and isinstance(l2, TN.Conv2d):
                    # print(idx, l1, l2)
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            # print(type(l1), dir(l1))
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

class fcn16s(TN.Module):

    def __init__(self, config):
        super(fcn16s, self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.learned_billinear = False
        self.n_classes = self.config.model.class_number
        self.class_number = self.config.model.class_number
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index

        self.conv_block1 = TN.Sequential(
            TN.Conv2d(3, 64, 3, padding=100),
            TN.ReLU(inplace=True),
            TN.Conv2d(64, 64, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = TN.Sequential(
            TN.Conv2d(64, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(128, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = TN.Sequential(
            TN.Conv2d(128, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = TN.Sequential(
            TN.Conv2d(256, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = TN.Sequential(
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = TN.Sequential(
            TN.Conv2d(512, 4096, 7),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, 4096, 1),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = TN.Conv2d(512, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = TN.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None


    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        out = F.upsample_bilinear(score, x.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, TN.Conv2d) and isinstance(l2, TN.Conv2d):
                    # print(idx, l1, l2)
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]

# FCN 8s
class fcn8s(TN.Module):

    def __init__(self, config):
        super(fcn8s, self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.learned_billinear = False
        self.n_classes = self.config.model.class_number
        self.class_number = self.config.model.class_number
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index

        self.conv_block1 = TN.Sequential(
            TN.Conv2d(3, 64, 3, padding=100),
            TN.ReLU(inplace=True),
            TN.Conv2d(64, 64, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block2 = TN.Sequential(
            TN.Conv2d(64, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(128, 128, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block3 = TN.Sequential(
            TN.Conv2d(128, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(256, 256, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block4 = TN.Sequential(
            TN.Conv2d(256, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.conv_block5 = TN.Sequential(
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.Conv2d(512, 512, 3, padding=1),
            TN.ReLU(inplace=True),
            TN.MaxPool2d(2, stride=2, ceil_mode=True),)

        self.classifier = TN.Sequential(
            TN.Conv2d(512, 4096, 7),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, 4096, 1),
            TN.ReLU(inplace=True),
            TN.Dropout2d(),
            TN.Conv2d(4096, self.n_classes, 1),)

        self.score_pool4 = TN.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = TN.Conv2d(256, self.n_classes, 1)

        # TODO: Add support for learned upsampling
        if self.learned_billinear:
            raise NotImplementedError
            # upscore = TN.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        score = self.classifier(conv5)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)

        score = F.upsample_bilinear(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample_bilinear(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample_bilinear(score, x.size()[2:])

        return out


    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, TN.Conv2d) and isinstance(l2, TN.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
        for i1, i2 in zip([0, 3], [0, 3]):
            l1 = vgg16.classifier[i1]
            l2 = self.classifier[i2]
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())
        n_class = self.classifier[6].weight.size()[0]
        if copy_fc8:
            l1 = vgg16.classifier[6]
            l2 = self.classifier[6]
            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
            l2.bias.data = l1.bias.data[:n_class]
