# -*- coding: utf-8 -*-
import torch.utils.data as TD
import random
from dataset.cityscapes import cityscapes
from easydict import EasyDict as edict
import argparse
import torchsummary

from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from utils.augmentor import Augmentations
from utils.torch_tools import do_train_or_val

if __name__ == '__main__':
    choices = ['edge', 'global', 'augmentor', 'momentum', 'midnet',
               'backbone', 'dict', 'fractal', 'upsample_type',
               'pretrained']
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="test for choices",
                        choices=choices,
                        default=random.choice(choices))
    
    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=2)
    
    parser.add_argument("--backbone_name",
                        help="backbone name",
                        choices=['vgg16','vgg19','vgg16_bn','vgg19_bn','resnet18','resnet34','resnet50','resnet101','resnet152'],
                        default='resnet50')
    
    parser.add_argument('--net_name',
                        help='net name for semantic segmentaion',
                        choices=['pspnet','psp_edge','psp_global','psp_fractal','psp_dict'],
                        default='pspnet')
    
    parser.add_argument('--midnet_scale',
                        help='pspnet scale',
                        type=int,
                        default=5)
    
    parser.add_argument('--midnet_name',
                        help='midnet name',
                        choices=['psp','aspp'],
                        default='psp')
    
    parser.add_argument('--n_epoch',
                        help='training/validating epoch',
                        type=int,
                        default=100)
    
    parser.add_argument('--augmentation',
                        help='true or false to do augmentation',
                        type=bool,
                        default=True)
    
    parser.add_argument('--upsample_type',
                        help='bilinear or duc upsample',
                        choices=['duc','bilinear'],
                        default='duc')
    
    parser.add_argument('--upsample_layer',
                        help='layer number for upsample',
                        type=int,
                        default=3)
    
    parser.add_argument('--use_momentum',
                        help='use mometnum or not?',
                        type=bool,
                        default=False)
    
    parser.add_argument('--backbone_pretrained',
                        help='when not use momentum, we can use weights pretrained on imagenet',
                        type=bool,
                        default=False)
    
    parser.add_argument('--input_shape',
                        help='input shape',
                        type=int,
                        default=0)
    
    parser.add_argument('--note',
                        help='comment for tensorboard log',
                        default='naive')
    args = parser.parse_args()
    
    config = edict()
    config.model = edict()
    config.model.upsample_type = args.upsample_type
    config.model.upsample_layer = args.upsample_layer
    config.model.use_momentum = args.use_momentum
    config.model.backbone_pretrained=args.backbone_pretrained
    config.model.eps=1e-5
    config.model.momentum=0.9
    config.model.class_number = 19
    config.model.backbone_name = args.backbone_name
    config.model.layer_preference = 'first'

    config.model.midnet_pool_sizes = [6, 3, 2, 1]
    config.model.midnet_scale = args.midnet_scale
    config.model.midnet_name=args.midnet_name
    
    if args.input_shape==0:
        if args.midnet_name=='psp':
            count_size=max(config.model.midnet_pool_sizes)*config.model.midnet_scale*2**args.upsample_layer
            input_shape=(count_size,count_size)
        else:
            input_shape=(72*8,72*8)
    else:
        input_shape=(args.input_shape,args.input_shape)
        
    config.model.input_shape=input_shape
    config.model.midnet_out_channels = 512

    config.dataset = edict()
    config.dataset.root_path = '/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split = random.choice(['test', 'val', 'train'])
    config.dataset.resize_shape = input_shape
    config.dataset.name = 'cityscapes'
    config.dataset.ignore_index = 255

    config.args = edict()
    config.args.n_epoch = args.n_epoch
    config.args.log_dir = '/home/yzbx/tmp/logs/pytorch'
    config.args.note = args.note
    # must change batch size here!!!
    batch_size = args.batch_size
    config.args.batch_size = batch_size

    # prefer setting
    config.model.backbone_lr_ratio = 1.0
    config.dataset.norm = True
    
    if args.augmentation:
#        augmentations = Augmentations(p=0.25,use_imgaug=False)
        augmentations = Augmentations(p=0.25,use_imgaug=True)
    else:
        augmentations = None
        
    train_dataset = cityscapes(
        config.dataset, split='train', augmentations=augmentations)
    train_loader = TD.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = cityscapes(config.dataset, split='val',
                             augmentations=augmentations)
    val_loader = TD.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)

    config.args.note = '_'.join([args.note,
                                 'bn'+str(batch_size),
                                 'aug',str(args.augmentation)[0],
                                 ])
    note=config.args.note
    test = args.test
    if test == 'edge':
        config.dataset.with_edge = True
        for edge_width in [10]:
            config.dataset.edge_width = edge_width
            config.args.note = '_'.join([note,'edge_width',str(edge_width)])
            net = psp_edge(config)
            do_train_or_val(net,config.args, train_loader, val_loader)
    elif test == 'global':
        config.model.gnet_dilation_sizes = [16, 8, 4]
        config.args.note = note
        net = psp_global(config)
        do_train_or_val(net,config.args, train_loader, val_loader)
    elif test == 'backbone':
        for backbone in ['vgg16','vgg19','vgg16_bn','vgg19_bn']:
            config.model.backbone_name = backbone
            config.args.note = '_'.join([args.note,
                                 'bn'+str(batch_size),
                                 'aug',str(args.augmentation)[0],
                                 backbone
                                 ])
            net = pspnet(config)
            do_train_or_val(net,config.args, train_loader, val_loader)
    elif test == 'dict':
        config.args.note = 'dict'
        dict_number = config.model.class_number*5+1
        dict_lenght = config.model.class_number*2+1
        config.model.dict_number = dict_number
        config.model.dict_length = dict_lenght
        config.args.note = '_'.join(
            [config.args.note, '%dx%d' % (dict_number, dict_lenght)])
        net = psp_dict(config)
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'fractal':
        before_upsample = True
        fractal_depth = 8
        fractal_fusion_type = 'mean'
        config.model.before_upsample = before_upsample
        config.model.fractal_depth = fractal_depth
        config.model.fractal_fusion_type = fractal_fusion_type

        location_str = 'before' if before_upsample else 'after'
        config.args.note = '_'.join([config.args.note, location_str, 'depth', str(
            fractal_depth), 'fusion', fractal_fusion_type])
        net = psp_fractal(config)
        do_train_or_val(net,config.args, train_loader, val_loader)
    elif test == 'upsample_type':
        backbone = 'resnet52'
        config.model.backbone_name = backbone
        for upsample_type in ['duc', 'bilinear']:
            config.model.upsample_type = upsample_type
            config.args.note = '_'.join([backbone, upsample_type, 'keras_psp'])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test=='momentum':
        backbone = 'vgg19_bn'
        config.model.backbone_name = backbone
        config.model.use_momentum = True
        for momentum in [0.1,0.3,0.5,0.7,0.9]:
            config.args.note= '_'.join([note,backbone,'mo',str(momentum)])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
            break
    elif test=='midnet':
        # midnet will change input shape
#        for midnet_name in ['psp','aspp']:
#        config.model.midnet_name=midnet_name
        for backbone in ['vgg19_bn','resnet50']:
            config.model.backbone_name = backbone
            config.args.note= '_'.join([note,backbone,args.midnet_name])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test=='pretrained':
        midnet_name='aspp'
        backbone=config.model.backbone_name
        config.model.midnet_name=midnet_name
        for pretrained in [True,False]:
            config.model.backbone_pretrained = pretrained
            config.args.note= '_'.join([note,'pretrain',str(pretrained)[0],backbone,midnet_name])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test=='summary':
        net=pspnet(config)
        height,width=input_shape
        torchsummary.summary(net,(3,height,width))
    else:
        raise NotImplementedError
        