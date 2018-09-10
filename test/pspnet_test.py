# -*- coding: utf-8 -*-
import torch.utils.data as TD
from dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config, image_normalizations
from easydict import EasyDict as edict
import argparse
import torchsummary
import torch
import json

from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from models.fcn import fcn, fcn8s, fcn16s, fcn32s
from models.psp_aux import psp_aux
from models.psp_convert import psp_convert
from models.psp_convert import CONFIG as psp_convert_config
from utils.augmentor import Augmentations
from utils.torch_tools import do_train_or_val
from utils.disc_tools import str2bool

if __name__ == '__main__':
    choices = ['edge', 'global', 'augmentor', 'momentum', 'midnet',
               'backbone', 'dict', 'fractal', 'upsample_type',
               'pretrained', 'summary', 'naive', 'coarse',
               'convert']
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",
                        help="test for choices",
                        choices=choices,
                        default='naive')

    parser.add_argument("--batch_size",
                        help="batch size",
                        type=int,
                        default=2)

    parser.add_argument("--learning_rate",
                        help="learning rate",
                        type=float,
                        default=0.0001)

    parser.add_argument("--optimizer",
                        help="optimizer name",
                        choices=['adam', 'sgd'],
                        default='adam')

    parser.add_argument("--use_reg",
                        help='use l1 and l2 regularizer or not (default False)',
                        default=False,
                        type=str2bool)
    
    parser.add_argument('--l1_reg',
                        help='l1 reg loss weights',
                        type=float,
                        default=1e-7)
    
    parser.add_argument('--l2_reg',
                        help='l2 reg loss weights',
                        type=float,
                        default=1e-5)

    parser.add_argument('--dataset_name',
                        help='dataset name',
                        choices=['ADEChallengeData2016', 'VOC2012', 'Kitti2015',
                                 'Cityscapes', 'Cityscapes_Fine', 'Cityscapes_Coarse'],
                        default='Cityscapes')

    parser.add_argument("--backbone_name",
                        help="backbone name",
                        choices=['vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn', 'resnet18',
                                 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        default='resnet50')
    
    parser.add_argument('--backbone_pretrained',
                        help='when not use momentum, we can use weights pretrained on imagenet',
                        type=str2bool,
                        default=False)
    
    # work for pspnet and psp_edge
    parser.add_argument('--backbone_freeze',
                        help='finetune/freeze backbone or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--net_name',
                        help='net name for semantic segmentaion',
                        choices=['pspnet', 'psp_edge', 'psp_global',
                                 'psp_fractal', 'psp_dict', 'psp_aux',
                                 'fcn', 'fcn8s', 'fcn16s', 'fcn32s'],
                        default='pspnet')

    parser.add_argument('--midnet_scale',
                        help='pspnet scale',
                        type=int,
                        default=5)

    parser.add_argument('--midnet_name',
                        help='midnet name',
                        choices=['psp', 'aspp'],
                        default='psp')

    parser.add_argument('--n_epoch',
                        help='training/validating epoch',
                        type=int,
                        default=100)

    parser.add_argument('--augmentation',
                        help='true or false to do augmentation',
                        type=str2bool,
                        default=True)

    parser.add_argument('--upsample_type',
                        help='bilinear or duc upsample',
                        choices=['duc', 'bilinear'],
                        default='duc')

    parser.add_argument('--auxnet_type',
                        help='bilinear or duc upsample',
                        choices=['duc', 'bilinear'],
                        default='duc')

    parser.add_argument('--upsample_layer',
                        help='layer number for upsample',
                        type=int,
                        default=3)

    parser.add_argument('--auxnet_layer',
                        help='layer number for auxnet',
                        type=int,
                        default=4)
    
    parser.add_argument('--edge_bg_weight',
                        help='weight for edge bg, the edge fg weight is 1.0',
                        type=float,
                        default=0.01)
    
    parser.add_argument('--edge_base_weight',
                        help='base weight for edge loss, weight for segmentation is 1.0',
                        type=float,
                        default=1.0)
    
    parser.add_argument('--edge_power',
                        help='weight for edge power',
                        type=float,
                        default=0.9)
    
    parser.add_argument('--edge_class_num',
                        help='class number for edge',
                        type=int,
                        default=2)
    
    parser.add_argument('--edge_width',
                        help='width for dilate edge',
                        type=int,
                        default=10)
    
    parser.add_argument('--edge_seg_order',
                        help='edge seg order',
                        choices=['first','later','same'],
                        default='same')

    parser.add_argument('--use_momentum',
                        help='use mometnum or not?',
                        type=str2bool,
                        default=False)

    parser.add_argument('--input_shape',
                        help='input shape',
                        type=int,
                        default=0)

    parser.add_argument('--augmentations_blur',
                        help='augmentations blur',
                        type=str2bool,
                        default=True)

    parser.add_argument('--augmentations_rotate',
                        help='augmentations rotate',
                        type=str2bool,
                        default=True)

    parser.add_argument('--note',
                        help='comment for tensorboard log',
                        default=None)
    args = parser.parse_args()

    config = edict()
    config.model = edict()
    config.model.upsample_type = args.upsample_type
    config.model.auxnet_type = args.auxnet_type
    config.model.upsample_layer = args.upsample_layer
    config.model.auxnet_layer = args.auxnet_layer
    config.model.use_momentum = args.use_momentum
    config.model.backbone_pretrained = args.backbone_pretrained
    config.model.eps = 1e-5
    config.model.momentum = 0.9
    config.model.learning_rate = args.learning_rate
    config.model.optimizer = args.optimizer
    config.model.use_reg = args.use_reg
    config.model.l1_reg=args.l1_reg
    config.model.l2_reg=args.l2_reg
    config.model.backbone_name = args.backbone_name
    config.model.backbone_freeze = args.backbone_freeze
    config.model.layer_preference = 'first'
    config.model.edge_seg_order=args.edge_seg_order

    config.model.midnet_pool_sizes = [6, 3, 2, 1]
    config.model.midnet_scale = args.midnet_scale
    config.model.midnet_name = args.midnet_name
    
    config.model.edge_bg_weight=args.edge_bg_weight
    config.model.edge_base_weight=args.edge_base_weight
    config.model.edge_power=args.edge_power

    config.dataset = edict()
    config.dataset.edge_class_num=args.edge_class_num
    config.dataset.edge_width=args.edge_width
    if args.dataset_name in ['VOC2012','Cityscapes']:
        config.dataset.norm_ways = args.dataset_name.lower()
    else:
        config.dataset.norm_ways = 'pytorch'
    
    if args.test == 'convert':
        input_shape = tuple(
            psp_convert_config[args.dataset_name]['input_size'])
    elif args.input_shape == 0:
        if args.midnet_name == 'psp':
            count_size = max(config.model.midnet_pool_sizes) * \
                config.model.midnet_scale*2**args.upsample_layer
            input_shape = (count_size, count_size)
        else:
            input_shape = (72*8, 72*8)
    else:
        input_shape = (args.input_shape, args.input_shape)

    if config.dataset.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.dataset.norm_ways)

    config.model.input_shape = input_shape
    config.model.midnet_out_channels = 512
    config.dataset = get_dataset_generalize_config(
        config.dataset, args.dataset_name)
    if config.dataset.ignore_index == 0:
        config.model.class_number = len(config.dataset.foreground_class_ids)+1
    else:
        config.model.class_number = len(config.dataset.foreground_class_ids)
    config.dataset.resize_shape = input_shape
    config.dataset.name = args.dataset_name.lower()
    config.dataset.augmentations_blur = args.augmentations_blur

    config.args = edict()
    config.args.n_epoch = args.n_epoch
    config.args.log_dir = '/home/yzbx/tmp/logs/pytorch'
    config.args.note = args.note
    # must change batch size here!!!
    batch_size = args.batch_size
    config.args.batch_size = batch_size

    if args.augmentation:
        #        augmentations = Augmentations(p=0.25,use_imgaug=False)
        augmentations = Augmentations(
            p=0.25, use_imgaug=True, rotate=args.augmentations_rotate)
    else:
        augmentations = None

    train_dataset = dataset_generalize(
        config.dataset, split='train',
        augmentations=augmentations,
        normalizations=normalizations)
    train_loader = TD.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        num_workers=8)

    val_dataset = dataset_generalize(config.dataset, 
                                     split='val',
                                     augmentations=augmentations,
                                     normalizations=normalizations)
    val_loader = TD.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=8)
    
    if args.note is None:
        config.args.note = '_'.join([args.test,
                                     'bn'+str(batch_size),
                                     'aug', str(args.augmentation)[0],
                                     ])
    else:
        config.args.note=args.note
        
    note = config.args.note
    test = args.test
    if test == 'naive':
        net = globals()[args.net_name](config)
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'edge':
        config.dataset.with_edge = True
        net = psp_edge(config)
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'global':
        config.model.gnet_dilation_sizes = [16, 8, 4]
        config.args.note = note
        net = psp_global(config)
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'backbone':
        for backbone in ['vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn']:
            config.model.backbone_name = backbone
            config.args.note = '_'.join([args.note,
                                         'bn'+str(batch_size),
                                         'aug', str(args.augmentation)[0],
                                         backbone
                                         ])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
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
        do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'upsample_type':
        backbone = 'resnet52'
        config.model.backbone_name = backbone
        for upsample_type in ['duc', 'bilinear']:
            config.model.upsample_type = upsample_type
            config.args.note = '_'.join([backbone, upsample_type, 'keras_psp'])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'momentum':
        backbone = 'vgg19_bn'
        config.model.backbone_name = backbone
        config.model.use_momentum = True
        for momentum in [0.1, 0.3, 0.5, 0.7, 0.9]:
            config.args.note = '_'.join([note, backbone, 'mo', str(momentum)])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
            break
    elif test == 'midnet':
        # midnet will change input shape
        #        for midnet_name in ['psp','aspp']:
        #        config.model.midnet_name=midnet_name
        for backbone in ['vgg19_bn', 'resnet50']:
            config.model.backbone_name = backbone
            config.args.note = '_'.join([note, backbone, args.midnet_name])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'pretrained':
        midnet_name = 'aspp'
        backbone = config.model.backbone_name
        config.model.midnet_name = midnet_name
        for pretrained in [True, False]:
            config.model.backbone_pretrained = pretrained
            config.args.note = '_'.join(
                [note, 'pretrain', str(pretrained)[0], backbone, midnet_name])
            net = pspnet(config)
            do_train_or_val(net, config.args, train_loader, val_loader)
    elif test == 'coarse':
        net = globals()[args.net_name](config)
        for dataset_name in ['Cityscapes', 'Cityscapes_Fine']:
            config.dataset = get_dataset_generalize_config(
                config.dataset, dataset_name)
            config.dataset.name = dataset_name.lower()

            coarse_train_dataset = dataset_generalize(
                config.dataset, 
                split='train',
                augmentations=augmentations,
                normalizations=normalizations)
            coarse_train_loader = TD.DataLoader(
                dataset=train_dataset, 
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=8)

            coarse_val_dataset = dataset_generalize(
                    config.dataset, 
                    split='val',
                    augmentations=augmentations,
                    normalizations=normalizations)
            coarse_val_loader = TD.DataLoader(
                dataset=val_dataset,
                batch_size=batch_size, 
                shuffle=True, 
                drop_last=False, 
                num_workers=8)
            do_train_or_val(net, config.args,
                            coarse_train_loader, coarse_val_loader)
    elif test == 'convert':
        train_loader = None
        load_caffe_weight = True
        net = psp_convert(dataset_name=args.dataset_name,
                          load_caffe_weight=load_caffe_weight)
        if load_caffe_weight:
            do_train_or_val(model=net, args=config.args,
                            train_loader=train_loader, val_loader=val_loader, config=config)
        else:
            net.load_state_dict(torch.load(
                psp_convert_config[args.dataset_name]['params']))
            do_train_or_val(model=net, args=config.args,
                            train_loader=train_loader, val_loader=val_loader, config=config)

    elif test == 'summary':
        net = pspnet(config)
        config_str = json.dumps(config, indent=2, sort_keys=True)
        print(config_str)
        print('args is '+'*'*30)
        print(args)
        height, width = input_shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torchsummary.summary(net.to(device), (3, height, width))
    else:
        raise NotImplementedError
