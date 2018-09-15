# -*- coding: utf-8 -*-
import torch.utils.data as TD
from dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config, image_normalizations
from easydict import EasyDict as edict
import torchsummary
import pandas as pd
import torch
import json

from utils.model_hyperopt import psp_opt
from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from models.fcn import fcn, fcn8s, fcn16s, fcn32s
from models.psp_aux import psp_aux
from models.merge_seg import merge_seg
from models.cross_merge import cross_merge
from models.psp_convert import psp_convert
from models.psp_convert import CONFIG as psp_convert_config
from utils.augmentor import Augmentations
from utils.torch_tools import do_train_or_val,keras_fit
from utils.config import get_parser

if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()

    config = edict()
    config.model = edict()
    config.model.upsample_type = args.upsample_type
    config.model.auxnet_type = args.auxnet_type
    config.model.upsample_layer = args.upsample_layer
    config.model.auxnet_layer = args.auxnet_layer
    config.model.cross_merge_times=args.cross_merge_times
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
    config.dataset.edge_with_gray=args.edge_with_gray
    config.dataset.with_edge=False
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
        if args.net_name in ['psp_edge','merge_seg','cross_merge']:
            config.dataset.with_edge = True
        net = globals()[args.net_name](config)
#        do_train_or_val(net, config.args, train_loader, val_loader)
        keras_fit(net,train_loader,val_loader)
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
        net = globals()[args.net_name](config)
        config_str = json.dumps(config, indent=2, sort_keys=True)
        print(config_str)
        print('args is '+'*'*30)
        print(args)
        height, width = input_shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torchsummary.summary(net.to(device), (3, height, width))
    elif test == 'hyperopt':
        psp_model=globals()[args.net_name]
        hyperopt=psp_opt(psp_model,config,train_loader,val_loader)
        if args.hyperopt=='tpe':
            hyperopt.tpe()
        elif args.hyperopt=='bayes':
            hyperopt.bayes()
        elif args.hyperopt=='skopt':
            hyperopt.skopt()
        else:
            assert False,'unknown hyperopt %s'%args.hyperopt
    else:
        raise NotImplementedError
