# -*- coding: utf-8 -*-
import torch.utils.data as TD
import sys
if '.' not in sys.path:
    sys.path.append('.')

from dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config, image_normalizations
from easydict import EasyDict as edict
import torchsummary
import pandas as pd
import torch
import json
import time
import os


from models.pspnet import pspnet
from models.psp_edge import psp_edge
from models.psp_global import psp_global
from models.psp_dict import psp_dict
from models.psp_fractal import psp_fractal
from models.fcn import fcn, fcn8s, fcn16s, fcn32s
from models.psp_aux import psp_aux
from models.psp_hed import psp_hed
from models.merge_seg import merge_seg
from models.cross_merge import cross_merge
from models.psp_convert import psp_convert
from models.psp_convert import CONFIG as psp_convert_config
from models.motionnet import motionnet,motion_panet
from utils.augmentor import Augmentations
from utils.torch_tools import keras_fit
from utils.benchmark import keras_benchmark,get_loader
from utils.config import get_parser,get_config

if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    config = get_config(args)

    if args.test == 'convert':
        input_shape = tuple(
            psp_convert_config[args.dataset_name]['input_size'])
        config.input_shape = input_shape
        config.resize_shape = input_shape
        print('convert input shape is',input_shape,'*'*30)

    if config.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.norm_ways)

    if config.augmentation:
        augmentations = Augmentations(config)
    else:
        augmentations = None

    # must change batch size here!!!
    batch_size = args.batch_size

    train_dataset = dataset_generalize(
        config, split='train',
        augmentations=augmentations,
        normalizations=normalizations)
    train_loader = TD.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)

    val_dataset = dataset_generalize(config,
                                     split='val',
                                     augmentations=None,
                                     normalizations=normalizations)
    val_loader = TD.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2)

    note = config.note
    test = args.test

    if test == 'naive':
        net = globals()[args.net_name](config)
        best_val_iou=keras_fit(net,train_loader,val_loader)
        print('best val iou is %0.3f'%best_val_iou)
    elif test == 'edge':
        config.with_edge = True
        net = psp_edge(config)
        keras_fit(net, train_loader, val_loader)
    elif test == 'global':
        config.gnet_dilation_sizes = [16, 8, 4]
        config.note = note
        net = psp_global(config)
        keras_fit(net, train_loader, val_loader)
    elif test == 'dict':
        config.note = 'dict'
        dict_number = config.class_number*5+1
        dict_lenght = config.class_number*2+1
        config.dict_number = dict_number
        config.dict_length = dict_lenght
        config.note = '_'.join(
            [config.note, '%dx%d' % (dict_number, dict_lenght)])
        net = psp_dict(config)
        keras_fit(net, train_loader, val_loader)
    elif test == 'fractal':
        before_upsample = True
        fractal_depth = 8
        fractal_fusion_type = 'mean'
        config.before_upsample = before_upsample
        config.fractal_depth = fractal_depth
        config.fractal_fusion_type = fractal_fusion_type

        location_str = 'before' if before_upsample else 'after'
        config.note = '_'.join([config.note, location_str, 'depth', str(
            fractal_depth), 'fusion', fractal_fusion_type])
        net = psp_fractal(config)
        keras_fit(net, train_loader, val_loader)
    elif test == 'coarse':
        net = globals()[args.net_name](config)
        for dataset_name in ['Cityscapes', 'Cityscapes_Fine']:
            config = get_dataset_generalize_config(
                config, dataset_name)
            config.dataset_name = dataset_name.lower()

            coarse_train_dataset = dataset_generalize(
                config,
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
                    config,
                    split='val',
                    augmentations=augmentations,
                    normalizations=normalizations)
            coarse_val_loader = TD.DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=8)
            keras_fit(net,coarse_train_loader, coarse_val_loader)
    elif test == 'convert':
        train_loader = None
        load_caffe_weight = True
        net = psp_convert(dataset_name=args.dataset_name,
                          load_caffe_weight=load_caffe_weight)
        if load_caffe_weight:
            keras_fit(model=net,
                            train_loader=train_loader, val_loader=val_loader, config=config)
        else:
            net.load_state_dict(torch.load(
                psp_convert_config[args.dataset_name]['params']))
            keras_fit(model=net,
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
        from utils.model_hyperopt import psp_opt
        config.log_dir = os.path.expanduser('~/tmp/logs/hyperopt')
        psp_model=globals()[args.net_name]
        config.n_calls=args.hyperopt_calls
        config.hyperkey=args.hyperkey
        hyperopt=psp_opt(psp_model,config,train_loader=None,val_loader=None)
        if args.hyperopt=='tpe':
            hyperopt.tpe()
        elif args.hyperopt=='bayes':
            hyperopt.bayes()
        elif args.hyperopt=='skopt':
            hyperopt.skopt()
        elif args.hyperopt=='loop':
            hyperopt.loop()
        else:
            assert False,'unknown hyperopt %s'%args.hyperopt
    elif test == 'benchmark':
        config.with_path=True
        config.augmentation=False
        net = globals()[args.net_name](config)
#        test_loader=get_loader(config,'val')
        test_loader=None
        keras_benchmark(model=net,
                        test_loader=test_loader,
                        config=config,
                        checkpoint_path=args.checkpoint_path,
                        predict_save_path=args.predict_save_path)
    elif test == 'cycle_lr':
        n_epoch=args.n_epoch
        net = globals()[args.net_name](config)
        for times in range(3):
            config.n_epoch=n_epoch*2**times
            config.note = note+'_%d'%times
            assert net.config.n_epoch==config.n_epoch
            keras_fit(model=net,train_loader=train_loader,val_loader=val_loader)
            # only load weight in the first time
            config.checkpoint_path=None
    else:
        raise NotImplementedError

    cmd_log_file=os.path.join(args.log_dir,'cmd_log.txt')
    with open(cmd_log_file, "a") as myfile:

        time_str = time.strftime("%Y-%m-%d  %H-%M-%S", time.localtime())
        myfile.write(time_str+'\n')
        myfile.write(" ".join(sys.argv)+'\n')