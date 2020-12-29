# -*- coding: utf-8 -*-
import torch.utils.data as TD
import torch
import json
import time
import os
import sys
if '.' not in sys.path:
    sys.path.append('.')

from torchseg.dataset.dataset_generalize import dataset_generalize, \
    get_dataset_generalize_config, image_normalizations
from torchseg.utils.augmentor import Augmentations
from torchseg.utils.torch_tools import keras_fit
from torchseg.utils import torchsummary
from torchseg.utils.benchmark import keras_benchmark
from torchseg.utils.configs.semanticseg_config import get_parser,get_config,get_net,get_sub_config
from torchseg.utils.distributed_tools import dist_train
from torchseg.models.semanticseg.psp_edge import psp_edge
from torchseg.models.semanticseg.psp_dict import psp_dict 
from torchseg.models.semanticseg.psp_fractal import psp_fractal
from torchseg.models.semanticseg.psp_global import psp_global

if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    config = get_config(args)

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
        net = get_net(config)
        best_val_iou=keras_fit(net,train_loader,val_loader)
        print('best val iou is %0.3f'%best_val_iou)
    elif test == 'dist':
        dist_train(config)
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
        net = get_net(config)
        for dataset_name in ['Cityscapes_Coarse', 'Cityscapes_Fine']:
            config = get_dataset_generalize_config(
                config, dataset_name)
            #config.dataset_name = dataset_name.lower()

            coarse_train_dataset = dataset_generalize(
                config,
                split='train',
                augmentations=augmentations,
                normalizations=normalizations)
            coarse_train_loader = TD.DataLoader(
                dataset=coarse_train_dataset,
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
                dataset=coarse_val_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=8)
            keras_fit(net,coarse_train_loader, coarse_val_loader)
    elif test == 'summary':
        net = get_net(config)
        config_str = json.dumps(config, indent=2, sort_keys=True)
        print(config_str)
        print('args is '+'*'*30)
        print(args)
        height, width = config.input_shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torchsummary.summary(net.to(device), (3, height, width))
    elif test == 'huawei':
        # train on cityscapes, validation on huawei
        net = get_net(config)
        
        train_datasets=[]
        for dataset_name in ['Cityscapes_Category','HuaWei']:
            config = get_dataset_generalize_config(
                    config, dataset_name)
            train_dataset = dataset_generalize(
                    config,
                    split='train',
                    augmentations=augmentations,
                    normalizations=normalizations)
            train_datasets.append(train_dataset)
        
        merge_dataset=TD.ConcatDataset(train_datasets)
        train_loader = TD.DataLoader(
            dataset=merge_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8)
        
        config=get_dataset_generalize_config(
                config, "HuaWei")
        val_dataset = dataset_generalize(
                config,
                split='val',
                augmentations=augmentations,
                normalizations=normalizations)
        val_loader = TD.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=8)
        keras_fit(net,train_loader, val_loader)
        
    elif test == 'benchmark':
        config.with_path=True
        config.augmentation=False
        net = get_net(config)
#        test_loader=get_loader(config,'val')
        test_loader=None
        keras_benchmark(model=net,
                        test_loader=test_loader,
                        config=config,
                        checkpoint_path=args.checkpoint_path,
                        predict_save_path=args.predict_save_path)
    elif test == 'cycle_lr':
        n_epoch=args.n_epoch
        sub_args=get_sub_config(config,test)
        net = get_net(config)
        for times in range(sub_args.cycle_times):
            if sub_args.cycle_period=='const':
                config.n_epoch=n_epoch
            elif sub_args.cycle_period=='linear':
                config.n_epoch=n_epoch*(1+times)
            elif sub_args.cycle_period=='exponent':
                config.n_epoch=n_epoch*2**times
            else:
                assert False,'unknown cycle_period {}'.format(sub_args.cycle_period)
                
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