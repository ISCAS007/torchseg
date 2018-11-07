# -*- coding: utf-8 -*-

from easydict import EasyDict as edict
import random
import json
import yaml
import os
import argparse
from utils.disc_tools import str2bool
from dataset.dataset_generalize import get_dataset_generalize_config

def get_config(args=None):
    # for test some function
    if args is None:
        config=edict()
        config.model=edict()
        config.model.class_number=20
        config.model.backbone_name='vgg16'
        config.model.layer_preference='last'
        config.model.input_shape=(224,224)
        
        config.dataset=edict()
        config.dataset.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
        config.dataset.cityscapes_split=random.choice(['test','val','train'])
        config.dataset.resize_shape=(224,224)
        config.dataset.name='cityscapes'
        
        config.training=edict()
        config.training.n_epoch=300
        config.training.batch_size=4
        config.training.log_dir=os.path.expanduser('~/tmp/logs/pytorch')
        config.training.note='default'
        
        return config
    
    config = edict()
    config.model = edict()
    config.model.upsample_type = args.upsample_type
    config.model.auxnet_type = args.auxnet_type
    config.model.upsample_layer = args.upsample_layer
    config.model.auxnet_layer = args.auxnet_layer
    config.model.cross_merge_times=args.cross_merge_times
    config.model.use_momentum = args.use_momentum
    config.model.backbone_pretrained = args.backbone_pretrained
    config.model.use_bn=args.use_bn
    # use_bn,use_dropout will change the variable in local_bn,local_dropout
    # use_bias will affect the bias in upsample
    os.environ['torchseg_use_bn']=str(args.use_bn)
    config.model.use_dropout=args.use_dropout
    os.environ['torchseg_use_dropout']=str(args.use_dropout)
    config.model.use_bias=args.use_bias
    os.environ['torchseg_use_bias']=str(args.use_bias)
    
    config.model.eps = 1e-5
    config.model.momentum = args.momentum
    config.model.learning_rate = args.learning_rate
    config.model.optimizer = args.optimizer
    config.model.use_lr_mult = args.use_lr_mult
    config.model.changed_lr_mult=args.changed_lr_mult
    config.model.new_lr_mult=args.new_lr_mult
    config.model.use_reg = args.use_reg
    
    config.model.use_class_weight=args.use_class_weight
    config.model.class_weight_alpha=args.class_weight_alpha
    config.model.focal_loss_gamma=args.focal_loss_gamma
    config.model.focal_loss_alpha=args.focal_loss_alpha
#    config.model.l1_reg=args.l1_reg
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
    config.model.aux_base_weight=args.aux_base_weight

    config.dataset = edict()
    config.dataset.edge_class_num=args.edge_class_num
    config.dataset.edge_width=args.edge_width
    config.dataset.edge_with_gray=args.edge_with_gray
    config.dataset.with_edge=False
    
#    if args.dataset_name in ['VOC2012','Cityscapes']:
#        config.dataset.norm_ways = args.dataset_name.lower()
#    else:
#        config.dataset.norm_ways = 'pytorch'
#    config.dataset.norm_ways = 'pytorch'
    config.dataset.norm_ways = args.norm_ways
    
    if args.input_shape == 0:
        if args.midnet_name == 'psp':
            upsample_ratio=args.upsample_layer
            if args.use_momentum and args.upsample_layer>=3:
                upsample_ratio=3
            count_size = max(config.model.midnet_pool_sizes) * \
                config.model.midnet_scale*2**upsample_ratio
            input_shape = (count_size, count_size)
        else:
            input_shape = (72*8, 72*8)
    else:
        input_shape = (args.input_shape, args.input_shape)
    
    print('convert input shape is',input_shape,'*'*30)
    
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
    config.dataset.dataset_use_part=args.dataset_use_part

    config.args = edict()
    config.args.n_epoch = args.n_epoch
    # for hyperopt use ~/tmp/logs/hyperopt
    config.args.log_dir = args.log_dir
    config.args.summary_image=args.summary_image
    config.args.save_model=args.save_model
    config.args.iou_save_threshold=args.iou_save_threshold
    config.args.batch_size = args.batch_size
    config.args.augmentation = args.augmentation
    config.args.augmentations_rotate=args.augmentations_rotate
    config.args.net_name=args.net_name
    if args.net_name in ['psp_edge','merge_seg','cross_merge','psp_hed']:
        config.dataset.with_edge = True
        
    if args.note is None:
        config.args.note = '_'.join([args.test,
                                     'bs'+str(args.batch_size),
                                     'aug', str(args.augmentation)[0],
                                     ])
    else:
        config.args.note=args.note
    
    return config

def get_share_config(config):
    return config

def dump_config(config,log_dir,filename='config.txt'):
    os.makedirs(log_dir,exists_ok=True)
    config_path=os.path.join(log_dir,filename)
    config_file=open(config_path,'w')
    json.dump(config,config_file,sort_keys=True)
    
def load_config(config_file):
    f=open(config_file,'r')
    l=f.readline()
    f.close()

    d=yaml.load(l)
    config=edict(d)

    return config

def get_parser():
    choices = ['edge', 'global', 'dict', 'fractal',
               'summary', 'naive', 'coarse',
               'convert','hyperopt','benchmark', 'cycle_lr']
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
    
    parser.add_argument('--use_bn',
                        help='use batch norm or not',
                        default=True,
                        type=str2bool)
    
    parser.add_argument('--use_bias',
                        help='use bias or not',
                        default=False,
                        type=str2bool)
    
    parser.add_argument('--use_dropout',
                        help='use dropout or not',
                        default=True,
                        type=str2bool)
    
    parser.add_argument('--use_lr_mult',
                        help='use lr mult or not',
                        default=True,
                        type=str2bool)
    
    parser.add_argument('--use_class_weight',
                        help='use class-wise weight for segmenation or not',
                        default=False,
                        type=str2bool)
    
    parser.add_argument('--class_weight_alpha',
                        help='smooth parameter [0,1] for class weight',
                        default=0.0,
                        type=float)
    
    parser.add_argument('--focal_loss_gamma',
                        help='gamma for focal loss, <0 then not use focal loss',
                        default=-1.0,
                        type=float)
    
    parser.add_argument('--focal_loss_alpha',
                        help='scale for focal loss, focal_loss=focal_loss*focal_loss_alpha',
                        default=1.0,
                        type=float)
    
    parser.add_argument('--changed_lr_mult',
                        help='unchanged_lr_mult=1, changed_lr_mult=?',
                        type=float,
                        default=10.0)
    
    parser.add_argument('--new_lr_mult',
                        help='unchanged_lr_mult=1, new_lr_mult=?',
                        type=float,
                        default=20.0)
    
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
    
    parser.add_argument('--dataset_use_part',
                        help='use images number in dataset (0 for use all)',
                        type=int,
                        default=0)

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
                                 'fcn', 'fcn8s', 'fcn16s', 'fcn32s',
                                 'merge_seg','cross_merge', 'psp_hed'],
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

    parser.add_argument('--upsample_type',
                        help='bilinear or duc upsample',
                        choices=['duc', 'bilinear','fcn'],
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
    
    parser.add_argument('--aux_base_weight',
                        help='aux weight for aux loss, weight for segmentation is 1.0',
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
    
    parser.add_argument('--edge_with_gray',
                        help='add semantic edge with gray edge',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--cross_merge_times',
                        help='cross merge times 1,2 or 3?',
                        type=int,
                        default=1)

    parser.add_argument('--use_momentum',
                        help='use mometnum or not?',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--momentum',
                        help='momentum for batch norm',
                        type=float,
                        default=0.1)

    parser.add_argument('--input_shape',
                        help='input shape',
                        type=int,
                        default=0)

    parser.add_argument('--augmentation',
                        help='true or false to do augmentation',
                        type=str2bool,
                        default=True)
    
    parser.add_argument('--augmentations_blur',
                        help='augmentations blur',
                        type=str2bool,
                        default=True)

    parser.add_argument('--augmentations_rotate',
                        help='augmentations rotate',
                        type=str2bool,
                        default=True)
    
    parser.add_argument('--norm_ways',
                        help='normalize image value ways',
                        choices=['caffe','pytorch','cityscapes','-1,1','0,1'],
                        default='pytorch')
    
    parser.add_argument('--hyperopt',
                        help='tree search or bayes search for hyper parameters',
                        choices=['bayes','skopt','loop'],
                        default='loop')
    
    parser.add_argument('--hyperkey',
                        help='key for single hyperopt,split with , eg: model.l2_reg',
                        default='model.l2_reg')
    
    parser.add_argument('--hyperopt_calls',
                        help='numbers for hyperopt calls',
                        type=int,
                        default=50)
    
    parser.add_argument('--summary_image',
                        help='summary image or not',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--note',
                        help='comment for tensorboard log',
                        default=None)
    
    # save model
    parser.add_argument('--save_model',
                        help='save model or not',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--iou_save_threshold',
                        help='validation iou save threshold',
                        type=float,
                        default=0.5)
    
    # benchmark
    parser.add_argument('--checkpoint_path',
                        help='checkpoint path used in benchmark, eg: model.pkl',
                        default=None)
    
    parser.add_argument('--predict_save_path',
                        help='benchmark result save path',
                        default=None)
    
    # log dir
    parser.add_argument('--log_dir',
                        help='log_dir for tensorboard',
                        default=os.path.expanduser('~/tmp/logs/pytorch'))
    return parser

def get_hyperparams(key,discrete=False):
    discrete_hyper_dict={
            'model.l2_reg':('choices',[1e-5,1e-4,1e-3,1e-2,1e-1]),
            'model.use_lr_mult':('choices',[True,False]),
            'model.changed_lr_mult':('choices',[1,2,5]),
            'model.new_lr_mult':('choices',[1,5,10]),
            'model.backbone_pretrained':('bool',[True,False]),
            'model.backbone_freeze':('bool',[True,False]),
            'model.learning_rate':('choices',[1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3]),
            'model.optimizer':('choices',['sgd','adam']),
            'model.edge_base_weight':('choices',[0.1,0.2,0.5,1.0]),
            'model.use_bn':('bool',[True,False]),
            'model.use_bias':('bool',[True,False]),
            'model.momentum':('choices',[0.1,0.05,0.01]),
            'model.upsample_layer':('choices',[3,4,5]),
            'model.use_class_weight':('bool',[True,False]),
            'model.focal_loss_gamma':('choices',[1.0,2.0,5.0]),
            'model.focal_loss_alpha':('choices',[1.0,5.0,10.0]),
            'model.class_weight_alpha':('choices',[0.2,0.4,0.6,0.8]),
            'model.use_dropout':('bool',[True,False]),
            }
    
    continuous_hyper_dict={
            'model.learning_rate':(float,[1e-5,1e-3]),
            }
    if discrete:
        return discrete_hyper_dict[key]
    else:
        return continuous_hyper_dict[key]