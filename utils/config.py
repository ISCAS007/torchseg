# -*- coding: utf-8 -*-

from easydict import EasyDict as edict
import random
import json
import yaml
import os
import argparse
from utils.disc_tools import str2bool

def get_config():
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
    choices = ['edge', 'global', 'augmentor', 'momentum', 'midnet',
               'backbone', 'dict', 'fractal', 'upsample_type',
               'pretrained', 'summary', 'naive', 'coarse',
               'convert','hyperopt']
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
    
    parser.add_argument('--use_lr_mult',
                        help='use lr mult or not',
                        default=True,
                        type=str2bool)
    
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
    
    return parser

def get_hyperparams(key,discrete=False):
    discrete_hyper_dict={
            'dataset.norm_ways':('choices',['caffe','pytorch','cityscapes','-1,1','0,1']),
            'model.l2_reg':('choices',[1e-5,1e-4,1e-3,1e-2,1e-1]),
            'model.use_lr_mult':('choices',[True,False]),
            'model.backbone_pretrained':('bool',[True,False]),
            'model.backbone_freeze':('bool',[True,False]),
            'model.learning_rate':('choices',[1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3]),
            'model.optimizer':('choices',['sgd','adam'])}
    
    continuous_hyper_dict={
            'model.learning_rate':(float,[1e-5,1e-3]),
            }
    if discrete:
        return discrete_hyper_dict[key]
    else:
        return continuous_hyper_dict[key]