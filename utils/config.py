# -*- coding: utf-8 -*-

from easydict import EasyDict as edict
import random
import json
import yaml
import os
import argparse
from utils.disc_tools import str2bool,lcm_list
from utils.augmentor import get_default_augmentor_config
from dataset.dataset_generalize import get_dataset_generalize_config

def get_default_config():
    config=edict()
    config.class_number=20

    config.layer_preference='last'
    config.input_shape=(224,224)
    config.accumulate=1

    config.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.cityscapes_split=random.choice(['test','val','train'])
    config.resize_shape=(224,224)

    config.batch_size=4
    config.log_dir=os.path.expanduser('~/tmp/logs/pytorch')
    config.note='default'
    config.test='naive'
    config.learning_rate=1e-4
    config.optimizer='adam'
    config.scheduler=None
    config.lr_weight_decay=1e-4
    config.lr_momentum=0.9
    config.use_bn=True
    config.use_bias=True
    config.use_dropout=False
    config.use_lr_mult=False
    config.center_loss=None
    config.center_loss_weight=1.0
    config.use_class_weight=False
    config.class_weight_alpha=0.0
    config.focal_loss_gamma=1.0
    config.focal_loss_alpha=1.0
    config.focal_loss_grad=True
    config.pre_lr_mult=1.0
    config.changed_lr_mult=1.0
    config.new_lr_mult=1.0
    config.use_reg=False
    config.l1_reg=1e-7
    config.l2_reg=1e-5
    config.dataset_name='Cityscapes'
    config.dataset_use_part=0
    config.backbone_name='vgg16'
    config.backbone_pretrained=True
    config.backbone_freeze=False
    config.freeze_layer=3
    config.freeze_ratio=0.0
    config.modify_resnet_head=False
    config.net_name='pspnet'
    config.midnet_scale=5
    config.midnet_name='psp'
    config.n_epoch=100
    config.upsample_type='bilinear'
    config.subclass_sigmoid=True
    config.auxnet_type='bilinear'
    config.upsample_layer=3
    config.deconv_layer=5
    config.auxnet_layer=4
    config.edge_bg_weight=0.01
    config.edge_base_weight=1.0
    config.aux_base_weight=1.0
    config.edge_power=0.9
    config.edge_class_num=2
    config.edge_width=10
    config.edge_seg_order='same'
    config.edge_with_gray=False
    config.cross_merge_times=1
    config.use_none_layer=False
    config.momentum=0.1
    config.input_shape=0
    config.augmentation=True
    config.augmentations_blur=True
    config.use_rotate=True
    config.keep_crop_ratio=True
    config.min_crop_size=None
    config.max_crop_size=None
    config.crop_size_step=0
    config.pad_for_crop=False
    config.norm_ways='pytorch'
    config.hyperopt='loop'
    config.hyperkey='model.l2_reg'
    config.hyperopt_calls=3
    config.summary_image=False
    config.note=None
    config.save_model=False
    config.iou_save_threshold=0.6
    config.checkpoint_path=None
    config.predict_save_path=None
    config.log_dir=os.path.expanduser('~/tmp/logs/pytorch')
    config.attention_type='n'
    config.additional_upsample=False
    config.midnet_pool_sizes = [6, 3, 2, 1]
    config.res_attention=False
    return config

def get_config(args=None):
    config=get_default_config()

    # for test some function
    if args is None:
        return config

    sort_keys=sorted(list(config.keys()))
    for key in sort_keys:
        if hasattr(args,key):
            print('{} = {} (default: {})'.format(key,args.__dict__[key],config[key]))
            config[key]=args.__dict__[key]
        else:
            print('{} : (default:{})'.format(key,config[key]))

    for key in args.__dict__.keys():
        if key not in config.keys():
            print('{} : unknown keys {}'.format(key,args.__dict__[key]))
            config[key]=args.__dict__[key]

    if config.upsample_layer > 3:
        config.use_none_layer = args.use_none_layer = True
    os.environ['use_none_layer'] = str(config.use_none_layer)

    # use_bn,use_dropout will change the variable in local_bn,local_dropout
    # use_bias will affect the bias in upsample
    os.environ['torchseg_use_bn']=str(args.use_bn)
    os.environ['torchseg_use_dropout']=str(args.use_dropout)
    os.environ['torchseg_use_bias']=str(args.use_bias)
    # when use resnet and use_none_layer=True
    os.environ['modify_resnet_head']=str(args.modify_resnet_head)

    config.layer_preference = 'first'
    config.with_edge=False

    if args.input_shape == 0:
        if args.net_name == 'motionnet':
            upsample_ratio=3
            count_size = lcm_list(config.midnet_pool_sizes) * \
                config.midnet_scale*2**upsample_ratio
            input_shape = (count_size, count_size)
        elif args.net_name == 'motion_panet':
            input_shape=(448,448)
        elif args.net_name == 'PSPUNet':
            if args.use_none_layer and args.deconv_layer>=3:
                upsample_ratio=3
            else:
                upsample_ratio=args.deconv_layer

            count_size=lcm_list(config.midnet_pool_sizes)*2**upsample_ratio
            input_shape=(count_size,count_size)
        elif args.midnet_name == 'psp':
            upsample_ratio=args.upsample_layer
            if args.use_none_layer and args.upsample_layer>=3:
                upsample_ratio=3
            count_size = lcm_list(config.midnet_pool_sizes) * \
                config.midnet_scale*2**upsample_ratio
            input_shape = (count_size, count_size)
        else:
            input_shape = (72*8, 72*8)
    else:
        input_shape = (args.input_shape, args.input_shape)

    print('compute input shape is',input_shape,'*'*30)

    config.input_shape = input_shape
    config.midnet_out_channels = 512
    config = get_dataset_generalize_config(
        config, args.dataset_name)
    if config.ignore_index == 0:
        config.class_number = len(config.foreground_class_ids)+1
    else:
        config.class_number = len(config.foreground_class_ids)
    config.resize_shape = input_shape
    config.dataset_name = args.dataset_name.lower()

    if args.net_name in ['psp_edge','merge_seg','cross_merge','psp_hed']:
        config.with_edge = True

    config=get_default_augmentor_config(config)
    config.use_rotate=config.use_rotate
    config.use_imgaug=True

    # image size != network input size != crop size
    if config.keep_crop_ratio is False:
        if args.min_crop_size is None:
            config.min_crop_size=[2*i for i in config.input_shape]
        if args.max_crop_size is None:
            config.max_crop_size=config.min_crop_size

        if not isinstance(config.min_crop_size,(tuple,list)):
            assert config.min_crop_size>0
        else:
            assert min(config.min_crop_size)>0
        if not isinstance(config.max_crop_size,(tuple,list)):
            assert config.max_crop_size>0
        else:
            assert min(config.max_crop_size)>0

        print('min_crop_size is',config.min_crop_size)
        print('max_crop_size is',config.max_crop_size)
        print('crop_size_step is',config.crop_size_step)

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

    d=yaml.load(l,Loader=yaml.FullLoader)
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

    parser.add_argument('--accumulate',
                        type=int,
                        default=1,
                        help='accumulate graident for batch (1)')

    parser.add_argument("--learning_rate",
                        help="learning rate",
                        type=float,
                        default=0.0001)

    parser.add_argument("--optimizer",
                        help="optimizer name",
                        choices=['adam', 'sgd' ,'adamax', 'amsgrad'],
                        default='adam')

    parser.add_argument("--scheduler",
                        help="learning rate scheduler, None or rop, poly_rop, cos_lr",
                        choices=['rop','poly_rop','cos_lr'],
                        default=None)

    parser.add_argument('--lr_weight_decay',
                        help='weight decay for learning rate',
                        type=float,
                        default=1e-4)

    parser.add_argument('--lr_momentum',
                        help='moemntum for learning rate',
                        type=float,
                        default=0.9)

    parser.add_argument('--use_bn',
                        help='use batch norm or not',
                        default=True,
                        type=str2bool)

    # 2018/11/08 change default from False to True
    parser.add_argument('--use_bias',
                        help='use bias or not',
                        default=True,
                        type=str2bool)

    # 2018/11/08 change default from True to False
    parser.add_argument('--use_dropout',
                        help='use dropout or not',
                        default=False,
                        type=str2bool)

    # 2018/11/17 change default value to False
    parser.add_argument('--use_lr_mult',
                        help='use lr mult or not',
                        default=False,
                        type=str2bool)

    # 2019/03/14 center loss
    parser.add_argument('--center_loss',
                        help='use center loss or not',
                        choices=['cos_loss','l1_loss','l2_loss',None],
                        default=None)

    parser.add_argument('--center_loss_weight',
                        help='center loss weight',
                        type=float,
                        default=1.0)

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

    parser.add_argument('--focal_loss_grad',
                       help='use grad or not for pt in focal loss',
                       default=True,
                       type=str2bool)

    parser.add_argument('--pre_lr_mult',
                       help='pretrained layer learning rate',
                       type=float,
                       default=1.0)

    # 2018/11/17 change default value from 10.0 to 1.0
    parser.add_argument('--changed_lr_mult',
                        help='unchanged_lr_mult=1, changed_lr_mult=?',
                        type=float,
                        default=1.0)

    # 2018/11/17 change default value from 20.0 to 1.0
    parser.add_argument('--new_lr_mult',
                        help='unchanged_lr_mult=1, new_lr_mult=?',
                        type=float,
                        default=1.0)

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
                                 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                                 'vgg16_gn','vgg19_gn','se_resnet50','vgg21','vgg21_bn'],
                        default='resnet50')

    # 2018/11/08 change default from False to True
    parser.add_argument('--backbone_pretrained',
                        help='when not use momentum, we can use weights pretrained on imagenet',
                        type=str2bool,
                        default=True)

    parser.add_argument('--backbone_freeze',
                        help='finetune/freeze backbone or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--freeze_layer',
                       help='finetune/freeze the layers in backbone or not',
                       type=int,
                       default=0)

    parser.add_argument('--freeze_ratio',
                       help='finetune/freeze part of the backbone',
                       type=float,
                       default=0.0)

    # change to False at 2018/11/21
    parser.add_argument('--modify_resnet_head',
                       help='modify the head of resnet or not, environment variable!!!',
                       type=str2bool,
                       default=False)

    parser.add_argument('--net_name',
                        help='net name for semantic segmentaion',
                        choices=['pspnet', 'psp_edge', 'psp_global',
                                 'psp_fractal', 'psp_dict', 'psp_aux',
                                 'fcn', 'fcn8s', 'fcn16s', 'fcn32s',
                                 'merge_seg','cross_merge', 'psp_hed',
                                 'motionnet','motion_panet', 'UNet', 'PSPUNet'],
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

    # 2018/11/08 change default from duc to bilinear
    parser.add_argument('--upsample_type',
                        help='bilinear or duc upsample',
                        choices=['duc', 'bilinear','fcn','subclass'],
                        default='bilinear')

    parser.add_argument('--subclass_sigmoid',
                        help='use sigmoid for upsample_subclass or not',
                        type=str2bool,
                        default=True)

    # 2018/11/08 change default from duc to bilinear
    parser.add_argument('--auxnet_type',
                        help='bilinear or duc upsample',
                        choices=['duc', 'bilinear'],
                        default='bilinear')

    parser.add_argument('--upsample_layer',
                        help='layer number for upsample',
                        type=int,
                        default=3)

    parser.add_argument('--deconv_layer',
                        help='layer number for start deconv',
                        type=int,
                        default=5)

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
                        help='cross merge times 1,2 or 3? for cross merge net',
                        type=int,
                        default=1)

    parser.add_argument('--use_none_layer',
                        help='use none layer to replace maxpool in backbone or not?',
                        type=str2bool,
                        default=False)

    parser.add_argument('--momentum',
                        help='momentum for batch norm',
                        type=float,
                        default=0.1)

    parser.add_argument('--input_shape',
                        help='input shape, can be auto computer by midnet_scale and upsample_layer + use_none_layer',
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

    parser.add_argument('--use_rotate',
                        help='augmentations rotate',
                        type=str2bool,
                        default=True)

    parser.add_argument('--keep_crop_ratio',
                       help='when crop the image, keep height:width or not',
                       type=str2bool,
                       default=True)

    # work when keep_crop_ratio=False
    parser.add_argument('--min_crop_size',
                       help='min size for crop the image in preprocessing',
                       type=int,
                       nargs='*',
                       default=None)

    # work when keep_crop_ratio=False
    parser.add_argument('--max_crop_size',
                       help='max size for crop the image in preprocessing',
                       type=int,
                       nargs='*',
                       default=None)

    # work when keep_crop_ratio=False
    parser.add_argument('--crop_size_step',
                       help='crop size step for min_crop_size and max_crop_size',
                       type=int,
                       default=0)

    parser.add_argument('--pad_for_crop',
                        help='padding image and mask for crop or not',
                        type=str2bool,
                        default=False)

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

    # change default value from 50 to 3 in 2018/11/16
    parser.add_argument('--hyperopt_calls',
                        help='numbers for hyperopt calls',
                        type=int,
                        default=3)

    parser.add_argument('--summary_image',
                        help='summary image or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--note',
                        help='comment for tensorboard log',
                        required=True,
                        default=None)

    # save model
    parser.add_argument('--save_model',
                        help='save model or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--iou_save_threshold',
                        help='validation iou save threshold',
                        type=float,
                        default=0.6)

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

    # attention 2019/09/16
    parser.add_argument('--attention_type',
                        help='attention type for motionnet 2019/09/16, {s,c,g,h,n}* ',
                        default='n')

    # additional_upsample 2019/09/24
    parser.add_argument('--additional_upsample',
                        help='use additional upsample for PSPUNet or not',
                        type=str2bool,
                        default=False)

    # pool_sizes 2019/09/25
    parser.add_argument('--midnet_pool_sizes',
                        help='psp module pool size settings',
                        type=int,
                        nargs='*',
                        default=[6,3,2,1])

    # res_attention 2019/09/25
    parser.add_argument('--res_attention',
                        help='use residual attention or not(default False)',
                        type=str2bool,
                        default=False)
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
            'model.optimizer':('choices',['adam', 'adamax', 'amsgrad']),
            'model.edge_base_weight':('choices',[0.1,0.2,0.5,1.0]),
            'model.use_bn':('bool',[True,False]),
            'model.use_bias':('bool',[True,False]),
            'model.momentum':('choices',[0.1,0.05,0.01]),
            'model.upsample_layer':('choices',[3,4,5]),
            'model.use_class_weight':('bool',[True,False]),
            'model.focal_loss_gamma':('choices',[1.0,2.0,5.0]),
            'model.focal_loss_alpha':('choices',[1.0,5.0,10.0]),
            'model.focal_loss_grad':('bool',[True,False]),
            'model.class_weight_alpha':('choices',[0.1, 0.2, 0.3]),
            'model.use_dropout':('bool',[True,False]),
            'model.upsample_type':('choices',['fcn','duc','bilinear']),
            'args.batch_size':('choices',[4,8,16]),
            'args.augmentation':('bool',[True,False]),
            'dataset.norm_ways':('choices',['caffe','pytorch','cityscapes','-1,1','0,1']),
            'model.freeze_layer':('choices',[0,1,2,3,4]),
            'model.freeze_ratio':('choices',[0.3,0.5,0.7]),
            'aug.crop_size_step':('choices',[32,64,128]),
            }

    continuous_hyper_dict={
            'model.learning_rate':(float,[1e-5,1e-3]),
            }
    if discrete:
        return discrete_hyper_dict[key]
    else:
        return continuous_hyper_dict[key]