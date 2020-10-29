# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import os
import warnings

dataset_root_dict={"fbms":os.path.expanduser('~/cvdataset/FBMS'),
                   "fbms-3d":os.path.expanduser('~/cvdataset/FBMS-3D'),
                   "cdnet2014":os.path.expanduser('~/cvdataset/cdnet2014'),
                   "segtrackv2":os.path.expanduser('~/cvdataset/SegTrackv2'),
                   "bmcnet":os.path.expanduser('~/cvdataset/BMCnet'),
                   "davis2016":os.path.expanduser('~/cvdataset/DAVIS'),
                   "davis2017":os.path.expanduser('~/cvdataset/DAVIS')}

def get_default_config():
    config=edict()
    config.accumulate=1
    config.always_merge_flow=False
    config.app='train'
    config.attention_type='c'
    config.aux_backbone=None
    config.aux_freeze=3
    config.aux_panet=False
    config.backbone_freeze=False
    config.backbone_name='vgg11'
    config.backbone_pretrained=True
    config.batch_size=4
    config.checkpoint_path=None
    config.dataset='cdnet2014'
    config.decode_main_layer=1
    config.deconv_layer=5
    config.epoch=30
    config.exception_value=1.0
    config.filter_feature=None
    config.filter_relu=True
    config.filter_type='main'
    config.frame_gap=5
    config.freeze_layer=1
    config.freeze_ratio=0.0
    config.fusion_type='all'
    config.ignore_pad_area=0
    config.init_lr=1e-4
    config.input_shape=[224,224]
    config.input_format='-'
    config.layer_preference='last'
    config.log_dir=os.path.expanduser('~/tmp/logs/motion')
    config.loss_name='ce'
    config.main_panet=False
    config.max_channel_number=1024
    config.merge_type='concat'
    config.min_channel_number=0
    config.modify_resnet_head=False
    config.motion_loss_weight=1.0
    config.net_name='motion_unet'
    config.norm_stn_pose=False
    config.note='test'
    config.optimizer='adam'
    config.pose_mask_reg=1.0
    config.psp_scale=5
    config.save_model=True
    config.seed=None
    config.share_backbone=None
    config.smooth_ratio=8
    config.sparse_conv=False
    config.sparse_ratio=0.5
    config.stn_loss_weight=1.0
    config.stn_object='images'
    config.subclass_sigmoid=False
    config.upsample_layer=1
    config.upsample_type='bilinear'
    config.use_bias=True
    config.use_bn=False
    config.use_dropout=False
    config.use_none_layer=False
    config.use_part_number=1000
    config.use_sync_bn=False

    return config

def update_default_config(args):
    config=get_default_config()
    if args.net_name=='motion_psp':
        if args.use_none_layer is False or args.upsample_layer<=3:
            min_size=30*config.psp_scale*2**config.upsample_layer
        else:
            min_size=30*config.psp_scale*2**3

        config.input_shape=[min_size,min_size]

    sort_keys=sorted(list(config.keys()))
    for key in sort_keys:
        if hasattr(args,key):
            print('{} = {} (default: {})'.format(key,args.__dict__[key],config[key]))
            config[key]=args.__dict__[key]
        else:
            print('{} : (default:{})'.format(key,config[key]))

    for key in args.__dict__.keys():
        if key not in config.keys():
            print('{} : unused keys {}'.format(key,args.__dict__[key]))

    if config.net_name.find('flow')>=0:
        assert config.frame_gap>0
        config.use_optical_flow=True
        if config.share_backbone is None:
            config.share_backbone=False
    else:
        config.use_optical_flow=False
        if config.share_backbone is None:
            config.share_backbone=True

    config.class_number=2

    # Historical compatibility
    if config.input_format=='_':
        config.use_aux_input=False
    else:
        config.use_aux_input=True

    if config.use_part_number!=1000:
        warnings.warn('optical flow may not valid when use_part_number!=1000')
    return config