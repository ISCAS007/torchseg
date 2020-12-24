# -*- coding: utf-8 -*-
"""
design patter: factory
"""
from .motion_stn import motion_stn, motion_net
from .motion_fcn import motion_fcn,motion_fcn2,motion_fcn_stn,motion_fcn2_flow,motion_fcn_flow
from .motion_unet import motion_unet,motion_unet_stn,motion_unet_flow
from .motion_panet import motion_panet,motion_panet_flow,motion_panet2,motion_panet2_flow,motion_panet2_stn,get_input_channel
from .motion_sparse import motion_sparse
from .motion_psp import motion_psp
from ..Anet.motion_anet import motion_anet
from .motion_mix import motion_mix,motion_mix_flow
from .motion_filter import motion_filter,motion_filter_flow
from .motion_attention import motion_attention,motion_attention_flow,motion_attention_stn
from .motion_diff import motion_diff
from .changenet import motion_changenet
import segmentation_models_pytorch as smp

def get_baseline_model_keys():
    return ['Unet','DeepLabV3','FPN','PAN','PSPNet','Linknet',
            'UnetPlusPlus','PSPNet','DeepLabV3Plus']

def get_motionseg_model_keys():
    keys=globals().keys()
    keys=[k for k in keys if k.startswith('motion')]
    smp_models=get_baseline_model_keys()

    return keys+smp_models

def get_motionseg_model(config):
    keys=get_motionseg_model_keys()

    if config.net_name in ['motion_stn','motion_net']:
        return globals()[config.net_name]()
    elif config.net_name in keys:
        if config.net_name.startswith('motion'):
            return globals()[config.net_name](config)
        else:
            #like motion_diff, use three output + sigmoid
            in_channels=3+get_input_channel(config.input_format)
            return smp.__dict__[config.net_name](encoder_name=config.backbone_name,
                                     encoder_depth=config.deconv_layer,
                                     encoder_weights='imagenet',
                                     classes=3,
                                     in_channels=in_channels)
    else:
        assert False,'net_name must in {}'.format(keys)

    return None

if __name__ == '__main__':
    g=globals().keys()
    print(g)
    l=locals().keys()
    print(l)