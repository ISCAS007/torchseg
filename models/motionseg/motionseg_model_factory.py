# -*- coding: utf-8 -*-
"""
design patter: factory
"""
from models.motionseg.motion_fcn import motion_fcn,motion_fcn2,motion_fcn_stn,motion_fcn2_flow,motion_fcn_flow
from models.motionseg.motion_unet import motion_unet,motion_unet_stn,motion_unet_flow
from models.motionseg.motion_panet import motion_panet,motion_panet_flow,motion_panet2,motion_panet2_flow,motion_panet2_stn
from models.motionseg.motion_sparse import motion_sparse
from models.motionseg.motion_psp import motion_psp
from models.Anet.motion_anet import motion_anet
from models.motionseg.motion_mix import motion_mix,motion_mix_flow
from models.motionseg.motion_filter import motion_filter,motion_filter_flow
from models.motionseg.motion_attention import motion_attention,motion_attention_flow
from models.motionseg.motion_diff import motion_diff


def get_motionseg_model_keys():
    keys=globals().keys()
    keys=[k for k in keys if k.startswith('motion')]
    return keys

def get_motionseg_model(config):
    keys=globals().keys()
    keys=[k for k in keys if k.startswith('motion')]
    if config.net_name in keys:
        return globals()[config.net_name](config)
    else:
        assert False,'net_name must in {}'.format(keys)

    return None

if __name__ == '__main__':
    g=globals().keys()
    print(g)
    l=locals().keys()
    print(l)