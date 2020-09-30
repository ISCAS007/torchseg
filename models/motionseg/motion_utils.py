import torch
import argparse
from utils.disc_tools import str2bool
from dataset.motionseg_dataset_factory import get_motionseg_dataset
from models.motionseg.motionseg_model_factory import get_motionseg_model,get_motionseg_model_keys
from utils.disc_tools import get_newest_file
import os
import glob

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--app',
                        help='application name, train(train and val), test(run benchmark for val dataset, save model output), benchmark(run benchmark for test dataset), summary(view model), dataset(view dataset), viz(visualization)',
                        choices=['train','summary','dataset','viz','test','benchmark','fps'],
                        default='train')

    parser.add_argument("--net_name",
                        help="network name",
                        choices=get_motionseg_model_keys(),
                        default='motion_unet')

    parser.add_argument('--dataset',
                        help='dataset name (FBMS)',
                        choices=['FBMS','cdnet2014','segtrackv2','BMCnet','DAVIS2017','DAVIS2016','all','all2','all3'],
                        default='cdnet2014')

    backbone_names=['vgg'+str(number) for number in [11,13,16,19,21]]
    backbone_names+=[s+'_bn' for s in backbone_names]
    backbone_names+=['resnet50','resnet101','MobileNetV2','se_resnet50','Anet']
    parser.add_argument('--backbone_name',
                        help='backbone for motion_fcn and motion_fcn_stn',
                        choices=backbone_names,
                        default='vgg11')

    parser.add_argument('--aux_backbone',
                        help='backbone for aux, currently only motion_panet2,motion_filter',
                        choices=backbone_names,
                        default=None)

    parser.add_argument('--input_format',
                        help='input format [Background(B),Neighbor Image(N),Optical Flow(O),Neighbor GroundTruth(G),None(-)] (-)',
                        default='-')

    parser.add_argument('--accumulate',
                        help='batch size accumulate (1)',
                        type=int,
                        default=1)

    parser.add_argument('--batch_size',
                        help='batch size for experiment',
                        type=int,
                        default=4)

    parser.add_argument('--epoch',
                        help='epoch for experiment',
                        type=int,
                        default=30)

    parser.add_argument('--upsample_layer',
                        help='upsample_layer for motion_fcn',
                        choices=[0,1,2,3,4,5],
                        type=int,
                        default=0)

    parser.add_argument('--freeze_layer',
                        help='freeze layer for motion_fcn',
                        choices=[0,1,2,3,4,5],
                        type=int,
                        default=1)

    parser.add_argument('--deconv_layer',
                        help='deconv layer for motion_unet',
                        choices=[1,2,3,4,5],
                        type=int,
                        default=5)

    parser.add_argument('--upsample_type',
                        help='upsample type for motion_unet (bilinear)',
                        choices=['bilinear','subclass','mid_decoder','smooth','duc'],
                        default='bilinear')

    parser.add_argument('--subclass_sigmoid',
                        help='use sigmoid or not in subclass upsample (False)',
                        type=str2bool,
                        default=False)

    parser.add_argument('--use_part_number',
                        help='the dataset size, 0 for total dataset',
                        type=int,
                        default=1000)

    parser.add_argument('--ignore_pad_area',
                        help='ignore pad area in loss function',
                        type=int,
                        default=0)

    parser.add_argument('--frame_gap',
                        help='the frame gap for dataset, 0 for random frame gap (5)',
                        type=int,
                        default=5)

    parser.add_argument('--use_none_layer',
                        help='use nono layer to replace maxpool2d or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--always_merge_flow',
                        help='@deprecated merge flow at every deconv layer or not (False)',
                        type=str2bool,
                        default=False)

    parser.add_argument("--save_model",
                        help="save model or not",
                        type=str2bool,
                        default=True)

    parser.add_argument("--stn_loss_weight",
                        help="stn loss weight (1.0)",
                        type=float,
                        default=1.0)
    parser.add_argument("--motion_loss_weight",
                        help="motion mask loss weight (1.0)",
                        type=float,
                        default=1.0)
    parser.add_argument('--pose_mask_reg',
                        help='regular weight for pose mask (0.0)',
                        type=float,
                        default=0.0)

    parser.add_argument('--norm_stn_pose',
                        help='norm stn pose or not (False)',
                        type=str2bool,
                        default=False)

    parser.add_argument("--stn_object",
                        help="use feature or images to compute stn loss",
                        choices=['images','features'],
                        default='images')

    parser.add_argument('--use_sync_bn',
                        help='use distribution trainning and sync batch norm(False)',
                        type=str2bool,
                        default=False)

    parser.add_argument("--note",
                        help="note for model",
                        default='test')

    # motion_sparse
    parser.add_argument('--sparse_ratio',
                        help='sparse ratio for motion_sparse',
                        type=float,
                        default=0.5)

    parser.add_argument('--sparse_conv',
                        help='use sparse conv for motion_sparse or not',
                        type=str2bool,
                        default=False)

    parser.add_argument('--input_shape',
                        help='input shape for model',
                        nargs=2,
                        type=int,
                        default=[224,224])

    parser.add_argument('--main_panet',
                        help='use main panet or not(False) currently motion_panet2 support only',
                        type=str2bool,
                        default=False)

    parser.add_argument('--aux_panet',
                        help='use aux panet or not(False) currently motion_panet2 support only',
                        type=str2bool,
                        default=False)

    parser.add_argument('--share_backbone',
                        help='share the backbone for main and aux(None), currently motion_panet2 support only',
                        type=str2bool,
                        default=None)

    parser.add_argument('--fusion_type',
                        help='type for fusion the aux with main(all), currently motion_panet2 and motion_unet_flow support only, first=HR,last=LR',
                        choices=['all','first','last', 'HR','LR'],
                        default='all')

    parser.add_argument('--decode_main_layer',
                        help='the number for decode layers, currently only motion_panet2 support',
                        type=int,
                        default=1)

    parser.add_argument('--min_channel_number',
                        help='the min channel number for decode layers, currently only motion_panet2 support',
                        type=int,
                        default=0)

    parser.add_argument('--max_channel_number',
                        help='the max channel number for decode layers, currently only motion_panet2 support',
                        type=int,
                        default=1024)

    parser.add_argument('--init_lr',
                        help='the learing rate for trainning model',
                        type=float,
                        default=1e-4)

    parser.add_argument('--smooth_ratio',
                        help='smooth ratio for smooth upsample, currently only motion_unet support',
                        type=int,
                        default=8)

    parser.add_argument('--filter_type',
                        help='filter type for motion_filter(main)',
                        choices=['main','all'],
                        default='main')

    parser.add_argument('--filter_feature',
                        help='filtered feature for motion_filter(aux for frame and flow, all for two frames)',
                        choices=['aux','all'],
                        default=None)

    parser.add_argument('--attention_type',
                        help='attention type for motion_attention, s for spatial attention, c for channel attention, g for global attention, n for no attention, choices like s,c,g,sc,cs,sg,gs,cg,gc,scg,...',
                        default='c')

    parser.add_argument('--aux_freeze',
                        help='freeze layer for aux backbone',
                        type=int,
                        default=3)

    parser.add_argument('--optimizer',
                        help='optimizer sgd/adam',
                        choices=['adam','sgd'],
                        default='adam')

    parser.add_argument('--filter_relu',
                        help='use relu in motion_filter or not',
                        type=str2bool,
                        default=True)

    parser.add_argument('--exception_value',
                        help='exception value for FBMS F-Measure',
                        type=float,
                        default=1.0)

    parser.add_argument('--loss_name',
                        help='use iou loss or not, iou loss not support ignore_index',
                        choices=['ce','dice','iou'],
                        default='ce')

    parser.add_argument('--seed',
                        help='distribution training seed(None)',
                        type=int,
                        default=None)

    # 2020/01/08
    parser.add_argument('--checkpoint_path',
                        help='the checkpoint path to load for test and validation',
                        default=None)

    return parser

def fine_tune_config(config):
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
    return config

def get_dataset(config,split):
    config=fine_tune_config(config)
    return get_motionseg_dataset(config,split)

def get_model(config):
    return get_motionseg_model(config)

def get_checkpoint_path(config):
    if config.checkpoint_path is None:
        log_dir = os.path.join(config['log_dir'], config['net_name'],
                               config['dataset'], config['note'])

        checkpoint_path_list=glob.glob(os.path.join(log_dir,'*','*.pkl'))
        assert len(checkpoint_path_list)>0,f'{log_dir} do not have checkpoint'
        checkpoint_path = get_newest_file(checkpoint_path_list)
    else:
        checkpoint_path=config.checkpoint_path

    return checkpoint_path

def get_load_convert_model(config):
    """
    get load and convert model
    """
    checkpoint_path=get_checkpoint_path(config)
    model=get_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    return model

def poly_lr_scheduler(config, optimizer, iter,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if config.optimizer=='adam':
        pass
    elif config.optimizer=='sgd':
        assert iter<=max_iter
        scale = (1 - iter/(1.0+max_iter))**power
        for i, p in enumerate(optimizer.param_groups):
            # optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['initial_lr'] * scale
            optimizer.param_groups[i]['lr']=config.init_lr*scale
