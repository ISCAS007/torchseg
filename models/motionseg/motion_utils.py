import torch
import argparse 
from utils.disc_tools import str2bool
from dataset.fbms_dataset import fbms_dataset
from dataset.cdnet_dataset import cdnet_dataset
from dataset.segtrackv2_dataset import segtrackv2_dataset
from dataset.bmcnet_dataset import bmcnet_dataset
from dataset.dataset_generalize import image_normalizations
from utils.augmentor import Augmentations
from models.motionseg.motion_fcn import motion_fcn,motion_fcn2,motion_fcn_stn,motion_fcn2_flow,motion_fcn_flow
from models.motionseg.motion_unet import motion_unet,motion_unet_stn,motion_unet_flow
from models.motionseg.motion_panet import motion_panet,motion_panet_flow,motion_panet2,motion_panet2_flow,motion_panet2_stn
from models.motionseg.motion_sparse import motion_sparse
from models.motionseg.motion_psp import motion_psp
from models.Anet.motion_anet import motion_anet
from models.motionseg.motion_mix import motion_mix,motion_mix_flow
from models.motionseg.motion_filter import motion_filter,motion_filter_flow
from easydict import EasyDict as edict
import torch.utils.data as td
import os

class Metric_Acc():
    def __init__(self):
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype=torch.int64
#        self.tp=torch.tensor(0,dtype=self.dtype,device=device)
#        self.fp=torch.tensor(0,dtype=self.dtype,device=device)
#        self.tn=torch.tensor(0,dtype=self.dtype,device=device)
#        self.fn=torch.tensor(0,dtype=self.dtype,device=device)
#        self.count=torch.tensor(0,dtype=self.dtype,device=device)
        self.tp=0
        self.fp=0
        self.tn=0
        self.fn=0
        self.count=0
        
        ## compute avg_p,avg_r,avg_f
        self.sum_p=0
        self.sum_r=0
        self.sum_f=0
        self.img_count=0
        
    def update(self,predicts,labels):
        # print(labels.shape,predicts.shape)
        if labels.shape != predicts.shape:
            pred=torch.argmax(predicts,dim=1,keepdim=True).type_as(labels)
        else:
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pred=(predicts>0.5).type_as(labels)
        
        self.tp+=torch.sum(((pred==1) & (labels==1)).to(self.dtype))
        self.fp+=torch.sum(((pred==1) & (labels==0)).to(self.dtype))
        self.tn+=torch.sum(((pred==0) & (labels==0)).to(self.dtype))
        self.fn+=torch.sum(((pred==0) & (labels==1)).to(self.dtype))
            
        self.count+=torch.sum(((labels<=1)).to(self.dtype))
        
        assert self.tp+self.fp+self.tn+self.fn==self.count, \
        'tp={}; fp={}; tn={}; fn={}; count={} \n pred {}, labels {}'.format(self.tp,
            self.fp,self.tn,self.fn,self.count,torch.unique(pred),torch.unique(labels))
        
        b=pred.size(0)
        for i in range(b):
            tp=torch.sum(((pred[i]==1) & (labels[i]==1)).to(torch.float32))
            fp=torch.sum(((pred[i]==1) & (labels[i]==0)).to(torch.float32))
#            tn=torch.sum(((pred[i]==0) & (labels[i]==0)).to(torch.float32))
            fn=torch.sum(((pred[i]==0) & (labels[i]==1)).to(torch.float32))
            
            if tp+fn==0:
                r=1
            else:
                r=tp/(tp+fn)
            
            if tp+fp==0:
                p=1
            else:
                p=tp/(tp+fp)
            
            if p+r==0:
                f=1
            else:
                f=2*p*r/(p+r)
            
            self.sum_p+=p
            self.sum_r+=r
            self.sum_f+=f
        self.img_count+=b
    
    def get_avg_metric(self):
        return self.sum_p/self.img_count,self.sum_r/self.img_count,self.sum_f/self.img_count
    
    def get_acc(self):
        return (self.tp+self.tn).to(torch.float32)/(self.count.to(torch.float32)+1e-5)
    
    def get_precision(self):
        return self.tp.to(torch.float32)/((self.tp+self.fp).to(torch.float32)+1e-5)
    
    def get_recall(self):
        return self.tp.to(torch.float32)/((self.tp+self.fn).to(torch.float32)+1e-5)
    
    def get_fmeasure(self):
        p=self.get_precision()
        r=self.get_recall()
        return 2*p*r/(p+r+1e-5)
    
    def reset(self):
        self.tp=0
        self.fp=0
        self.tn=0
        self.fn=0
        self.count=0
        
        self.sum_p=0
        self.sum_r=0
        self.sum_f=0
        self.img_count=0
        
        
class Metric_Mean():
    def __init__(self):
        self.total=0
        self.count=0
        
    def update(self,value):
        self.total+=value
        self.count+=1.0
        
    def get_mean(self):
        return self.total/self.count
    
    def reset(self):
        self.total=0
        self.count=0

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--app',
                        help='application name',
                        choices=['train','summary','dataset'],
                        default='train')
    
    parser.add_argument("--net_name",
                        help="network name",
                        choices=['motion_stn','motion_net','motion_fcn','motion_fcn_stn',
                                 'motion_unet','motion_unet_stn','motion_fcn2','motion_sparse',
                                 'motion_psp','motion_fcn2_flow','motion_fcn_flow','motion_unet_flow',
#                                 'motion_panet','motion_panet_flow','motion_anet',
                                 'motion_panet2','motion_panet2_flow','motion_mix','motion_mix_flow',
                                 'motion_panet2_stn','motion_filter','motion_filter_flow'],
                        default='motion_unet')
    
    parser.add_argument('--dataset',
                        help='dataset name (FBMS)',
                        choices=['FBMS','cdnet2014','segtrackv2','BMCnet','all'],
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
    
    parser.add_argument('--flow_backbone',
                        help='backbone for flow network(vgg11), currently motion_panet2 support only',
                        choices=['vgg'+str(number) for number in [11,13,16,19]],
                        default='vgg11')
    
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
    
    parser.add_argument('--frame_gap',
                        help='the frame gap for dataset(5)',
                        type=int,
                        default=5)
    
    parser.add_argument('--use_none_layer',
                        help='use nono layer to replace maxpool2d or not',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--use_aux_input',
                        help='use aux image as input or not(True)',
                        type=str2bool,
                        default=True)
    
    parser.add_argument('--always_merge_flow',
                        help='@deprecated merge flow at every deconv layer or not (False)',
                        type=str2bool,
                        default=False)
    
    parser.add_argument('--ignore_outOfRoi',
                        help='padding for out of roi or not, false for padding',
                        type=str2bool,
                        default=True)
    
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
    
    parser.add_argument('--aux_freeze',
                        help='freeze layer for aux backbone',
                        type=int,
                        default=3)
    
    parser.add_argument('--optimizer',
                        help='optimizer sgd/adam',
                        choices=['adam','sgd'],
                        default='adam')
    
    return parser

def get_default_config():
    config=edict()
    config.input_shape=[224,224]
    config.backbone_name='vgg11'
    config.upsample_layer=1
    config.deconv_layer=5
    config.use_none_layer=False
    config.net_name='motion_unet'
    config.backbone_freeze=False
    config.backbone_pretrained=True
    config.freeze_layer=1
    config.freeze_ratio=0.0
    config.modify_resnet_head=False
    config.layer_preference='last'
    config.merge_type='concat'
    config.always_merge_flow=False
    config.use_aux_input=True
    
    config.use_part_number=1000
    config.ignore_outOfRoi=True
    config.dataset='cdnet2014'
    config.frame_gap=5
    config.log_dir=os.path.expanduser('~/tmp/logs/motion')
    config.init_lr=1e-4
    
    config.use_bn=False
    config.use_dropout=False
    config.use_bias=True
    config.upsample_type='bilinear'
    config.note='test'
    config.batch_size=4
    config.epoch=30
    config.app='train'
    config.save_model=True
    config.stn_loss_weight=1.0
    config.motion_loss_weight=1.0
    config.pose_mask_reg=1.0
    config.norm_stn_pose=False
    config.stn_object='images'
    config.sparse_ratio=0.5
    config.sparse_conv=False
    config.psp_scale=5
    
    config.upsample_type='bilinear'
    config.subclass_sigmoid=False
    config.flow_backbone='vgg11'
    config.main_panet=False
    config.aux_panet=False
    # note, false for flow
    config.share_backbone=None
    config.fusion_type='all'
    config.decode_main_layer=1
    config.min_channel_number=0
    config.smooth_ratio=8
    config.filter_type='main'
    config.aux_backbone=None
    config.optimizer='adam'
    config.aux_freeze=3
    return config

def get_other_config(config):
    if config.net_name.find('flow')>=0:
        config.use_optical_flow=True
        if config.share_backbone is None:
            config.share_backbone=False
    else:
        config.use_optical_flow=False
        if config.share_backbone is None:
            config.share_backbone=True
        
    if config.dataset=='FBMS':
        config['root_path']=os.path.expanduser('~/cvdataset/FBMS')
    elif config.dataset=='cdnet2014':
        config['root_path']=os.path.expanduser('~/cvdataset/cdnet2014')
    elif config.dataset=='segtrackv2':
        config['root_path']=os.path.expanduser('~/cvdataset/SegTrackv2')
    elif config.dataset=='BMCnet':
        config['root_path']=os.path.expanduser('~/cvdataset/BMCnet')
    elif config.dataset=='all':
        pass
    else:
        assert False
        
    return config

def get_dataset(config,split):
    normer=image_normalizations(ways='-1,1')
    augmentations = Augmentations()
    config=get_other_config(config)
    if config.dataset=='FBMS':
        xxx_dataset=fbms_dataset(config,split,normalizations=normer,augmentations=augmentations)
    elif config.dataset=='cdnet2014':
        xxx_dataset=cdnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
        print(xxx_dataset.train_set,xxx_dataset.val_set)
    elif config.dataset=='segtrackv2':
        xxx_dataset=segtrackv2_dataset(config,split,normalizations=normer,augmentations=augmentations)
    elif config.dataset=='BMCnet':
        xxx_dataset=bmcnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
    elif config.dataset=='all':
        config['root_path']=os.path.expanduser('~/cvdataset/FBMS')
        fbms=fbms_dataset(config,split,normalizations=normer,augmentations=augmentations)
        config['root_path']=os.path.expanduser('~/cvdataset/cdnet2014')
        cdnet=cdnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
        config['root_path']=os.path.expanduser('~/cvdataset/SegTrackv2')
        segtrackv2=segtrackv2_dataset(config,split,normalizations=normer,augmentations=augmentations)
        config['root_path']=os.path.expanduser('~/cvdataset/BMCnet')
        bmcnet=bmcnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
        xxx_dataset=td.ConcatDataset([fbms,cdnet,segtrackv2,bmcnet])
    else:
        assert False,'dataset={}'.format(config.dataset)
        
    return xxx_dataset

def get_model(config):
    model=globals()[config['net_name']](config)
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