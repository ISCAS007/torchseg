import torch
import argparse 
from utils.disc_tools import str2bool
from easydict import EasyDict as edict
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
                        choices=['train','summary'],
                        default='train')
    
    parser.add_argument("--net_name",
                        help="network name",
                        choices=['motion_stn','motion_net','motion_fcn','motion_fcn_stn',
                                 'motion_unet','motion_unet_stn','motion_fcn2','motion_sparse',
                                 'motion_psp'],
                        default='motion_unet')
    
    parser.add_argument('--dataset',
                        help='dataset name (FBMS)',
                        choices=['FBMS','cdnet2014'],
                        default='cdnet2014')
    
    backbone_names=['vgg'+str(number) for number in [11,13,16,19]]
    backbone_names+=[s+'_bn' for s in backbone_names]
    backbone_names+=['resnet50','resnet101']
    parser.add_argument('--backbone_name',
                        help='backbone for motion_fcn and motion_fcn_stn',
                        choices=backbone_names,
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
    
    parser.add_argument('--use_part_number',
                        help='the dataset size, 0 for total dataset',
                        type=int,
                        default=1000)
    
    parser.add_argument('--use_none_layer',
                        help='use nono layer to replace maxpool2d or not',
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
                        help='regular weight for pose mask (1.0)',
                        type=float,
                        default=1.0)
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
    
    config.use_part_number=1000
    config.ignore_outOfRoi=True
    config.dataset='cdnet2014'
    config['frame_gap']=5
    config['log_dir']=os.path.expanduser('~/tmp/logs/motion')
    config['init_lr']=1e-4
    
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
    config.stn_object='images'
    config.sparse_ratio=0.5
    config.sparse_conv=False
    config.psp_scale=5
    
    return config