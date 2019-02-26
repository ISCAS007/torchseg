# -*- coding: utf-8 -*-

from dataset.fbms_dataset import fbms_dataset
from dataset.cdnet_dataset import cdnet_dataset
import torch.utils.data as td
from models.motion_stn import motion_stn, stn_loss, motion_net
from models.motionseg.motion_fcn import motion_fcn,motion_fcn_stn
from models.motionseg.motion_unet import motion_unet,motion_unet_stn
from utils.torch_tools import init_writer
from dataset.dataset_generalize import image_normalizations
import os
import torch
import time
import argparse
import numpy as np
from utils.disc_tools import str2bool

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
        self.count+=1
        
    def get_mean(self):
        return self.total/self.count
    
    def reset(self):
        self.total=0
        self.count=0

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net_name",
                        help="network name",
                        choices=['motion_stn','motion_net','motion_fcn','motion_fcn_stn',
                                 'motion_unet','motion_unet_stn'],
                        default='motion_stn')
    
    parser.add_argument('--dataset',
                        help='dataset name (FBMS)',
                        choices=['FBMS','cdnet2014'],
                        default='FBMS')
    
    backbone_names=['vgg'+str(number) for number in [11,13,16,19]]
    backbone_names+=[s+'_bn' for s in backbone_names]
    backbone_names+=['resnet50','resnet101']
    parser.add_argument('--backbone_name',
                        help='backbone for motion_fcn and motion_fcn_stn',
                        choices=backbone_names,
                        default='vgg16')
    
    parser.add_argument('--upsample_layer',
                        help='upsample_layer for motion_fcn',
                        choices=[3,4,5],
                        type=int,
                        default=3)
    
    parser.add_argument('--freeze_layer',
                        help='freeze layer for motion_fcn',
                        choices=[0,1,2,3],
                        type=int,
                        default=1)
    
    parser.add_argument('--use_part_number',
                        help='the dataset size, 0 for total dataset',
                        type=int,
                        default=0)
    
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
                        default='000')
    
    return parser

if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    
    config={}
    config['dataset']=args.dataset
    config['use_part_number']=args.use_part_number
    config['ignore_outOfRoi']=args.ignore_outOfRoi
    config['net_name']=args.net_name
    
    if args.dataset=='FBMS':
        config['train_path']='dataset/FBMS/Trainingset'
        config['test_path']=config['val_path']='dataset/FBMS/Testset'
    elif args.dataset=='cdnet2014':
        config['root_path']='dataset/cdnet2014'
    else:
        assert False
        
    config['frame_gap']=5
    config['log_dir']=os.path.expanduser('~/tmp/logs/motion')
    config['epoch']=30
    config['init_lr']=1e-4
    config['stn_loss_weight']=args.stn_loss_weight
    config['motion_loss_weight']=args.motion_loss_weight
    config['pose_mask_reg']=args.pose_mask_reg
    # features or images
    config['stn_object']=args.stn_object
    config['note']=args.note
    config['save_model']=args.save_model
    config['backbone_name']=args.backbone_name
    config['upsample_layer']=args.upsample_layer
    config['freeze_layer']=args.freeze_layer
    
    normer=image_normalizations(ways='-1,1')
    dataset_loaders={}
    for split in ['train','val']:
        if args.dataset=='FBMS':
            xxx_dataset=fbms_dataset(config,split,normalizations=normer)
        else:
            xxx_dataset=cdnet_dataset(config,split,normalizations=normer)
        
        xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=2)
        dataset_loaders[split]=xxx_loader
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(config['log_dir'], config['net_name'],
                           config['dataset'], config['note'], time_str)
    checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % config['epoch'])
    
    writer=init_writer(config,log_dir)
    
    if config['net_name'] in ['motion_stn','motion_net']:
        model=globals()[config['net_name']]() 
    else:
        model=globals()[config['net_name']](config)
    
    seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=255)
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(
                optimizer_params, lr=config['init_lr'], amsgrad=False)
    
    metric_acc=Metric_Acc()
    metric_stn_loss=Metric_Mean()
    metric_mask_loss=Metric_Mean()
    metric_total_loss=Metric_Mean()
    
    for epoch in range(config['epoch']):
        for split in ['train','val']:
            if split=='train':
                model.train()
            else:
                model.eval()
                
            metric_acc.reset()
            metric_stn_loss.reset()
            metric_mask_loss.reset()
            metric_total_loss.reset()
            for frames,gt in dataset_loaders[split]:
                images = [torch.autograd.Variable(img.to(device).float()) for img in frames]
                labels=torch.autograd.Variable(gt.to(device).long())
                
                if split=='train':
                    optimizer.zero_grad()
                    
                outputs=model.forward(images)
                mask_loss_value=seg_loss_fn(outputs['masks'][0],torch.squeeze(labels,dim=1))
                
                # config['net_name'].find('stn')>=0
                if config['net_name'] in ['motion_stn','motion_fcn_stn','motion_unet_stn']:
                    if config['stn_object']=='features':
                        stn_loss_value=stn_loss(outputs['features'],labels.float(),outputs['pose'],config['pose_mask_reg'])
                    elif config['stn_object']=='images':
                        stn_loss_value=stn_loss(outputs['stn_images'],labels.float(),outputs['pose'],config['pose_mask_reg'])
                    else:
                        assert False,'unknown stn object %s'%config['stn_object']
                    
                    total_loss_value=mask_loss_value*config['motion_loss_weight']+stn_loss_value*config['stn_loss_weight']
                else:
                    stn_loss_value=torch.tensor(0.0)
                    total_loss_value=mask_loss_value
                metric_acc.update(outputs['masks'][0],labels)
                metric_stn_loss.update(stn_loss_value.item())
                metric_mask_loss.update(mask_loss_value.item())
                metric_total_loss.update(total_loss_value.item())
                if split=='train':
                    total_loss_value.backward()
                    optimizer.step()
            acc=metric_acc.get_acc()
            precision=metric_acc.get_precision()
            recall=metric_acc.get_recall()
            fmeasure=metric_acc.get_fmeasure()
            mean_stn_loss=metric_stn_loss.get_mean()
            mean_mask_loss=metric_mask_loss.get_mean()
            mean_total_loss=metric_total_loss.get_mean()
            writer.add_scalar(split+'/acc',acc,epoch)
            writer.add_scalar(split+'/precision',precision,epoch)
            writer.add_scalar(split+'/recall',recall,epoch)
            writer.add_scalar(split+'/fmeasure',fmeasure,epoch)
            writer.add_scalar(split+'/stn_loss',mean_stn_loss,epoch)
            writer.add_scalar(split+'/mask_loss',mean_mask_loss,epoch)
            writer.add_scalar(split+'/total_loss',mean_total_loss,epoch)
            
            if epoch % 10 == 0:
                print(split,'fmeasure=%0.4f'%fmeasure,'loss=',mean_total_loss)
    
    if config['save_model']:
        torch.save(model.state_dict(),checkpoint_path)
    
    writer.close()