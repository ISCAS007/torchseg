# -*- coding: utf-8 -*-

from dataset.fbms_dataset import fbms_dataset
from dataset.cdnet_dataset import cdnet_dataset
import torch.utils.data as td
from models.motion_stn import motion_stn, stn_loss, motion_net
from models.motionseg.motion_fcn import motion_fcn,motion_fcn2,motion_fcn_stn
from models.motionseg.motion_unet import motion_unet,motion_unet_stn
from models.motionseg.motion_sparse import motion_sparse
from models.motionseg.motion_psp import motion_psp
from models.motionseg.motion_utils import Metric_Acc,Metric_Mean,get_parser,get_default_config
from utils.torch_tools import init_writer
from dataset.dataset_generalize import image_normalizations
from utils.augmentor import Augmentations
from easydict import EasyDict as edict
import os
import torch
import time
import argparse
import numpy as np
from utils.disc_tools import str2bool
import torchsummary
import sys

if __name__ == '__main__':
    parser=get_parser()
    args = parser.parse_args()
    
    config=get_default_config()
    
    if args.net_name=='motion_psp':
        if args.use_none_layer is False or args.upsample_layer<=3:
            min_size=30*config.psp_scale*2**config.upsample_layer
        else:
            min_size=30*config.psp_scale*2**3
            
        config.input_shape=[min_size,min_size]
        
    for key in config.keys():
        if hasattr(args,key):
            print('{} = {} (default: {})'.format(key,args.__dict__[key],config[key]))
            config[key]=args.__dict__[key]
        else:
            print('{} : (default:{})'.format(key,config[key]))
    
    for key in args.__dict__.keys():
        if key not in config.keys():
            print('{} : unused keys {}'.format(key,args.__dict__[key]))
    
    if args.dataset=='FBMS':
        config['train_path']='dataset/FBMS/Trainingset'
        config['test_path']=config['val_path']='dataset/FBMS/Testset'
    elif args.dataset=='cdnet2014':
        config['root_path']='dataset/cdnet2014'
    else:
        assert False
    
    if config['net_name'] in ['motion_stn','motion_net']:
        model=globals()[config['net_name']]() 
    else:
        model=globals()[config['net_name']](config)
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if args.app=='summary':
        torchsummary.summary(model, ((3, config.input_shape[0], config.input_shape[1]),
                                     (3, config.input_shape[0], config.input_shape[1])))
        sys.exit(0)
        
    normer=image_normalizations(ways='-1,1')
    augmentations = Augmentations()
    dataset_loaders={}
    for split in ['train','val']:
        if args.dataset=='FBMS':
            xxx_dataset=fbms_dataset(config,split,normalizations=normer,augmentations=augmentations)
        else:
            xxx_dataset=cdnet_dataset(config,split,normalizations=normer,augmentations=augmentations)
            print(xxx_dataset.train_set,xxx_dataset.val_set)
        
        xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=args.batch_size,shuffle=True,drop_last=False,num_workers=2)
        dataset_loaders[split]=xxx_loader
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(config['log_dir'], config['net_name'],
                           config['dataset'], config['note'], time_str)
    checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % config['epoch'])
    
    writer=init_writer(config,log_dir)
    
    seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=255)
    
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