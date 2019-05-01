# -*- coding: utf-8 -*-

import os
from utils.disc_tools import str2bool
from easydict import EasyDict as edict
from models.motionseg.motion_utils import (get_parser,
                                           get_default_config,
                                           fine_tune_config,
                                           get_model,
                                           Metric_Mean,
                                           poly_lr_scheduler,
                                           )
from dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config, image_normalizations
from utils.augmentor import Augmentations
from utils.torch_tools import init_writer
import torch.nn.functional as F
from utils.metrics import runningScore
from tqdm import trange,tqdm
import torch.utils.data as TD
import torch
import time
        
def get_parser_plus():
    parser = get_parser()
    parser.add_argument('--semantic_dataset',
                    help='semantic segmentation dataset name',
                    choices=['ADEChallengeData2016', 'VOC2012', 'Kitti2015',
                             'Cityscapes', 'Cityscapes_Fine', 'Cityscapes_Coarse'],
                    default='Cityscapes')
    return parser

def get_default_config_plus():
    config=get_default_config()
    config.semantic_dataset='Cityscapes'
    return config

def fine_tune_config_plus(config):
    config=fine_tune_config(config)
    return config

def get_config():
    parser=get_parser_plus()
    args = parser.parse_args()
    config=get_default_config_plus()
    
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
    
    config=fine_tune_config_plus(config)
    return config

def get_model_plus(config):
    assert config.class_number!=2
    #todo add config.class_number
    model=get_model(config)
    return model

def get_loaders(config):
    """
    update config.class_number
    return loaders
    """
    dataset_config = edict()
    dataset_config.with_edge=False
    dataset_config.resize_shape=config.input_shape
    dataset_config=get_dataset_generalize_config(
                dataset_config, config.semantic_dataset)
    
    config.class_number=len(dataset_config.foreground_class_ids)+1
    normer=image_normalizations(ways='-1,1')
    batch_size=config.batch_size
    
    dataset_loaders={}
    for split in ['train','val']:
        augmentations = Augmentations() if split=='train' else None
        batch_size=config.batch_size if split=='train' else 1
        drop_last=True if split=='train' else False
        
        xxx_dataset=dataset_generalize(dataset_config, 
                                       split=split,
                                       augmentations=augmentations,
                                       normalizations=normer)
        xxx_loader=TD.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=True,drop_last=drop_last,num_workers=2)
        dataset_loaders[split]=xxx_loader

    return dataset_loaders,config

if __name__ == '__main__':
    config=get_config()
    dataset_loaders,config=get_loaders(config)
    model=get_model(config)
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(config['log_dir'], config['net_name'],
                           config['dataset'], config['note'], time_str)
    checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % config['epoch'])
    
    writer=init_writer(config,log_dir)
    seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=255)
    
    optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]
    if config.optimizer=='adam':
        optimizer = torch.optim.Adam(
                    optimizer_params, lr=config['init_lr'], amsgrad=False)
    else:
        assert config.init_lr>1e-3
        optimizer = torch.optim.SGD(
                    optimizer_params, lr=config['init_lr'], momentum=0.9, weight_decay=1e-4)
    
    metric_mask_loss=Metric_Mean()
    metric_total_loss=Metric_Mean()
    running_metrics = runningScore(config.class_number)
    tqdm_epoch = trange(config['epoch'], desc='{} epochs'.format(config.note), leave=True)
    for epoch in tqdm_epoch:
        for split in ['train','val']:
            if split=='train':
                model.train()
            else:
                model.eval()
            
            metric_mask_loss.reset()
            metric_total_loss.reset()
            tqdm_step = tqdm(dataset_loaders[split], desc='steps', leave=False)
            N=len(dataset_loaders[split])
            for step,(img,gt) in enumerate(tqdm_step):
                images = [torch.autograd.Variable(img.to(device).float()) for i in range(2)]
                gt.unsqueeze_(1)
                origin_labels=torch.autograd.Variable(gt.to(device).long())
                resize_labels=F.interpolate(origin_labels.float(),size=config.input_shape,mode='nearest').long()
                
                if split=='train':
                    poly_lr_scheduler(config,optimizer,
                              iter=epoch*N+step,
                              max_iter=config.epoch*N)
                    optimizer.zero_grad()
                    
                outputs=model.forward(images)

                mask_loss_value=seg_loss_fn(outputs['masks'][0],torch.squeeze(resize_labels,dim=1))
                total_loss_value=mask_loss_value
                    
                origin_mask=F.interpolate(outputs['masks'][0], size=origin_labels.shape[2:4],mode='bilinear',align_corners=True)
                
                metric_mask_loss.update(mask_loss_value.item())
                metric_total_loss.update(total_loss_value.item())
                running_metrics.update(torch.squeeze(origin_labels,dim=1).data.cpu().numpy(),
                                       torch.argmax(origin_mask,dim=1).data.cpu().numpy())
                if split=='train':
                    total_loss_value.backward()
                    optimizer.step()
            mean_mask_loss=metric_mask_loss.get_mean()
            mean_total_loss=metric_total_loss.get_mean()
            score, class_iou = running_metrics.get_scores()
            acc= score['Overall Acc: \t']
            iou = score['Mean IoU : \t']
            
            writer.add_scalar(split+'/mask_loss',mean_mask_loss,epoch)
            writer.add_scalar(split+'/total_loss',mean_total_loss,epoch)
            writer.add_scalar(split+'/acc',acc,epoch)
            writer.add_scalar(split+'/iou',iou,epoch)
            
            if split=='train':
                tqdm_epoch.set_postfix(train_iou=iou)
            else:
                tqdm_epoch.set_postfix(val_iou=iou)
                
            if epoch % 10 == 0:
                print(split,'iou=%0.4f'%iou,
                      'total_loss=',mean_total_loss)
    
    if config['save_model']:
        torch.save(model.state_dict(),checkpoint_path)
    
    writer.close()