# -*- coding: utf-8 -*-

from dataset.fbms_dataset import fbms_dataset
import torch.utils.data as td
from models.motion_stn import motion_stn, stn_loss, motion_net
from utils.torch_tools import init_writer
from dataset.dataset_generalize import image_normalizations
import os
import torch
import time

class Metric_Acc():
    def __init__(self):
        self.tp=0
        self.count=0
        
    def update(self,predicts,labels):
        pred=torch.ge(predicts,0.5).to(device).long()
        result=(pred==labels).to(torch.float)
        self.tp+=torch.sum(result)
        count=1
        for s in result.shape:
            count*=s
            
        self.count+=count
        
    def get_acc(self):
        return self.tp/self.count
    
    def reset(self):
        self.tp=0
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
        
config={}
config['dataset']='FBMS'
config['net_name']='motion_stn'
config['train_path']='/media/sdb/CVDataset/ObjectSegmentation/FBMS/Trainingset'
config['val_path']='/media/sdb/CVDataset/ObjectSegmentation/FBMS/Testset'
config['frame_gap']=5
config['log_dir']=os.path.expanduser('~/tmp/logs')
config['epoch']=30
config['init_lr']=1e-4
config['stn_loss_weight']=0.01
# features or images
config['stn_object']='images'
config['note']='images'
config['save_model']=True

normer=image_normalizations(ways='-1,1')
dataset_loaders={}
for split in ['train','val']:
    xxx_dataset=fbms_dataset(config,split,normalizations=normer)
    xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=2)
    dataset_loaders[split]=xxx_loader

time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
log_dir = os.path.join(config['log_dir'], config['net_name'],
                       config['dataset'], config['note'], time_str)
checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % config['epoch'])

writer=init_writer(config,log_dir)

model=globals()[config['net_name']]()
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

seg_loss_fn=torch.nn.BCELoss()
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
            mask_loss_value=seg_loss_fn(outputs['masks'][0],labels.float())
            
            if config['net_name']=='motion_stn':
                if config['stn_object']=='features':
                    stn_loss_value=stn_loss(outputs['features'],labels.float())
                elif config['stn_object']=='images':
                    stn_loss_value=stn_loss(outputs['stn_images'],labels.float())
                else:
                    assert False,'unknown stn object %s'%config['stn_object']
                
                total_loss_value=mask_loss_value+stn_loss_value*config['stn_loss_weight']
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
        mean_stn_loss=metric_stn_loss.get_mean()
        mean_mask_loss=metric_mask_loss.get_mean()
        mean_total_loss=metric_total_loss.get_mean()
        writer.add_scalar(split+'/acc',acc,epoch)
        writer.add_scalar(split+'/stn_loss',mean_stn_loss,epoch)
        writer.add_scalar(split+'/mask_loss',mean_mask_loss,epoch)
        writer.add_scalar(split+'/total_loss',mean_total_loss,epoch)
        
        if epoch % 10 == 0:
            print(split,'acc=%0.4f'%acc,'loss=',mean_total_loss)

if config['save_model']:
    torch.save(model.state_dict(),checkpoint_path)

writer.close()