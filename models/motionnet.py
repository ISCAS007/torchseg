# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as TD
import random
from dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from models.backbone import backbone
from utils.metrics import runningScore,get_scores
from utils.torch_tools import freeze_layer
from models.upsample import upsample_duc,upsample_bilinear
from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
import os

class motionnet(TN.Module):
    def __init__(self,config):
        super(motionnet, self).__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config.model)
        if hasattr(self.config.model,'backbone_lr_ratio'):
            backbone_lr_raio=self.config.model.backbone_lr_ratio
            if backbone_lr_raio==0:
                freeze_layer(self.backbone)
        
        self.upsample_type = self.config.model.upsample_type
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        
        if self.upsample_type=='duc':
            r=2**self.upsample_layer
            feature_map_channel=self.backbone.get_feature_map_channel(self.upsample_layer)
            self.decoder=upsample_duc(feature_map_channel,self.class_number,r)
        elif self.upsample_type=='bilinear':
            output_shape=self.input_shape[0:2]
            feature_map_channel=self.backbone.get_feature_map_channel(self.upsample_layer)
            self.decoder=upsample_bilinear(feature_map_channel,self.class_number,output_shape)
        else:
            assert False,'unknown upsample type %s'%self.upsample_type

    def forward(self, x):        
        feature_map=self.backbone.forward(x,self.upsample_layer)
        x=self.decoder(feature_map)
        return x
    
        
    def train(self,args,trainloader,valloader=None):
        # use gpu memory
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.backbone.model.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)
#        loss_fn=random.choice([torch.nn.NLLLoss(),torch.nn.CrossEntropyLoss()])
        loss_fn=torch.nn.CrossEntropyLoss()
        
        # metrics
        running_metrics = runningScore(self.class_number)
        
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir=os.path.join(args.log_dir,self.name,self.dataset_name,args.note,time_str)
        checkpoint_path=os.path.join(log_dir,"{}_{}_best_model.pkl".format(self.name, self.dataset_name))
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        best_iou=0.0
        for epoch in range(args.n_epoch):
            # set model to train mode
            super(motionnet,self).train(True)
            for i, (images, labels) in enumerate(trainloader):
                images = Variable(images.to(device).float())
                labels = Variable(labels.to(device).long())
                
                optimizer.zero_grad()
                outputs = self.forward(images)
                
#                print(outputs.shape,labels.shape)
    
                loss = loss_fn(input=outputs, target=labels)
    
                loss.backward()
                optimizer.step()
                
                if (i+1) % 20 == 0:
                    print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.data))
                    predicts = outputs.data.cpu().numpy().argmax(1)
                    trues = labels.data.cpu().numpy()
                    running_metrics.update(trues,predicts)
                    score, class_iou = running_metrics.get_scores()
                    for k, v in score.items():
                        print(k, v)
                    running_metrics.reset()
                    
            writer.add_scalar('train/loss', loss.data, epoch)
            writer.add_scalar('train/acc', score['Overall Acc: \t'], epoch)
            writer.add_scalar('train/iou', score['Mean IoU : \t'], epoch)
            
            if valloader is not None:
                super(motionnet,self).train(False)
                running_metrics.reset()
                for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
                    with torch.no_grad:
                        images_val = Variable(images_val.to(device).float())
                        labels_val = Variable(labels_val.to(device).long())
            
                        outputs_val = self.forward(images_val)
                        predicts_val = outputs_val.data.cpu().numpy().argmax(1)
                        trues_val = labels_val.data.cpu().numpy()
                        running_metrics.update(trues_val, predicts_val)
                        
                        loss_val = loss_fn(input=outputs_val, target=labels_val)
        
                score, class_iou = running_metrics.get_scores()
                for k, v in score.items():
                    print(k, v)
                running_metrics.reset()
        
                if score['Mean IoU : \t'] >= best_iou:
                    best_iou = score['Mean IoU : \t']
                    state = {'epoch': epoch+1,
                             'miou': best_iou,
                             'model_state': self.state_dict(),
                             'optimizer_state' : optimizer.state_dict(),}
                    
                    torch.save(state, checkpoint_path)

                writer.add_scalar('val/loss', loss_val.data, epoch)
                writer.add_scalar('val/acc', score['Overall Acc: \t'], epoch)
                writer.add_scalar('val/iou', score['Mean IoU : \t'], epoch)
                
                if (epoch+1) % (1+args.n_epoch//10) == 0:
                    image_num=min(5,predicts_val.shape[0])
                    image_idxs=np.random.permutation(predicts_val.shape[0])[0:image_num]
                    for idx in image_idxs:
                        writer.add_image('val/images_%d'%idx,images_val[idx],epoch)
                        writer.add_image('val/predicts_%d'%idx, predicts_val[idx], epoch)
                        writer.add_image('val/trues_%d'%idx, trues_val[idx], epoch)
        
        writer.close()
                    
if __name__ == '__main__':
    config=edict()
    config.model=edict()
    config.model.upsample_type='duc'
    config.model.upsample_layer=3
    config.model.class_number=20
    config.model.backbone_name='vgg16'
    config.model.layer_preference='last'
    config.model.input_shape=(224,224)
    
    config.dataset=edict()
    config.dataset.root_path='/media/sdb/CVDataset/ObjectSegmentation/archives/Cityscapes_archives'
    config.dataset.cityscapes_split=random.choice(['test','val','train'])
    config.dataset.resize_shape=(224,224)
    config.dataset.name='cityscapes'
    config=get_dataset_generalize_config(config,'Cityscapes')
    train_dataset=dataset_generalize(config.dataset,split='train')
    train_loader=TD.DataLoader(dataset=train_dataset,batch_size=32, shuffle=True,drop_last=False)
    
    val_dataset=dataset_generalize(config.dataset,split='val')
    val_loader=TD.DataLoader(dataset=val_dataset,batch_size=32, shuffle=True,drop_last=False)
    config.args=edict()
    config.args.n_epoch=300
    config.args.log_dir='/home/yzbx/tmp/logs/pytorch'
    config.args.note='image'
    net=motionnet(config)
    net.train(config.args,train_loader,val_loader)