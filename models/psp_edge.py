# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from models.backbone import backbone
from utils.metrics import runningScore
from models.upsample import get_midnet,get_suffix_net
import numpy as np
from tensorboardX import SummaryWriter
import time
import os

class psp_edge(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config.model)
        
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.midnet_out_channels=self.config.model.midnet_out_channels
        
        self.midnet=get_midnet(self.config,
                               self.midnet_input_shape,
                               self.midnet_out_channels)
        
        self.seg_decoder=get_suffix_net(self.config,self.midnet_out_channels,self.class_number)
        self.edge_decoder=get_suffix_net(self.config,self.midnet_out_channels,2)
            
        lr = 0.0001
        self.optimizer = torch.optim.Adam(params=[{'params': self.backbone.parameters(), 'lr': lr},
                                             {'params': self.midnet.parameters(),'lr': 10*lr},
                                             {'params': self.seg_decoder.parameters(), 'lr': 20*lr},
                                             {'params': self.edge_decoder.parameters(),'lr':20*lr}], lr=lr)
        
    def forward(self, x):        
        feature_map=self.backbone.forward(x,self.upsample_layer)
        feature_mid=self.midnet(feature_map)
        seg=self.seg_decoder(feature_mid)
        edge=self.edge_decoder(feature_mid)
        return seg,edge
    
    def do_train_or_val(self,args,train_loader=None,val_loader=None):
        # use gpu memory
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.backbone.model.to(device)
        optimizer = self.optimizer
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        # metrics
        running_metrics = runningScore(self.class_number)
        
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir=os.path.join(args.log_dir,self.name,self.dataset_name,args.note,time_str)
        checkpoint_path=os.path.join(log_dir,"{}_{}_best_model.pkl".format(self.name, self.dataset_name))
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        best_iou=0.6
        
        loaders=[train_loader,val_loader]
        loader_names=['train','val']
        for epoch in range(args.n_epoch):
            for loader,loader_name in zip(loaders,loader_names):
                if loader is None:
                    continue
                
                if loader_name == 'val':
                    if epoch % (1+args.n_epoch//10) == 0:
                        val_image=True
                    else:
                        val_image=False
                        
                    if val_image or epoch % 10 == 0:
                        val_log=True
                    else:
                        val_log=False
                    
                    if not val_log:
                        continue
                    
                    self.eval()
                else:
                    self.train()
                
                print(loader_name+'.'*50)
                n_step=len(loader)
                losses=[]
                edge_losses=[]
                seg_losses=[]                
                running_metrics.reset()
                for i, (images, labels, edges) in enumerate(loader):
                    images = torch.autograd.Variable(images.to(device).float())
                    labels = torch.autograd.Variable(labels.to(device).long())
                    edges = torch.autograd.Variable(edges.to(device).long())
                    
                    if loader_name=='train':
                        optimizer.zero_grad()
                    seg_output,edge_output = self.forward(images)
                    seg_loss = loss_fn(input=seg_output, target=labels)
                    edge_loss = loss_fn(input=edge_output,target=edges)
                    loss = seg_loss + edge_loss  
                    
                    if loader_name=='train':
                        loss.backward()
                        optimizer.step()
                    
                    losses.append(loss.data.cpu().numpy())
                    seg_losses.append(seg_loss.data.cpu().numpy())
                    edge_losses.append(edge_loss.data.cpu().numpy())
                    predicts = seg_output.data.cpu().numpy().argmax(1)
                    trues = labels.data.cpu().numpy()
                    running_metrics.update(trues,predicts)
                    score, class_iou = running_metrics.get_scores()
                    
                    if (i+1) % 5 == 0:
                        print("%s Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" % (loader_name,epoch+1, args.n_epoch, i, n_step, loss.data))
                        for k, v in score.items():
                            print(k, v)
                    
                writer.add_scalar('%s/loss'%loader_name, np.mean(seg_losses), epoch)
                writer.add_scalar('%s/edge_loss'%loader_name, np.mean(edge_losses), epoch)
                writer.add_scalar('%s/total_loss'%loader_name, np.mean(losses), epoch)
                writer.add_scalar('%s/acc'%loader_name, score['Overall Acc: \t'], epoch)
                writer.add_scalar('%s/iou'%loader_name, score['Mean IoU : \t'], epoch)
            
                if loader_name=='val':
                    if score['Mean IoU : \t'] >= best_iou:
                        best_iou = score['Mean IoU : \t']
                        state = {'epoch': epoch+1,
                                 'miou': best_iou,
                                 'model_state': self.state_dict(),
                                 'optimizer_state' : optimizer.state_dict(),}
                        
                        torch.save(state, checkpoint_path)
                    
                    if val_image:
                        print('write image to tensorboard'+'.'*50)
                        idx=np.random.randint(predicts.shape[0])
                        writer.add_image('val/images',images[idx,:,:,:],epoch)
                        writer.add_image('val/predicts', torch.from_numpy(predicts[idx,:,:]), epoch)
                        writer.add_image('val/trues', torch.from_numpy(trues[idx,:,:]), epoch)
                        diff_img=(predicts[idx,:,:]==trues[idx,:,:]).astype(np.uint8)
                        writer.add_image('val/difference', torch.from_numpy(diff_img), epoch)
                
        
        writer.close()