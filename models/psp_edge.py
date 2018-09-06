# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from models.backbone import backbone
from utils.metrics import runningScore
from models.upsample import get_midnet,get_suffix_net
import numpy as np
from tensorboardX import SummaryWriter
from utils.torch_tools import get_optimizer,poly_lr_scheduler,freeze_layer
from dataset.dataset_generalize import image_normalizations
import json
import time
import os

class psp_edge(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        self.backbone=backbone(config.model)
        if hasattr(self.config.model,'backbone_freeze'):
            if self.config.model.backbone_freeze:
                print('freeze backbone weights'+'*'*30)
                freeze_layer(self.backbone)
        
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        self.edge_class_num=self.config.dataset.edge_class_num
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.midnet_out_channels=2*self.midnet_input_shape[1]
        
        self.midnet=get_midnet(self.config,
                               self.midnet_input_shape,
                               self.midnet_out_channels)
        
        self.seg_decoder=get_suffix_net(self.config,self.midnet_out_channels,self.class_number)
        self.edge_decoder=get_suffix_net(self.config,self.midnet_out_channels,self.edge_class_num)
        
    def forward(self, x):        
        feature_map=self.backbone.forward(x,self.upsample_layer)
        feature_mid=self.midnet(feature_map)
        seg=self.seg_decoder(feature_mid)
        edge=self.edge_decoder(feature_mid)
        return seg,edge
    
    def do_train_or_val(self,args,train_loader=None,val_loader=None):
        config=self.config
        # use gpu memory
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.backbone.model.to(device)
        seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        
        if hasattr(config.dataset, 'edge_class_num'):
            edge_class_num = config.dataset.edge_class_num
        else:
            edge_class_num = 2
        
        edge_bg_weight=config.model.edge_bg_weight
        if edge_class_num==2:
            # edge bg=0, fg=1
            edge_weight_list=[edge_bg_weight,1.0]
        else:
            # edge fg=0, bg=1,2,...,edge_class_num-1
            edge_weight_list=[edge_bg_weight for i in range(edge_class_num)]
            edge_weight_list[0]=1.0
        
        edge_loss_weight=torch.tensor(data=edge_weight_list,dtype=torch.float32).to(device)
        edge_loss_fn=torch.nn.CrossEntropyLoss(weight=edge_loss_weight,ignore_index=self.ignore_index)
        optimizer = get_optimizer(self,config)
        
        # metrics
        running_metrics = runningScore(self.class_number)
        
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir=os.path.join(args.log_dir,self.name,self.dataset_name,args.note,time_str)
        checkpoint_path=os.path.join(log_dir,"{}_{}_best_model.pkl".format(self.name, self.dataset_name))
        writer = None
        best_iou=0.6
        
        power = 0.9
        init_lr = config.model.learning_rate if hasattr(
        config.model, 'learning_rate') else 0.0001
        loaders=[train_loader,val_loader]
        loader_names=['train','val']
        
        if device.type=='cuda':
            gpu_num=torch.cuda.device_count()
            if gpu_num > 1:
                device_ids=[i for i in range(gpu_num)]
                self=torch.nn.DataParallel(self,device_ids=device_ids)
                print('use multi gpu',device_ids,'*'*30)
                time.sleep(3)
            else:
                print('use single gpu','*'*30)
        else:
            print('use cpu only','*'*30)
        
        normalizations=image_normalizations(config.dataset.norm_ways)
        edge_base_weight=config.model.edge_base_weight
        edge_power=config.model.edge_power
        for epoch in range(args.n_epoch):
            edge_weight = edge_base_weight*(1 - epoch/(1.0+args.n_epoch))**edge_power
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
                    # work only for sgd
                    poly_lr_scheduler(optimizer,
                                      init_lr=init_lr,
                                      iter=epoch*len(loader)+i,
                                      max_iter=args.n_epoch*len(loader),
                                      power=power)


                    images = torch.autograd.Variable(images.to(device).float())
                    labels = torch.autograd.Variable(labels.to(device).long())
                    edges = torch.autograd.Variable(edges.to(device).long())
                    
                    if loader_name=='train':
                        optimizer.zero_grad()
                    seg_output,edge_output = self.forward(images)
                    seg_loss=seg_loss_fn(input=seg_output,target=labels)
                    edge_loss=edge_loss_fn(input=edge_output,target=edges)
                    loss = seg_loss + edge_weight*edge_loss  
                    
                    if loader_name=='train':
                        loss.backward()
                        optimizer.step()
                    
                    losses.append(loss.data.cpu().numpy())
                    seg_losses.append(seg_loss.data.cpu().numpy())
                    edge_losses.append(edge_loss.data.cpu().numpy())
                    predicts = seg_output.data.cpu().numpy().argmax(1)
                    edge_pre = edge_output.data.cpu().numpy().argmax(1)
                    trues = labels.data.cpu().numpy()
                    running_metrics.update(trues,predicts)
                    score, class_iou = running_metrics.get_scores()
                    
                    if (i+1) % 5 == 0:
                        print("%s Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" % (loader_name,epoch+1, args.n_epoch, i, n_step, loss.data))
                        for k, v in score.items():
                            print(k, v)
                
                if writer is None:
                    os.makedirs(log_dir, exist_ok=True)
                    writer = SummaryWriter(log_dir=log_dir)
                    config_str = json.dumps(config, indent=2, sort_keys=True).replace(
                        '\n', '\n\n').replace('  ', '\t')
                    writer.add_text(tag='config', text_string=config_str)
    
                    # write config to config.txt
                    config_path = os.path.join(log_dir, 'config.txt')
                    config_file = open(config_path, 'w')
                    json.dump(config, config_file, sort_keys=True)
                    config_file.close()
                    
                writer.add_scalar('edge/edge_weight', edge_weight, epoch)
                writer.add_scalar('%s/loss'%loader_name, np.mean(seg_losses), epoch)
                writer.add_scalar('%s/edge_loss'%loader_name, np.mean(edge_losses), epoch)
                writer.add_scalar('%s/total_loss'%loader_name, np.mean(losses), epoch)
                writer.add_scalar('%s/acc'%loader_name, score['Overall Acc: \t'], epoch)
                writer.add_scalar('%s/iou'%loader_name, score['Mean IoU : \t'], epoch)
                writer.add_scalar('%s/lr' % loader_name,
                              optimizer.param_groups[0]['lr'], epoch)
                
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
                        pixel_scale=255//config.model.class_number
                        edge_pixel_scale=255//config.dataset.edge_class_num
                        idx=np.random.randint(predicts.shape[0])
                        origin_img = images.data.cpu().numpy()
                        origin_img = origin_img.transpose((0, 2, 3, 1))
                        if normalizations is not None:
                            origin_img = normalizations.backward(origin_img)
                        
                        origin_edge = edges.data.cpu().numpy()
                        print('origine edge shape',origin_edge.shape)
#                        origin_edge = origin_edge.transpose((0, 2, 3, 1))
                        
                        writer.add_image(
                        'val/images', origin_img[idx, :, :, :].astype(np.uint8), epoch)
                        writer.add_image(
                        'val/edges', torch.from_numpy(edge_pixel_scale*origin_edge[idx, :, :].astype(np.uint8)), epoch)
                        writer.add_image(
                            'val/predicts', torch.from_numpy((predicts[idx, :, :]*pixel_scale).astype(np.uint8)), epoch)
                        writer.add_image(
                            'val/edges_predicts', torch.from_numpy((edge_pixel_scale*edge_pre[idx, :, :]).astype(np.uint8)), epoch)
                        writer.add_image(
                            'val/trues', torch.from_numpy((trues[idx, :, :]*pixel_scale).astype(np.uint8)), epoch)
                        diff_img = (predicts[idx, :, :] ==
                                    trues[idx, :, :]).astype(np.uint8)
                        writer.add_image('val/difference',
                                         torch.from_numpy(diff_img), epoch)
                
        
        writer.close()