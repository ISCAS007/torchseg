# -*- coding: utf-8 -*-
import torch
import torch.nn as TN
from models.backbone import backbone
from models.upsample import get_midnet, get_suffix_net
from tensorboardX import SummaryWriter
from utils.metrics import runningScore
import numpy as np
import time
import os

class psp_aux(TN.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.name=self.__class__.__name__
        
        use_momentum=self.config.model.use_momentum if hasattr(self.config.model,'use_momentum') else False
        self.backbone=backbone(config.model,use_momentum=use_momentum)
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name=self.config.dataset.name
        self.ignore_index=self.config.dataset.ignore_index
        
        self.midnet_input_shape=self.backbone.get_output_shape(self.upsample_layer,self.input_shape)
        self.auxnet_layer=self.config.model.auxnet_layer
        self.auxnet_input_shape=self.backbone.get_output_shape(self.auxnet_layer,self.input_shape)
        self.midnet_out_channels=2*self.midnet_input_shape[1]
        self.auxnet_out_channels=self.auxnet_input_shape[1]
        
        self.midnet=get_midnet(self.config,
                               self.midnet_input_shape,
                               self.midnet_out_channels)
        
        self.auxnet=get_suffix_net(self.config,
                                   self.auxnet_out_channels,
                                   self.class_number,
                                   aux=True)
        
        self.decoder=get_suffix_net(self.config,
                                    self.midnet_out_channels,
                                    self.class_number)
        
        lr = 0.0001
        self.optimizer = torch.optim.Adam(params=[{'params': self.backbone.parameters(), 'lr': lr},
                                             {'params': self.midnet.parameters(), 'lr': 10*lr},
                                             {'params': self.auxnet.parameters(), 'lr': 20*lr},
                                             {'params': self.decoder.parameters(), 'lr': 20*lr}], lr=lr)
        
        print('class number is %d'%self.class_number,'ignore_index is %d'%self.ignore_index,'*'*30)
    
    def forward(self,x):
        main,aux=self.backbone.forward_aux(x,self.upsample_layer,self.auxnet_layer)
#        print('main,aux shape is',main.shape,aux.shape)
        main=self.midnet(main)
        main=self.decoder(main)
        aux=self.auxnet(aux)
        
        return main,aux
    
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
                aux_losses=[]
                main_losses=[]                
                running_metrics.reset()
                for i, (images, labels) in enumerate(loader):
                    images = torch.autograd.Variable(images.to(device).float())
                    labels = torch.autograd.Variable(labels.to(device).long())
                    
                    if loader_name=='train':
                        optimizer.zero_grad()
                    main_output,aux_output = self.forward(images)
                    main_loss = loss_fn(input=main_output, target=labels)
                    aux_loss = loss_fn(input=aux_output,target=labels)
                    loss = main_loss + aux_loss  
                    
                    if loader_name=='train':
                        loss.backward()
                        optimizer.step()
                    
                    losses.append(loss.data.cpu().numpy())
                    main_losses.append(main_loss.data.cpu().numpy())
                    aux_losses.append(aux_loss.data.cpu().numpy())
                    predicts = main_output.data.cpu().numpy().argmax(1)
                    trues = labels.data.cpu().numpy()
                    running_metrics.update(trues,predicts)
                    score, class_iou = running_metrics.get_scores()
                    
                    if (i+1) % 5 == 0:
                        print("%s Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" % (loader_name,epoch+1, args.n_epoch, i, n_step, loss.data))
                        for k, v in score.items():
                            print(k, v)
                    
                writer.add_scalar('%s/loss'%loader_name, np.mean(main_losses), epoch)
                writer.add_scalar('%s/aux_loss'%loader_name, np.mean(aux_losses), epoch)
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