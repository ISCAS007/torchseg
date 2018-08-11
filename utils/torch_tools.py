# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import json
import os
from tensorboardX import SummaryWriter
from utils.metrics import runningScore

def freeze_layer(layer):
    """
    freeze layer weights
    """
    for param in layer.parameters():
        param.requires_grad = False

def do_train_or_val(model, args, train_loader=None, val_loader=None):
    if hasattr(model,'do_train_or_val'):
        print('warning: use do_train_or_val in model'+'*'*30)
        model.do_train_or_val(args,train_loader,val_loader)
        return 0
    
    # use gpu memory
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if hasattr(model,'backbone'):
        if hasattr(model.backbone,'model'):
            model.backbone.model.to(device)
    
    lr=model.config.model.learning_rate if hasattr(model.config.model,'learning_rate') else 0.0001
    if hasattr(model,'optimizer'):
        print('use optimizer in model'+'*'*30)
        optimizer=model.optimizer
    else:
        print('use default optimizer'+'*'*30)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
    
    print('use l1 and l2 reg loss'+'*'*30)
    l1_reg,l2_reg=torch.tensor(data=0,dtype=torch.float,device=device,requires_grad=False),torch.tensor(data=0,dtype=torch.float,device=device,requires_grad=False)
        
    if hasattr(model,'loss_fn'):
        print('use loss function in model'+'*'*30)
        loss_fn=model.loss_fn
    else:
        print('use default loss funtion with ignore_index=%d'%model.ignore_index,'*'*30)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model.ignore_index)

    #TODO metrics supprot ignore_index
    running_metrics = runningScore(model.class_number)

    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(args.log_dir, model.name,
                           model.dataset_name, args.note, time_str)
    checkpoint_path = os.path.join(
        log_dir, "{}_{}_best_model.pkl".format(model.name, model.dataset_name))
    writer=None
    best_iou = 0.6

    loaders = [train_loader, val_loader]
    loader_names = ['train', 'val']
    for epoch in range(args.n_epoch):
        for loader, loader_name in zip(loaders, loader_names):
            if loader is None:
                continue

            if loader_name == 'val':
                if epoch % (1+args.n_epoch//10) == 0:
                    val_image = True
                else:
                    val_image = False

                if val_image or epoch % 10 == 0:
                    val_log = True
                else:
                    val_log = False

                if not val_log:
                    continue

                model.eval()
            else:
                model.train()

            print(loader_name+'.'*50)
            n_step = len(loader)
            losses = []
            reg = 1e-6
            running_metrics.reset()
            for i, (images, labels) in enumerate(loader):
                images = torch.autograd.Variable(images.to(device).float())
                labels = torch.autograd.Variable(labels.to(device).long())

                if loader_name == 'train':
                    optimizer.zero_grad()
                seg_output = model.forward(images)
                loss = loss_fn(input=seg_output, target=labels)

                if loader_name == 'train':
                    l2_loss = torch.autograd.Variable(torch.FloatTensor(1), device=device,requires_grad=True)
                    l1_loss = torch.autograd.Variable(torch.FloatTensor(1), device=device,requires_grad=True)
                    for name, param in model.named_parameters():
                        if 'bias' not in name:
                            l2_loss = l2_loss + torch.norm(param,2)
                            l1_loss = l1_loss + torch.norm(param,1)
                    loss = loss + reg*l2_loss+reg*l1_loss
                    loss.backward()
                    optimizer.step()

                losses.append(loss.data.cpu().numpy())
                predicts = seg_output.data.cpu().numpy().argmax(1)
                trues = labels.data.cpu().numpy()
                running_metrics.update(trues, predicts)
                score, class_iou = running_metrics.get_scores()

                if (i+1) % 5 == 0:
                    print("%s, Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" %
                          (loader_name, epoch+1, args.n_epoch, i, n_step, loss.data))
                    for k, v in score.items():
                        print(k, v)
            
            if writer is None:
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                config_str=json.dumps(model.config,indent=2,sort_keys=True).replace('\n','\n\n').replace('  ','\t')
                writer.add_text(tag='config',text_string=config_str)
                
                # write config to config.txt
                config_path=os.path.join(log_dir,'config.txt')
                config_file=open(config_path,'w')
                json.dump(model.config,config_file,sort_keys=True)
        
            writer.add_scalar('%s/loss' % loader_name,
                              np.mean(losses), epoch)
            writer.add_scalar('%s/l1_loss' % loader_name,
                              reg*l1_loss, epoch)
            writer.add_scalar('%s/l2_loss' % loader_name,
                              reg*l2_loss, epoch)
            writer.add_scalar('%s/acc' % loader_name,
                              score['Overall Acc: \t'], epoch)
            writer.add_scalar('%s/iou' % loader_name,
                              score['Mean IoU : \t'], epoch)
            writer.add_scalar('%s/lr'%loader_name,
                              optimizer.param_groups[0]['lr'],epoch)

            if loader_name == 'val':
                if score['Mean IoU : \t'] >= best_iou:
                    best_iou = score['Mean IoU : \t']
                    state = {'epoch': epoch+1,
                             'miou': best_iou,
                             'model_state': model.state_dict(),
                             'optimizer_state': optimizer.state_dict(), }

                    torch.save(state, checkpoint_path)

                if val_image:
                    print('write image to tensorboard'+'.'*50)
                    idx = np.random.choice(predicts.shape[0])
                    writer.add_image(
                        'val/images', images[idx, :, :, :], epoch)
                    writer.add_image(
                        'val/predicts', torch.from_numpy(predicts[idx, :, :]), epoch)
                    writer.add_image(
                        'val/trues', torch.from_numpy(trues[idx, :, :]), epoch)
                    diff_img = (predicts[idx, :, :] ==
                                trues[idx, :, :]).astype(np.uint8)
                    writer.add_image('val/difference',
                                     torch.from_numpy(diff_img), epoch)

    writer.close()