# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import json
import os
from tensorboardX import SummaryWriter
from utils.metrics import get_scores,runningScore
from dataset.dataset_generalize import image_normalizations
from tqdm import tqdm,trange


def add_image(summary_writer, name, image, step):
  """
  add numpy/tensor image with shape [2d,3d,4d] to summary
  support [h,w] [h,w,1], [1,h,w], [h,w,3], [3,h,w], [b,h,w,c], [b,c,h,w] shape
  combie with numpy,tensor
  
  note: data should in right range such as [0,1], [0,255] and right dtype
  dtype: np.uint8 for [0,255]
  """
  if isinstance(image,np.ndarray):
    if image.ndim==2:
      summary_writer.add_image(name,torch.from_numpy(image),step)
    elif image.ndim==3:
      a,b,c=image.shape
      if min(a,c)==1:
        if a==1:
          summary_writer.add_image(name,torch.from_numpy(image[0,:,:]),step)
        else:
          summary_writer.add_image(name,torch.from_numpy(image[:,:,0]),step)
      else:
        if a==3:
          summary_writer.add_image(name,torch.from_numpy(image),step)
        elif c==3:
          summary_writer.add_image(name,image,step)
        else:
          assert False,'unexcepted image shape %s'%str(image.shape)
    elif image.ndim==4:
      add_image(summary_writer, name, image[0,:,:,:], step)
    else:
      assert False,'unexcepted image shape %s'%str(image.shape)
  elif isinstance(image,torch.Tensor):
    if image.dim()==2:
      summary_writer.add_imge(name,image,step)
    elif image.dim()==3:
      a,b,c=image.shape
      if min(a,c)==1:
        if a==1:
          summary_writer.add_image(name,image[0,:,:],step)
        else:
          summary_writer.add_image(name,image[:,:,0],step)
      else:
        if a==3:
          summary_writer.add_image(name,image,step)
        elif c==3:
          summary_writer.add_image(name,image.data.cpu().numpy(),step)
        else:
          assert False,'unexcepted image shape %s'%str(image.shape)
    elif image.dim()==4:
      add_image(summary_writer, name, image[0,:,:,:], step)
    else:
      assert False,'unexcepted image shape %s'%str(image.shape)
  else:
    assert False,'unknown type %s'%type(image)

def freeze_layer(layer):
    """
    freeze layer weights
    """
    for param in layer.parameters():
        param.requires_grad = False


def poly_lr_scheduler(optimizer, init_lr, iter,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if type(optimizer) != torch.optim.SGD:
        return init_lr

    if iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/(1.0+max_iter))**power
    for i,p in enumerate(optimizer.param_groups):
        lr_mult = p['lr_mult'] if 'lr_mult' in p.keys() else 1.0
        optimizer.param_groups[i]['lr'] = lr*lr_mult

    return lr


def get_optimizer(model, config):
    init_lr = config.model.learning_rate if hasattr(
        config.model, 'learning_rate') else 0.0001
    optimizer_str = config.model.optimizer if hasattr(
        config.model, 'optimizer') else 'adam'
            
    if hasattr(model,'optimizer_params'):
        optimizer_params = model.optimizer_params
        for i,p in enumerate(optimizer_params):
            lr_mult = p['lr_mult'] if 'lr_mult' in p.keys() else 1.0
            optimizer_params[i]['lr']=init_lr*lr_mult
    else:
        optimizer_params = [
            p for p in model.parameters() if p.requires_grad]

    if optimizer_str == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_params, lr=init_lr, weight_decay=0.0001)
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_params, lr=init_lr, momentum=0.9, weight_decay=0.0001)
    else:
        assert False, 'unknown optimizer %s' % optimizer_str

    return optimizer

def keras_fit(model,train_loader=None,val_loader=None,config=None):
    """
    target to multiple output model
    remove args (depracated)
    """
    # support for pure model without config
    if config is None:
        config=model.config
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
#    if hasattr(model, 'backbone'):
#        if hasattr(model.backbone, 'model'):
#            model.backbone.model.to(device)
    
    optimizer = get_optimizer(model, config)
    init_lr = config.model.learning_rate
    
    loss_fn_dict=get_loss_fn_dict(config)
    metric_fn_dict={}
    # output for main output
    running_metrics = runningScore(config.model.class_number)
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(config.args.log_dir, model.name,
                           config.dataset.name, config.args.note, time_str)
#    checkpoint_path = os.path.join(
#        log_dir, "{}_{}_best_model.pkl".format(model.name, config.dataset.name))
    writer = None
    best_iou = 0.0
    loaders = [train_loader, val_loader]
    loader_names = ['train', 'val']
    
    # support for multiple gpu, model will be changed, model.name will not exist
    if device.type == 'cuda':
        gpu_num = torch.cuda.device_count()
        if gpu_num > 1:
            device_ids = [i for i in range(gpu_num)]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # eval module
    if train_loader is None:
        config.args.n_epoch = 1
        
    
    for epoch in trange(config.args.n_epoch,desc='epoches',leave=False):
        for loader, loader_name in zip(loaders, loader_names):
            if loader is None:
                continue
            
            # summary only 10 image
            if epoch % (1+config.args.n_epoch//10) == 0:
                summary_all = True
            else:
                summary_all = False
                    
            if loader_name == 'val':
                # summary metrics every 10 epoch
                if summary_all or epoch % 10 == 0:
                    val_log = True
                else:
                    val_log = False

                if not val_log:
                    continue

                model.eval()
            else:
                model.train()
                
            losses_dict = {}
            running_metrics.reset()
            for k,v in metric_fn_dict.items():
                metric_fn_dict[k].reset()
            for i, (datas) in enumerate(tqdm(loader,desc='steps',leave=False)):
                # work only for sgd
                poly_lr_scheduler(optimizer,
                                  init_lr=init_lr,
                                  iter=epoch*len(loader)+i,
                                  max_iter=config.args.n_epoch*len(loader))
                
                # support for w/o edge
                if len(datas) == 2:
                    images,labels=datas
                    images = torch.autograd.Variable(images.to(device).float())
                    labels = torch.autograd.Variable(labels.to(device).long())
                    targets_dict={'seg':labels,'img':images}
                elif len(datas) == 3:
                    images,labels,edges=datas
                    images = torch.autograd.Variable(images.to(device).float())
                    labels = torch.autograd.Variable(labels.to(device).long())
                    edges = torch.autograd.Variable(edges.to(device).long())
                    targets_dict={'seg':labels,'edge':edges,'img':images}
                else:
                    assert False,'unexcepted loader output size %d'%len(datas)
                
                if loader_name == 'train':
                    optimizer.zero_grad()
                
                outputs=model.forward(images)
                if isinstance(outputs,dict):
                    outputs_dict=outputs
                elif isinstance(outputs,(list,tuple)):
                    assert len(outputs)==2,'unexpected outputs length %d'%len(outputs)
                    if len(datas)==3:
                        outputs_dict={'seg':outputs[0],'edge':outputs[1]}
                    elif len(datas)==2:
                        outputs_dict={'seg':outputs[0],'aux':outputs[1]}
                    else:
                        assert False,'unexcepted loader output size %d'%len(datas)
                elif isinstance(outputs,torch.Tensor):
                    outputs_dict={'seg':outputs}
                else:
                    assert False,'unexcepted outputs type %s'%type(outputs)
                
                # return reg loss and predict loss
                loss_weight_dict=get_loss_weight(step=i+epoch*len(loader),max_step=config.args.n_epoch*len(loader),config=config)
                loss_dict=get_loss(outputs_dict,targets_dict,loss_fn_dict,config,model,loss_weight_dict=loss_weight_dict,prefix_note=loader_name)
                for k,v in loss_dict.items():
                    if k in losses_dict.keys():
                        losses_dict[k].append(v.data.cpu().numpy())
                    else:
                        losses_dict[k]=[v.data.cpu().numpy()]
                
                if loader_name=='train':
                    loss_dict['%s/total_loss'%loader_name].backward()
                    optimizer.step()
                
                # other metric for edge and aux, not run for each epoch to save time
                running_metrics,metric_fn_dict=update_metric(outputs_dict,
                                                             targets_dict,
                                                             running_metrics,
                                                             metric_fn_dict,
                                                             config,
                                                             summary_all=summary_all,
                                                             prefix_note=loader_name)
                # note, the dict format seg_xxx, aux_xxx for seg
                # edge_xxx for edge
            
            metric_dict,class_iou_dict=get_metric(running_metrics,metric_fn_dict,summary_all=summary_all,prefix_note=loader_name)
            if loader_name=='val':
                val_iou=metric_dict['val/iou']
                if val_iou >= best_iou:
                    best_iou = val_iou
            image_dict=get_image_dict(outputs_dict,targets_dict,config,summary_all=summary_all,prefix_note=loader_name)
            lr_dict=get_lr_dict(optimizer,prefix_note=loader_name)
            if writer is None:
                writer=init_writer(config=config,log_dir=log_dir)
            
            if loader_name=='train':
                weight_dict={}
                for k,v in loss_weight_dict.items():
                    weight_dict['%s/weight_%s'%(loader_name,k)]=v
            else:
                weight_dict={}
            write_summary(writer=writer,
                          losses_dict=losses_dict,
                          metric_dict=metric_dict,
                          class_iou_dict=class_iou_dict,
                          lr_dict=lr_dict,
                          image_dict=image_dict,
                          weight_dict=weight_dict,
                          epoch=epoch)
    writer.close()
    return best_iou
            
def get_loss_fn_dict(config):
    """
    remove support for model loss_fn
    """
#    if hasattr(model, 'loss_fn'):
#        loss_fn = model.loss_fn
#    else:
#        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    ignore_index=config.dataset.ignore_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn_dict={}
    loss_fn_dict['seg'] = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    if config.dataset.with_edge:
        if hasattr(config.dataset, 'edge_class_num'):
            edge_class_num = config.dataset.edge_class_num
        else:
            edge_class_num = 2
    
        edge_bg_weight = config.model.edge_bg_weight
        if edge_class_num == 2:
            # edge fg=0, bg=1
            edge_weight_list = [1.0, edge_bg_weight]
        else:
            # edge fg=0, bg=1,2,...,edge_class_num-1
            edge_weight_list = [edge_bg_weight for i in range(edge_class_num)]
            edge_weight_list[0] = 1.0
    
        edge_loss_weight = torch.tensor(
            data=edge_weight_list, dtype=torch.float32).to(device)
        loss_fn_dict['edge'] = torch.nn.CrossEntropyLoss(
            weight=edge_loss_weight, ignore_index=ignore_index)
    
    return loss_fn_dict

def get_metric_fn_dict(config):
    metric_fn_dict={}
    metric_fn_dict['seg']=get_scores
    if config.dataset.with_edge:
        metric_fn_dict['edge']=get_scores
    
    return metric_fn_dict

def get_loss_weight(step,max_step,config=None):
    loss_weight_dict={}
    loss_power=0.9
    edge_power=aux_power=loss_power
    edge_base_weight=aux_base_weight=1.0
    if config is not None:
        if hasattr(config.model,'edge_power'):
            edge_power=config.model.edge_power
        if hasattr(config.model,'aux_power'):
            aux_power=config.model.aux_power
        if hasattr(config.model,'edge_base_weight'):
            edge_base_weight=config.model.edge_base_weight
        if hasattr(config.model,'aux_base_weight'):
            aux_base_weight=config.model.aux_base_weight
            
        config.model.poly_loss_weight=True
        if hasattr(config.model,'poly_loss_weight'):
            if config.model.poly_loss_weight:
                loss_weight_dict['seg']=1.0
                # from small to big
                loss_weight_dict['edge']=edge_base_weight*(step/(1.0+max_step))**edge_power
                # from big to small
                loss_weight_dict['aux']=aux_base_weight*(1-step/(1.0+max_step))**aux_power
                
                return loss_weight_dict
            
    loss_weight_dict['seg']=1.0
    loss_weight_dict['edge']=edge_base_weight
    loss_weight_dict['aux']=aux_base_weight
    
    return loss_weight_dict

def get_loss(outputs_dict,targets_dict,loss_fn_dict,config,model,loss_weight_dict=None,prefix_note='train'):
    """
    return loss for backward
    return reg loss for summary
    input tensor, output tensor
    """
    if loss_weight_dict is None:
        loss_weight_dict={}
        loss_weight_dict['seg']=1.0
        loss_weight_dict['edge']=1.0
        loss_weight_dict['aux']=1.0
    
    loss_dict={}
    for key,value in outputs_dict.items():
        if key.startswith(('seg','aux')):
            loss=loss_fn_dict['seg'](input=value,target=targets_dict['seg'])*loss_weight_dict[key[0:3]]
        elif key.startswith('edge'):
            loss=loss_fn_dict['edge'](input=value,target=targets_dict['edge'])*loss_weight_dict['edge']
        else:
            assert False,'unexcepted key %s in outputs_dict'%key
        
        # split main loss and branch loss in summary
        if key in ['seg','edge']:
            loss_dict[prefix_note+'_loss/'+key]=loss
            # for history code
            if key=='seg':
                loss_dict['%s/%s'%(prefix_note,'loss')]=loss
            else:
                loss_dict['%s/%s'%(prefix_note,'edge_loss')]=loss
        else:
            loss_dict['%s_branch_loss/%s'%(prefix_note,key)]=loss
    
        if 'total_loss' not in loss_dict.keys():
            loss_dict['%s/total_loss'%prefix_note]=loss
        else:
            loss_dict['%s/total_loss'%prefix_note]+=loss
            
    if config.model.use_reg:
#        l1_reg = config.model.l1_reg
        l2_reg = config.model.l2_reg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        l2_loss = torch.autograd.Variable(
                            torch.FloatTensor(1), requires_grad=True).to(device)
#        l1_loss = torch.autograd.Variable(
#            torch.FloatTensor(1), requires_grad=True).to(device)
        l2_loss = 0
#        l1_loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad==False:
                continue
            if 'bias' not in name:
                l2_loss = l2_loss + torch.norm(param, 2)
#                l1_loss = l1_loss + torch.norm(param, 1)
#                l2_loss = l2_loss + torch.sum(param**2)/2
        # for history code
#        loss_dict['%s/l1_loss'%prefix_note]=l1_loss*l1_reg
        loss_dict['%s/l2_loss'%prefix_note]=l2_loss*l2_reg
#        loss_dict['%s/total_loss'%prefix_note]+=l1_loss*l1_reg+l2_loss*l2_reg
        loss_dict['%s/total_loss'%prefix_note]+=l2_loss*l2_reg
    return loss_dict
    
def update_metric(outputs_dict,targets_dict,running_metrics,metric_fn_dict,config,summary_all=False,prefix_note='train'):
    """
    update running_metrics and metric_fn_dict summary
    running_metrics: update seg miou and acc for summary
    metric_fn_dict: update aux,edge miou and acc for summary
    """   
    if summary_all:
        # convert tensor to numpy, 
        np_outputs_dict={}
        for key,value in outputs_dict.items():
            np_outputs_dict[key]=torch.argmax(value,dim=1).data.cpu().numpy()
            if key not in metric_fn_dict.keys():
                metric_fn_dict[key]=runningScore(config.model.class_number)
            
        np_targets_dict={}
        for key,value in targets_dict.items():
            np_targets_dict[key]=value.data.cpu().numpy()
        
        # main metric, run for each epoch
        running_metrics.update(np_targets_dict['seg'], np_outputs_dict['seg'])
        for key,value in np_outputs_dict.items():
            if key.startswith(('seg','aux')):
                metric_fn_dict[key].update(np_targets_dict['seg'],value)
            elif key.startswith('edge'):
                metric_fn_dict[key].update(np_targets_dict['edge'],value)
            else:
                assert False,'unexcepted key %s in outputs_dict'%key
    else:
        # main metric, run for each epoch
        running_metrics.update(targets_dict['seg'].data.cpu().numpy(), torch.argmax(outputs_dict['seg'],dim=1).data.cpu().numpy())
    return running_metrics,metric_fn_dict

def get_metric(running_metrics,metric_fn_dict,summary_all=False,prefix_note='train'):
    metric_dict={}
    score, class_iou = running_metrics.get_scores()
    metric_dict['%s/acc'%prefix_note]=score['Overall Acc: \t']
    metric_dict['%s/iou'%prefix_note]=score['Mean IoU : \t']
    
    class_iou_dict={}
    for k,v in class_iou.items():
        class_iou_dict['%s_class_iou/%02d'%(prefix_note,k)]=v

    if summary_all:
        for key,v in metric_fn_dict.items():
            score, class_iou = metric_fn_dict[key].get_scores()
            if key in ['seg','edge']:
                metric_dict[prefix_note+'_metric/'+key+'_acc']=score['Overall Acc: \t']
                metric_dict[prefix_note+'_metric/'+key+'_iou']=score['Mean IoU : \t']
            else:
                metric_dict[prefix_note+'_branch_metric/'+key+'_acc']=score['Overall Acc: \t']
                metric_dict[prefix_note+'_branch_metric/'+key+'_iou']=score['Mean IoU : \t']
            
    return metric_dict,class_iou_dict
    
def get_image_dict(outputs_dict,targets_dict,config,summary_all=False,prefix_note='train'):
    image_dict={}
    if summary_all:
        gpu_num = torch.cuda.device_count()
        # for parallel, the true batch size for image will be config.args.batch_size//gpu_num
        idx = np.random.randint(config.args.batch_size//gpu_num)
        # convert tensor to numpy, 
        np_outputs_dict={}
        for key,value in outputs_dict.items():
            np_outputs_dict[key]=value[idx].data.cpu().numpy().argmax(1)
            
        np_targets_dict={}
        for key,value in targets_dict.items():
            np_targets_dict[key]=value[idx].data.cpu().numpy()
        
        seg_pixel_scale = 255//config.model.class_number
        edge_pixel_scale = 255//config.dataset.edge_class_num
        
        for k,v in np_outputs_dict.items():
            if k.startswith(('seg','aux')):
                image_dict['%s/predict_%s'%(prefix_note,k)]=(v*seg_pixel_scale).astype(np.uint8)
            elif k.startswith('edge'):
                image_dict['%s/predict_%s'%(prefix_note,k)]=(v*edge_pixel_scale).astype(np.uint8)
            else:
                assert False,'unexcepted key %s in outputs_dict'%key
                
        normalizations = image_normalizations(config.dataset.norm_ways)
        for k,v in np_targets_dict.items():
            if k=='img':
                org_img=v.transpose((1,2,0))
                if normalizations is not None:
                    org_img = normalizations.backward(org_img)
                image_dict['%s/%s'%(prefix_note,k)]=org_img.astype(np.uint8)
            elif k=='seg':
                image_dict['%s/%s'%(prefix_note,k)]=(v*seg_pixel_scale).astype(np.uint8)
            elif k=='edge':
                image_dict['%s/%s'%(prefix_note,k)]=(v*edge_pixel_scale).astype(np.uint8)
            else:
                assert False,'unexcepted key %s in outputs_dict'%key
    return image_dict
            
def get_lr_dict(optimizer,prefix_note='train'):
    lr_dict={}
    for idx,params in enumerate(optimizer.param_groups):
        lr_dict['%s_lr/%d'%(prefix_note,idx)]=params['lr']

    return lr_dict

def init_writer(config,log_dir):
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
    
    return writer

def write_summary(writer,losses_dict,metric_dict,class_iou_dict,lr_dict,image_dict,weight_dict,epoch):
    # losses_dict value is numpy
    for k,v in losses_dict.items():
        writer.add_scalar(k,np.mean(v),epoch)
    
    # summary metrics
    for k,v in metric_dict.items():
        writer.add_scalar(k,v,epoch)
    
    # summary class iou
    for k,v in class_iou_dict.items():
        writer.add_scalar(k, v, epoch)
    
    # summary for class weight
    for k,v in weight_dict.items():
        writer.add_scalar(k,v,epoch)
    
    # summary learning rate
    for k,v in lr_dict.items():
        writer.add_scalar(k,v,epoch)
    
    # summary image
    for k,v in image_dict.items():
        add_image(summary_writer=writer,name=k,image=v,step=epoch)