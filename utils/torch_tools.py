# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import json
import os
from tensorboardX import SummaryWriter
from utils.metrics import runningScore
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


def do_train_or_val(model, args, train_loader=None, val_loader=None, config=None):
    if config is None:
        config = model.config

    ignore_index = config.dataset.ignore_index
    class_number = config.model.class_number
    dataset_name = config.dataset.name

    if hasattr(model, 'do_train_or_val'):
#        print('warning: use do_train_or_val in model'+'*'*30)
        model.do_train_or_val(args, train_loader, val_loader)
        return 0

    # use gpu memory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'model'):
            model.backbone.model.to(device)

    optimizer = get_optimizer(model, config)

    use_reg = config.model.use_reg if hasattr(
        config.model, 'use_reg') else False
#    if use_reg:
#        print('use l1 and l2 reg loss'+'*'*30)

    if hasattr(model, 'loss_fn'):
#        print('use loss function in model'+'*'*30)
        loss_fn = model.loss_fn
    else:
#        print('use default loss funtion with ignore_index=%d' %
#              ignore_index, '*'*30)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    # TODO metrics supprot ignore_index
    running_metrics = runningScore(class_number)

    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(args.log_dir, model.name,
                           dataset_name, args.note, time_str)
    checkpoint_path = os.path.join(
        log_dir, "{}_{}_best_model.pkl".format(model.name, dataset_name))
    writer = None
    best_iou = 0.0

    power = 0.9

    init_lr = config.model.learning_rate if hasattr(
        config.model, 'learning_rate') else 0.0001
    loaders = [train_loader, val_loader]
    loader_names = ['train', 'val']

    if device.type == 'cuda':
        gpu_num = torch.cuda.device_count()
        if gpu_num > 1:
            device_ids = [i for i in range(gpu_num)]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
#            print('use multi gpu', device_ids, '*'*30)
            time.sleep(3)
        else:
#            print('use single gpu', '*'*30)
            pass
    else:
#        print('use cpu only', '*'*30)
        pass

    if train_loader is None:
        args.n_epoch = 1

    normalizations = image_normalizations(config.dataset.norm_ways)
    for epoch in trange(args.n_epoch,desc='epoches',leave=False):
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

#            print(loader_name+'.'*50)
            n_step = len(loader)
            losses = []
            l1_reg = config.model.l1_reg
            l2_reg = config.model.l2_reg
            running_metrics.reset()
            for i, (images, labels) in enumerate(trange(loader,desc='steps',leave=False)):
                # work only for sgd
                poly_lr_scheduler(optimizer,
                                  init_lr=init_lr,
                                  iter=epoch*len(loader)+i,
                                  max_iter=args.n_epoch*len(loader),
                                  power=power)

                images = torch.autograd.Variable(images.to(device).float())
                labels = torch.autograd.Variable(labels.to(device).long())

                if loader_name == 'train':
                    optimizer.zero_grad()
                seg_output = model.forward(images)
                loss = loss_fn(input=seg_output, target=labels)

                if loader_name == 'train':
                    if use_reg:
                        l2_loss = torch.autograd.Variable(
                            torch.FloatTensor(1), requires_grad=True).to(device)
                        l1_loss = torch.autograd.Variable(
                            torch.FloatTensor(1), requires_grad=True).to(device)
                        l2_loss = 0
                        l1_loss = 0
                        for name, param in model.named_parameters():
                            if param.requires_grad==False:
                                continue
                            if 'bias' not in name:
#                                l2_loss = l2_loss + torch.norm(param, 2)
                                l1_loss = l1_loss + torch.norm(param, 1)
                                l2_loss = l2_loss + torch.sum(param**2)/2
                        loss = loss + l2_loss*l2_reg + l1_loss*l1_reg
                    loss.backward()
                    optimizer.step()

                losses.append(loss.data.cpu().numpy())
                predicts = seg_output.data.cpu().numpy().argmax(1)
                trues = labels.data.cpu().numpy()
                running_metrics.update(trues, predicts)
                score, class_iou = running_metrics.get_scores()

#                if (i+1) % 5 == 0:
#                    print("%s, Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" %
#                          (loader_name, epoch+1, args.n_epoch, i, n_step, loss.data))
#                    for k, v in score.items():
#                        print(k, v)

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

            writer.add_scalar('%s/loss' % loader_name,
                              np.mean(losses), epoch)
            if use_reg:
                writer.add_scalar('%s/l1_loss' % loader_name,
                                  l1_loss*l1_reg, epoch)
                writer.add_scalar('%s/l2_loss' % loader_name,
                                  l2_loss*l2_reg, epoch)
            writer.add_scalar('%s/acc' % loader_name,
                              score['Overall Acc: \t'], epoch)
            writer.add_scalar('%s/iou' % loader_name,
                              score['Mean IoU : \t'], epoch)
            writer.add_scalar('%s/lr' % loader_name,
                              optimizer.param_groups[0]['lr'], epoch)
            for idx,params in enumerate(optimizer.param_groups):
                if idx>0:
                    writer.add_scalar('%s/lr_%d' % (loader_name,idx),
                              optimizer.param_groups[idx]['lr'], epoch)
                        
            if loader_name == 'val':
                if score['Mean IoU : \t'] >= best_iou:
                    best_iou = score['Mean IoU : \t']
#                    state = {'epoch': epoch+1,
#                             'miou': best_iou,
#                             'model_state': model.state_dict(),
#                             'optimizer_state': optimizer.state_dict(), }
#
#                    torch.save(state, checkpoint_path)

                if val_image:
#                    print('write image to tensorboard'+'.'*50)
                    pixel_scale = 255//config.model.class_number
                    idx = np.random.choice(predicts.shape[0])

                    origin_img = images.data.cpu().numpy()
                    origin_img = origin_img.transpose((0, 2, 3, 1))
                    if normalizations is not None:
                        origin_img = normalizations.backward(origin_img)

                    writer.add_image(
                        'val/images', origin_img[idx, :, :, :].astype(np.uint8), epoch)
                    writer.add_image(
                        'val/predicts', torch.from_numpy((predicts[idx, :, :]*pixel_scale).astype(np.uint8)), epoch)
                    writer.add_image(
                        'val/trues', torch.from_numpy((trues[idx, :, :]*pixel_scale).astype(np.uint8)), epoch)
                    diff_img = (predicts[idx, :, :] ==
                                trues[idx, :, :]).astype(np.uint8)
                    writer.add_image('val/difference',
                                     torch.from_numpy(diff_img), epoch)

    writer.close()
    return best_iou
