# -*- coding: utf-8 -*-
import numpy as np
import torch
import time
import json
import os
# from packaging import version
# if version.parse(torch.__version__) < version.parse('1.1.0'):
#     from tensorboardX import SummaryWriter
# else:
#     from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
    
from .metrics import runningScore
from .disc_tools import save_model_if_necessary, get_newest_file
from .center_loss2d import CenterLoss
from ..dataset.dataset_generalize import image_normalizations, dataset_generalize
from .augmentor import Augmentations
from .losses import get_loss_fn
from .poly_plateau import poly_rop,poly_lr_scheduler
import torch.utils.data as TD
from torch.optim.lr_scheduler import CosineAnnealingLR as cos_lr
from torch.optim.lr_scheduler import ReduceLROnPlateau as rop
from tqdm import tqdm, trange
from easydict import EasyDict as edict
import glob

def get_merged_dataset(config,split,normalizations=None,augmentations=None):
    """
    load multiple dataset from config.dataset_names, which is a list/tuple
    config project on config.dataset, which is a string
    """
    if not hasattr(config,'dataset_names'):
        config.dataset_names=[config.dataset_name]
        
    assert isinstance(config.dataset_names,(list,tuple))
    
    dconfig=edict(config.copy())
    datasets=[]
    for dataset_name in config.dataset_names:
        dconfig.dataset_name=dataset_name
        dataset = dataset_generalize(dconfig,
                                     split=split,
                                     augmentations=augmentations if split=='train' else None,
                                     normalizations=normalizations)
        datasets.append(dataset)
    
    if len(datasets)==1:
        return datasets[0]
    else:
        merged_dataset=TD.ConcatDataset(datasets)
        return merged_dataset
        
def get_loaders(config):
    if config.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.norm_ways)

    if config.augmentation:
        augmentations = Augmentations(config)
    else:
        augmentations = None
    
    loaders={}
    for split in ['train','val']:
        dataset = get_merged_dataset(config,split=split,
                                    augmentations=augmentations if split=='train' else None,
                                    normalizations=normalizations)
        
        loaders[split]=TD.DataLoader(
                        dataset=dataset,
                        batch_size=config.batch_size,
                        shuffle=True if split=='train' else False,
                        drop_last=True if split=='train' else False,
                        num_workers=2*config.batch_size)
        
    return loaders['train'], loaders['val']


def add_image(summary_writer, name, image, step):
    """
    add numpy/tensor image with shape [2d,3d,4d] to summary
    support [h,w] [h,w,1], [1,h,w], [h,w,3], [3,h,w], [b,h,w,c], [b,c,h,w] shape
    combie with numpy,tensor

    note: data should in right range such as [0,1], [0,255] and right dtype
    dtype: np.uint8 for [0,255]
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            summary_writer.add_image(name, torch.from_numpy(image), step)
        elif image.ndim == 3:
            a, b, c = image.shape
            if min(a, c) == 1:
                if a == 1:
                    summary_writer.add_image(
                        name, torch.from_numpy(image[0, :, :]), step)
                else:
                    summary_writer.add_image(
                        name, torch.from_numpy(image[:, :, 0]), step)
            else:
                if a == 3:
                    summary_writer.add_image(
                        name, torch.from_numpy(image), step)
                elif c == 3:
                    summary_writer.add_image(name, image, step)
                else:
                    assert False, 'unexcepted image shape %s' % str(
                        image.shape)
        elif image.ndim == 4:
            add_image(summary_writer, name, image[0, :, :, :], step)
        else:
            assert False, 'unexcepted image shape %s' % str(image.shape)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 2:
            summary_writer.add_imge(name, image, step)
        elif image.dim() == 3:
            a, b, c = image.shape
            if min(a, c) == 1:
                if a == 1:
                    summary_writer.add_image(name, image[0, :, :], step)
                else:
                    summary_writer.add_image(name, image[:, :, 0], step)
            else:
                if a == 3:
                    summary_writer.add_image(name, image, step)
                elif c == 3:
                    summary_writer.add_image(
                        name, image.data.cpu().numpy(), step)
                else:
                    assert False, 'unexcepted image shape %s' % str(
                        image.shape)
        elif image.dim() == 4:
            add_image(summary_writer, name, image[0, :, :, :], step)
        else:
            assert False, 'unexcepted image shape %s' % str(image.shape)
    else:
        assert False, 'unknown type %s' % type(image)


def freeze_layer(layer):
    """
    freeze layer weights
    """
    for param in layer.parameters():
        param.requires_grad = False


def is_main_process(config):
    return not config.mp_dist or (config.mp_dist and config.rank % config.ngpus_per_node == 0)

def train_val(model, optimizer, scheduler, loss_fn_dict,
              metric_fn_dict, running_metrics,
              loader, config, epoch, summary_all, loader_name,
              summary_metric=True,
              center_loss_model=None,
              center_optimizer=None):
    if loader_name == 'train':
        model.train()
    else:
        model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    losses_dict = {}
    running_metrics.reset()
    for k, v in metric_fn_dict.items():
        metric_fn_dict[k].reset()

    assert config.accumulate>=1
    total_loss=0
    if is_main_process(config):
        tqdm_step = tqdm(loader, desc='steps', leave=False)
    else:
        tqdm_step = loader

    for i, (datas) in enumerate(tqdm_step):
        if loader_name == 'train' and hasattr(config,'scheduler') and config.scheduler in ['poly']:
            # init max_iter
            scheduler.max_iter=config.n_epoch*len(loader)
            # scheduler learning rate
            scheduler.step()

        # support for w/o edge
        if len(datas) == 2:
            images, labels = datas
            images = torch.autograd.Variable(images.to(device).float())
            labels = torch.autograd.Variable(labels.to(device).long())
            targets_dict = {'seg': labels, 'img': images}
        elif len(datas) == 3:
            images, labels, edges = datas
            images = torch.autograd.Variable(images.to(device).float())
            labels = torch.autograd.Variable(labels.to(device).long())
            edges = torch.autograd.Variable(edges.to(device).long())
            targets_dict = {'seg': labels,
                            'edge': edges, 'img': images}
        else:
            assert False, 'unexcepted loader output size %d' % len(datas)

        if config.test=='dist':
            images=images.cuda(config.gpu,non_blocking=True)
            for key,value in targets_dict.items():
                targets_dict[key]=value.cuda(config.gpu,non_blocking=True)

        if config.net_name in ['GlobalLocalNet']:
            h,w=config.origin_input_shape
            targets_dict['seg']=targets_dict['seg'][:,h//4:(h*3)//4,w//4:(w*3)//4]

        outputs = model.forward(images)

        if isinstance(outputs, dict):
            outputs_dict = outputs
        elif isinstance(outputs, (list, tuple)):
            assert len(outputs) == 2, 'unexpected outputs length %d' % len(
                outputs)
            if len(datas) == 3:
                outputs_dict = {'seg': outputs[0], 'edge': outputs[1]}
            elif len(datas) == 2:
                outputs_dict = {'seg': outputs[0], 'aux': outputs[1]}
            else:
                assert False, 'unexcepted loader output size %d' % len(
                    datas)
        elif isinstance(outputs, torch.Tensor):
            outputs_dict = {'seg': outputs}
        else:
            assert False, 'unexcepted outputs type %s' % type(outputs)

        if loader_name == 'train':
            # return adaptive reg loss, edge loss, aux loss and seg loss weight (train only)
            loss_weight_dict = get_loss_weight(
                step=i+epoch*len(loader), max_step=config.n_epoch*len(loader), config=config)
        else:
            # use 1.0 by default
            loss_weight_dict = None

        # total loss = [seg_weight, edge_weight, reg_weight, aux_weight ...] * [seg_loss, edge_loss, reg_loss, aux_loss ...]
        loss_dict = get_loss(outputs_dict, targets_dict, loss_fn_dict, config,
                             model, loss_weight_dict=loss_weight_dict, prefix_note=loader_name)
        # not support test
        if config.center_loss is not None:
            center_loss=center_loss_model(model.center_feature,labels)
            loss_dict['%s/center_loss'%loader_name]=center_loss
            loss_dict['%s/total_loss' % loader_name]+=config.center_loss_weight*center_loss
        # record loss for summary
        for k, v in loss_dict.items():
            if k in losses_dict.keys():
                losses_dict[k].append(v.data.cpu().numpy())
            else:
                losses_dict[k] = [v.data.cpu().numpy()]

        if loader_name == 'train':
            total_loss=loss_dict['%s/total_loss' % loader_name]/config.accumulate
            total_loss.backward()
            if (i+1) % config.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                if config.center_loss is not None:
                    for param in center_loss_model.parameters():
                        param.grad.data *= (1. / config.center_loss_weight)
                    center_optimizer.step()
                    center_optimizer.zero_grad()

        if summary_metric:
            # summary_all other metric for edge and aux, not run for each epoch to save time
            running_metrics, metric_fn_dict = update_metric(outputs_dict,
                                                            targets_dict,
                                                            running_metrics,
                                                            metric_fn_dict,
                                                            config,
                                                            summary_all=summary_all,
                                                            prefix_note=loader_name)
    return outputs_dict, targets_dict, running_metrics, metric_fn_dict, losses_dict, loss_weight_dict

def get_optimizer(model, config):
    init_lr = config.learning_rate if hasattr(
        config, 'learning_rate') else 0.0001
    optimizer_str = config.optimizer if hasattr(
        config, 'optimizer') else 'adam'
    lr_weight_decay = config.lr_weight_decay if hasattr(
        config, 'lr_weight_decay') else 0.0001
    lr_momentum = config.lr_momentum if hasattr(
        config, 'lr_momentum') else 0.9
    nesterov = config.sgd_nesterov if hasattr(config,'sgd_nesterov') else False

    if hasattr(model, 'optimizer_params'):
        optimizer_params = model.optimizer_params
        # clear the params with learning rate = 0
        # optimizer_params = [p for p in model.optimizer_params if 'lr_mult' not in p.keys() or p['lr_mult']>0]
    else:
        optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]

    # init optimizer learning rate
    for i, p in enumerate(optimizer_params):
        lr_mult = p['lr_mult'] if 'lr_mult' in p.keys() else 1.0
        optimizer_params[i]['initial_lr'] = optimizer_params[i]['lr'] = init_lr*lr_mult

    if optimizer_str == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_params, lr=init_lr, weight_decay=lr_weight_decay, amsgrad=False)
    elif optimizer_str == 'amsgrad':
        optimizer = torch.optim.Adam(
            optimizer_params, lr=init_lr, weight_decay=lr_weight_decay, amsgrad=True)
    elif optimizer_str == 'adamax':
        optimizer = torch.optim.Adamax(
            optimizer_params, lr=init_lr, weight_decay=lr_weight_decay)
    elif optimizer_str == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_params, lr=init_lr, momentum=lr_momentum,
            weight_decay=lr_weight_decay, nesterov=nesterov)
    else:
        assert False, 'unknown optimizer %s' % optimizer_str

    return optimizer


def get_scheduler(optimizer, config):
    scheduler = config.scheduler if hasattr(
        config, 'scheduler') else None
    
    if hasattr(config,'rop_patience'):
        patience=config.rop_patience
    else:
        patience=10
        
    if hasattr(config,'rop_factor'):
        factor=config.rop_factor
    else:
        factor=0.1
        
    if hasattr(config,'rop_cooldown'):
        cooldown=config.rop_cooldown
    else:
        cooldown=0
        
    if hasattr(config,'poly_max_iter'):
        poly_max_iter=config.poly_max_iter
    else:
        poly_max_iter=50
        
    if hasattr(config,'poly_power'):
        poly_power=config.poly_power
    else:
        poly_power=0.9
        
    if hasattr(config,'cos_T_max'):
        T_max=config.cos_T_max
    else:
        T_max=50
    
    if scheduler == 'poly':
        # need reset max_iter on training loop
        scheduler = poly_lr_scheduler(optimizer=optimizer,
                                      max_iter=0,
                                      power=poly_power)
    elif scheduler == 'rop':
        # 'max' for acc and miou, 'min' for loss
        scheduler = rop(optimizer=optimizer, 
                        mode='min',
                        threshold=1e-4,
                        patience=patience, 
                        verbose=False,
                        factor=factor,
                        cooldown=cooldown)
    elif scheduler in ['poly_rop','pop']:
        # 'max' for acc and miou, 'min' for loss
        scheduler = poly_rop(poly_max_iter=poly_max_iter,
                             poly_power=poly_power,
                             optimizer = optimizer,
                             mode= 'min',
                             threshold=1e-4,
                             patience=patience, 
                             verbose=False,
                             cooldown=cooldown)
    elif scheduler in ['cos','cos_lr']:
        scheduler = cos_lr(optimizer=optimizer,
                           T_max=T_max)
    else:
        assert scheduler is None

    return scheduler


def get_ckpt_path(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        log_dir = checkpoint_path
        ckpt_files = glob.glob(os.path.join(
            log_dir, '**', 'model-best-*.pkl'), recursive=True)

        # use best model first, then use the last model, because the last model will be the newest one if exist.
        if len(ckpt_files) == 0:
            ckpt_files = glob.glob(os.path.join(
                log_dir, '**', '*.pkl'), recursive=True)

        assert len(
            ckpt_files) > 0, 'no weight file found under %s, \n please specify checkpoint path' % log_dir
        checkpoint_path = get_newest_file(ckpt_files)
        print('no checkpoint file given, auto find %s' % checkpoint_path)
        return checkpoint_path
    else:
        if checkpoint_path.endswith("config.txt"):
            print('find checkpoint file by config.txt path')
            return get_ckpt_path(os.path.dirname(checkpoint_path))
        else:
            return checkpoint_path

def load_ckpt(model,ckpt_path):
    state_dict = torch.load(ckpt_path)
    if 'model_state' in state_dict.keys():
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    return model

def keras_fit(model, train_loader=None, val_loader=None, config=None):
    """
    target to multiple output model
    remove args (depracated)
    """
    # support for pure model without config
    if config is None:
        config = model.config

    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if config.checkpoint_path is not None:
        ckpt_path = get_ckpt_path(config.checkpoint_path)
        print('load checkpoint file from', ckpt_path)
        state_dict = torch.load(ckpt_path)
        if 'model_state' in state_dict.keys():
            model.load_state_dict(state_dict['model_state'])
        else:
            model.load_state_dict(state_dict)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    if config.center_loss is not None:
        center_loss_model=CenterLoss(model.center_channels,model.class_number,
                                     ignore_index=config.ignore_index,
                                     loss_fn=config.center_loss).to(device)
        center_optimizer=torch.optim.SGD(center_loss_model.parameters(), lr=0.5)
    else:
        center_loss_model=None
        center_optimizer=None

    loss_fn_dict = get_loss_fn_dict(config)
    # for different output, generate the metric_fn_dict automaticly.
    metric_fn_dict = {}
    # output for main output
    running_metrics = runningScore(config.class_number)

    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir = os.path.join(config.log_dir, config.net_name,
                           config.dataset_name, config.note, time_str)
#    checkpoint_path = os.path.join(
#        log_dir, "{}_{}_best_model.pkl".format(model.name, config.name))
    writer = None
    best_iou = 0.0
    # create loader from config
    if train_loader is None and val_loader is None:
        train_loader, val_loader = get_loaders(config)

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
        config.n_epoch = 1

    summary_all_step=max(1,config.n_epoch//10)
    # 1<= summary_metric_step <=10
    summary_metric_step=max(min(10*config.accumulate,config.n_epoch//10),1)

    tqdm_epoch = trange(config.n_epoch, desc='{} epoches'.format(config.note), leave=True)
    for epoch in tqdm_epoch:
        tqdm_epoch.set_postfix(best_iou=best_iou)
        for loader, loader_name in zip(loaders, loader_names):
            if loader is None:
                continue

            # summary all only 10 times
            if epoch % summary_all_step == 0:
                summary_all = True
                summary_metric = True
            else:
                summary_all = False
                if epoch % summary_metric_step == 0 or epoch == config.n_epoch-1:
                    summary_metric=True
                else:
                    summary_metric=False

            # summary_all=True ==> summary_metric=True
            if loader_name == 'val':
                # val at summary, and val for plateau scheduler
                if (not summary_metric) and (scheduler is None):
                    continue

                with torch.no_grad():
                    outputs_dict, targets_dict, \
                    running_metrics, metric_fn_dict, \
                    losses_dict, loss_weight_dict = train_val(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn_dict=loss_fn_dict,
                        metric_fn_dict=metric_fn_dict,
                        running_metrics=running_metrics,
                        loader=loader,
                        config=config,
                        epoch=epoch,
                        summary_all=summary_all,
                        summary_metric=summary_metric,
                        loader_name=loader_name,
                        center_loss_model=center_loss_model,
                        center_optimizer=center_optimizer)

                # use rop/poly_rop to schedule learning rate
                if isinstance(scheduler,(poly_rop,rop)):
                    total_loss=sum(losses_dict['%s/total_loss' % loader_name])
                    scheduler.step(total_loss)
            else:
                outputs_dict, targets_dict, \
                running_metrics, metric_fn_dict, \
                losses_dict, loss_weight_dict = train_val(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn_dict=loss_fn_dict,
                    metric_fn_dict=metric_fn_dict,
                    running_metrics=running_metrics,
                    loader=loader,
                    config=config,
                    epoch=epoch,
                    summary_all=summary_all,
                    summary_metric=summary_metric,
                    loader_name=loader_name,
                    center_loss_model=center_loss_model,
                    center_optimizer=center_optimizer)

                # use cos_lr to shceduler the learning rate
                if isinstance(scheduler,cos_lr):
                    scheduler.step()

            metric_dict, class_iou_dict = get_metric(
                running_metrics, metric_fn_dict,
                summary_all=summary_all, prefix_note=loader_name, summary_metric=summary_metric)
            if loader_name == 'val' and summary_metric:
                val_iou = metric_dict['val/iou']
                tqdm.write('epoch %d,curruent val iou is %0.5f' %
                        (epoch, val_iou))
                if val_iou >= best_iou:
                    best_iou = val_iou
                    iou_save_threshold = config.iou_save_threshold

                    # save the best the model if good enough
                    if best_iou >= iou_save_threshold:
                        print('save current best model', '*'*30)
                        checkpoint_path = os.path.join(
                            log_dir, 'model-best-%d.pkl' % epoch)
                        save_model_if_necessary(model, config, checkpoint_path)

                # save the last model if the best model not good enough
                if epoch == config.n_epoch-1 and best_iou < iou_save_threshold:
                    print('save the last model', '*'*30)
                    checkpoint_path = os.path.join(
                        log_dir, 'model-last-%d.pkl' % epoch)
                    save_model_if_necessary(model, config, checkpoint_path)

            # return valid image when summary_all=True
            image_dict = get_image_dict(
                outputs_dict, targets_dict, config,
                summary_all=summary_all, prefix_note=loader_name)

            if writer is None:
                writer = init_writer(config=config, log_dir=log_dir)

            # change weight and learning rate (train only)
            if loader_name == 'train':
                weight_dict = {}
                for k, v in loss_weight_dict.items():
                    weight_dict['%s/weight_%s' % (loader_name, k)] = v

                lr_dict = get_lr_dict(optimizer, prefix_note=loader_name)
            else:
                weight_dict = {}
                lr_dict = {}

            write_summary(writer=writer,
                          losses_dict=losses_dict,
                          metric_dict=metric_dict,
                          class_iou_dict=class_iou_dict,
                          lr_dict=lr_dict,
                          image_dict=image_dict,
                          weight_dict=weight_dict,
                          epoch=epoch)
    writer.close()

    print('total epoch is %d, best iou is' % config.n_epoch, best_iou)
    return best_iou


def get_loss_fn_dict(config):
    """
    remove support for model loss_fn
    """
#    if hasattr(model, 'loss_fn'):
#        loss_fn = model.loss_fn
#    else:
#        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    ignore_index = config.ignore_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if hasattr(config, 'counts') and config.use_class_weight:
        count_sum = 1.0*np.sum(config.counts)
        weight_raw = [count_sum/count for count in config.counts]
        # make the total loss not change!!!
        weight_sum = np.sum(weight_raw)
        class_number = len(config.counts)
        seg_weight_list = [class_number*w/weight_sum for w in weight_raw]
        seg_loss_weight = torch.tensor(
            data=seg_weight_list, dtype=torch.float32).to(device)

        alpha = config.class_weight_alpha
        seg_loss_weight = seg_loss_weight+(1.0-seg_loss_weight)*alpha
        print('segmentation class weight is', '*'*30)
        for idx, w in enumerate(seg_weight_list):
            print(idx, '%0.3f' % w)
        print('total class weight sum is', np.sum(seg_weight_list))
    else:
        seg_loss_weight = None

    loss_fn_dict = {}
    loss_fn_dict['seg']=get_loss_fn(config.loss_type,
                                    class_number=config.class_number,
                                    weight=seg_loss_weight,
                                    ignore_index=ignore_index,
                                    focal_loss_alpha=config.focal_loss_alpha,
                                    focal_loss_gamma=config.focal_loss_gamma,
                                    focal_loss_grad=config.focal_loss_grad)
    # if config.focal_loss_gamma < 0:
    #     loss_fn_dict['seg'] = torch.nn.CrossEntropyLoss(
    #         ignore_index=ignore_index, weight=seg_loss_weight)
    # else:
    #     loss_fn_dict['seg'] = FocalLoss2d(alpha=config.focal_loss_alpha,
    #                                       gamma=config.focal_loss_gamma,
    #                                       weight=seg_loss_weight,
    #                                       ignore_index=ignore_index,
    #                                       with_grad=config.focal_loss_grad)

    if config.with_edge:
        if hasattr(config, 'edge_class_num'):
            edge_class_num = config.edge_class_num
        else:
            edge_class_num = 2

        edge_bg_weight = config.edge_bg_weight
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

def get_loss_weight(step, max_step, config=None):
    """
    dynamic loss weight for psp_edge and psp_aux
    """
    loss_weight_dict = {}
    loss_power = 0.9
    edge_power = aux_power = loss_power
    edge_base_weight = main_base_weight = aux_base_weight = 1.0

    if config is not None:
        if hasattr(config, 'edge_power'):
            edge_power = config.edge_power
        if hasattr(config, 'aux_power'):
            aux_power = config.aux_power
        if hasattr(config, 'edge_base_weight'):
            edge_base_weight = config.edge_base_weight
        if hasattr(config, 'aux_base_weight'):
            aux_base_weight = config.aux_base_weight
        if hasattr(config, 'main_base_weight'):
            main_base_weight = config.main_base_weight

        config.poly_loss_weight = False
        if hasattr(config, 'poly_loss_weight'):
            if config.poly_loss_weight:
                loss_weight_dict['seg'] = main_base_weight
                # from small to big
                loss_weight_dict['edge'] = edge_base_weight * \
                    (step/(1.0+max_step))**edge_power
                # from big to small
                loss_weight_dict['aux'] = aux_base_weight * \
                    (1-step/(1.0+max_step))**aux_power

                return loss_weight_dict

    loss_weight_dict['seg'] = main_base_weight
    loss_weight_dict['edge'] = edge_base_weight
    loss_weight_dict['aux'] = aux_base_weight

    return loss_weight_dict


def get_loss(outputs_dict, targets_dict, loss_fn_dict, config, model, loss_weight_dict=None, prefix_note='train'):
    """
    return loss for backward
    return reg loss for summary
    input tensor, output tensor
    """
    if loss_weight_dict is None:
        loss_weight_dict = {}
        loss_weight_dict['seg'] = 1.0
        loss_weight_dict['edge'] = 1.0
        loss_weight_dict['aux'] = 1.0

    loss_dict = {}
    for key, value in outputs_dict.items():
        if key.startswith(('seg', 'aux')):
            loss = loss_fn_dict['seg'](
                input=value, target=targets_dict['seg'])*loss_weight_dict[key[0:3]]
        elif key.startswith('edge'):
            loss = loss_fn_dict['edge'](
                input=value, target=targets_dict['edge'])*loss_weight_dict['edge']
        else:
            assert False, 'unexcepted key %s in outputs_dict' % key

        # split main loss and branch loss in summary
        if key in ['seg', 'edge']:
            loss_dict[prefix_note+'_loss/'+key] = loss
            # for history code
            if key == 'seg':
                loss_dict['%s/%s' % (prefix_note, 'loss')] = loss
            else:
                loss_dict['%s/%s' % (prefix_note, 'edge_loss')] = loss
        else:
            loss_dict['%s_branch_loss/%s' % (prefix_note, key)] = loss

        if 'total_loss' not in loss_dict.keys():
            loss_dict['%s/total_loss' % prefix_note] = loss
        else:
            loss_dict['%s/total_loss' % prefix_note] += loss

    # use weight decay instead will be better
    if config.use_reg:
        #        l1_reg = config.l1_reg
        l2_reg = config.l2_reg
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        l2_loss = torch.autograd.Variable(
            torch.FloatTensor(1), requires_grad=True).to(device)
#        l1_loss = torch.autograd.Variable(
#            torch.FloatTensor(1), requires_grad=True).to(device)
        l2_loss = 0
#        l1_loss = 0
        for name, param in model.named_parameters():
            if param.requires_grad == False:
                continue
            if 'bias' not in name:
                l2_loss = l2_loss + torch.norm(param, 2)
#                l1_loss = l1_loss + torch.norm(param, 1)
#                l2_loss = l2_loss + torch.sum(param**2)/2
        # for history code
#        loss_dict['%s/l1_loss'%prefix_note]=l1_loss*l1_reg
        loss_dict['%s/l2_loss' % prefix_note] = l2_loss*l2_reg
#        loss_dict['%s/total_loss'%prefix_note]+=l1_loss*l1_reg+l2_loss*l2_reg
        loss_dict['%s/total_loss' % prefix_note] += l2_loss*l2_reg
    return loss_dict


def update_metric(outputs_dict,
                  targets_dict,
                  running_metrics,
                  metric_fn_dict,
                  config,
                  summary_all=False,
                  prefix_note='train'):
    """
    update running_metrics and metric_fn_dict summary
    running_metrics: update seg miou and acc for summary
    metric_fn_dict: update aux,edge miou and acc for summary
    """
    if summary_all:
        # convert tensor to numpy,
        np_outputs_dict = {}
        for key, value in outputs_dict.items():
            np_outputs_dict[key] = torch.argmax(
                value, dim=1).data.cpu().numpy()
            if key not in metric_fn_dict.keys():
                if key.startswith(('seg','aux')):
                    metric_fn_dict[key] = runningScore(config.class_number)
                elif key.startswith('edge'):
                    metric_fn_dict[key] = runningScore(config.edge_class_num)
                else:
                    assert False, 'unexcepted key %s in outputs_dict' % key

        np_targets_dict = {}
        for key, value in targets_dict.items():
            np_targets_dict[key] = value.data.cpu().numpy()

        # main metric, run for each epoch
        running_metrics.update(np_targets_dict['seg'], np_outputs_dict['seg'])
        for key, value in np_outputs_dict.items():
            if key.startswith(('seg', 'aux')):
                metric_fn_dict[key].update(np_targets_dict['seg'], value)
            elif key.startswith('edge'):
                metric_fn_dict[key].update(np_targets_dict['edge'], value)
            else:
                assert False, 'unexcepted key %s in outputs_dict' % key
    else:
        # main metric, run for each epoch
        running_metrics.update(targets_dict['seg'].data.cpu().numpy(
        ), torch.argmax(outputs_dict['seg'], dim=1).data.cpu().numpy())
    return running_metrics, metric_fn_dict


def get_metric(running_metrics, metric_fn_dict, summary_all=False, prefix_note='train',summary_metric=True):
    """
    running_metrics: main metric
    metric_fn_dict: all metric
    summary_all: use metric_fn_dict or not
    """
    if not summary_metric:
        return {},{}

    metric_dict = {}
    score, class_iou = running_metrics.get_scores()
    metric_dict['%s/acc' % prefix_note] = score['Overall Acc']
    metric_dict['%s/iou' % prefix_note] = score['Mean IoU']

    class_iou_dict = {}
    for k, v in class_iou.items():
        class_iou_dict['%s_class_iou/%02d' % (prefix_note, k)] = v

    if summary_all:
        for key, v in metric_fn_dict.items():
            score, class_iou = metric_fn_dict[key].get_scores()
            if key in ['seg', 'edge']:
                metric_dict[prefix_note+'_metric/'+key +
                            '_acc'] = score['Overall Acc']
                metric_dict[prefix_note+'_metric/' +
                            key+'_iou'] = score['Mean IoU']
            else:
                metric_dict[prefix_note+'_branch_metric/' +
                            key+'_acc'] = score['Overall Acc']
                metric_dict[prefix_note+'_branch_metric/' +
                            key+'_iou'] = score['Mean IoU']

    return metric_dict, class_iou_dict


def get_image_dict(outputs_dict, targets_dict, config, summary_all=False, prefix_note='train'):
    image_dict = {}
    if summary_all and config.summary_image:
        gpu_num = torch.cuda.device_count()
        # for parallel, the true batch size for image will be config.batch_size//gpu_num
        idx = np.random.randint(config.batch_size//gpu_num)
        # convert tensor to numpy,
        np_outputs_dict = {}
        for key, value in outputs_dict.items():
            np_outputs_dict[key] = value[idx].data.cpu().numpy().argmax(1)

        np_targets_dict = {}
        for key, value in targets_dict.items():
            np_targets_dict[key] = value[idx].data.cpu().numpy()

        seg_pixel_scale = 255//config.class_number
        edge_pixel_scale = 255//config.edge_class_num

        for k, v in np_outputs_dict.items():
            if k.startswith(('seg', 'aux')):
                image_dict['%s/predict_%s' %
                           (prefix_note, k)] = (v*seg_pixel_scale).astype(np.uint8)
            elif k.startswith('edge'):
                image_dict['%s/predict_%s' %
                           (prefix_note, k)] = (v*edge_pixel_scale).astype(np.uint8)
            else:
                assert False, 'unexcepted key %s in outputs_dict' % key

        normalizations = image_normalizations(config.norm_ways)
        for k, v in np_targets_dict.items():
            if k == 'img':
                org_img = v.transpose((1, 2, 0))
                if normalizations is not None:
                    org_img = normalizations.backward(org_img)
                image_dict['%s/%s' %
                           (prefix_note, k)] = org_img.astype(np.uint8)
            elif k == 'seg':
                image_dict['%s/%s' % (prefix_note, k)
                           ] = (v*seg_pixel_scale).astype(np.uint8)
            elif k == 'edge':
                image_dict['%s/%s' % (prefix_note, k)
                           ] = (v*edge_pixel_scale).astype(np.uint8)
            else:
                assert False, 'unexcepted key %s in outputs_dict' % key
    return image_dict


def get_lr_dict(optimizer, prefix_note='train'):
    lr_dict = {}
    for idx, params in enumerate(optimizer.param_groups):
        lr_dict['%s_lr/%d' % (prefix_note, idx)] = params['lr']

    return lr_dict


def init_writer(config, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    config_str = json.dumps(config, indent=2, sort_keys=True).replace(
        '\n', '\n\n').replace('  ', '\t')
    writer.add_text(tag='config', text_string=config_str)

    # write config to config.txt
    config_path = os.path.join(log_dir, 'config.txt')
    config_file = open(config_path, 'w')
    json.dump(config, config_file, sort_keys=True)
    config_file.close()

    return writer


def write_summary(writer,
                  losses_dict,
                  metric_dict,
                  class_iou_dict,
                  lr_dict,
                  image_dict,
                  weight_dict,
                  epoch):
    # losses_dict value is numpy
    for k, v in losses_dict.items():
        writer.add_scalar(k, np.mean(v), epoch)
        writer.add_scalar(k+'_std', np.std(v), epoch)

    # summary metrics
    for k, v in metric_dict.items():
        writer.add_scalar(k, v, epoch)

    # summary class iou
    for k, v in class_iou_dict.items():
        writer.add_scalar(k, v, epoch)

    # summary for class weight
    for k, v in weight_dict.items():
        writer.add_scalar(k, v, epoch)

    # summary learning rate
    for k, v in lr_dict.items():
        writer.add_scalar(k, v, epoch)

    # summary image
    for k, v in image_dict.items():
        add_image(summary_writer=writer, name=k, image=v, step=epoch)
