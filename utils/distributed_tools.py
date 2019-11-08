# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from utils.config import get_net
from utils.augmentor import Augmentations
from dataset.dataset_generalize import dataset_generalize, image_normalizations
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.torch_tools import (get_optimizer,get_scheduler,get_loss_fn_dict,
                         train_val,get_metric,get_image_dict,
                         get_lr_dict,init_writer,write_summary,is_main_process)
from utils.metrics import runningScore
import random
import time
from tqdm import trange,tqdm
from utils.poly_plateau import poly_rop
from torch.optim.lr_scheduler import ReduceLROnPlateau as rop
from torch.optim.lr_scheduler import CosineAnnealingLR as cos_lr
from utils.disc_tools import save_model_if_necessary

def get_dist_module(config):
    model=get_net(config)
    if config.dist:
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model.cuda(config.gpu)
            if config.use_sync_bn:
                model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model=DDP(model,device_ids=[config.gpu],find_unused_parameters=True)
        else:
            model.cuda()
            if config.use_sync_bn:
                model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model=DDP(model,find_unused_parameters=True)
    elif config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model=model.cuda(config.gpu)
    else:
        if config.use_sync_bn:
            model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model=torch.nn.DataParallel(model).cuda()


    # criterion=torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index).cuda(config.gpu)
    loss_fn_dict = get_loss_fn_dict(config)
    for key,value in loss_fn_dict.items():
        loss_fn_dict[key]=value.cuda(config.gpu)

    optimizer=get_optimizer(model,config)

    return model,loss_fn_dict,optimizer

def main_worker(gpu,ngpus_per_node,config):
    config.gpu=gpu

    if config.dist:
        if config.dist_url=='env://' and config.rank==-1:
            config.rank=int(os.environ['RANK'])

        if config.mp_dist:
            config.rank=config.rank*ngpus_per_node+gpu

        dist.init_process_group(backend=config.dist_backend,
                                init_method=config.dist_url,
                                world_size=config.world_size,
                                rank=config.rank)

    model,loss_fn_dict,optimizer=get_dist_module(config)

    cudnn.benchmark=True

    if config.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.norm_ways)

    if config.augmentation:
        augmentations = Augmentations(config)
    else:
        augmentations = None

    train_dataset = dataset_generalize(config,
                                       split='train',
                                       augmentations=augmentations,
                                       normalizations=normalizations)


    val_dataset = dataset_generalize(config,
                                     split='val',
                                     augmentations=None,
                                     normalizations=normalizations)

    if config.dist:
        train_sampler=torch.utils.data.DistributedSampler(train_dataset)
    else:
        train_sampler=None

    if config.dist and config.gpu is not None:
        batch_size=int(config.batch_size/ngpus_per_node)
        num_workers=int(config.num_workers/ngpus_per_node)
    else:
        batch_size=config.batch_size
        num_workers=config.num_workers

    train_loader=torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=(train_sampler is None),
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             sampler=train_sampler)

    val_loader=torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)

    scheduler = get_scheduler(optimizer, config)

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

    loaders = [train_loader, val_loader]
    loader_names = ['train', 'val']

    # eval module
    if train_loader is None:
        config.n_epoch = 1

    summary_all_step=max(1,config.n_epoch//10)
    # 1<= summary_metric_step <=10
    summary_metric_step=max(min(10*config.accumulate,config.n_epoch//10),1)

    config.ngpus_per_node=ngpus_per_node
    if is_main_process(config):
        tqdm_epoch = trange(config.n_epoch, desc='{} epoches'.format(config.note), leave=True)
    else:
        tqdm_epoch = range(config.n_epoch)
    for epoch in tqdm_epoch:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process(config):
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
                        loader_name=loader_name)

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
                    loader_name=loader_name)

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

def dist_train(config):
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic=True

    config.dist=config.n_node > 1 or config.mp_dist

    ngpus_per_node=torch.cuda.device_count()
    if config.mp_dist:
        config.world_size=ngpus_per_node*config.n_node
        mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,config))
    else:
        config.world_size=ngpus_per_node
        main_worker(config.gpu,ngpus_per_node,config)
