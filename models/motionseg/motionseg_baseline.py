# -*- coding: utf-8 -*-
"""
baseline model for motion seg
conda activate torch1.6
python models/motionseg/motionseg_baseline.py --dataset cdnet2014 --note baseline --net_name Unet
python models/motionseg/motionseg_baseline.py --dataset cdnet2014 --note baseline --net_name DeepLabV3Plus --backbone_name resnet50

different from test/fbms_train.py, here the model is the baseline model, but in fbms_train, the model behavior like motion_diff.
"""

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from utils.configs.motionseg_config import update_default_config
from dataset.motionseg_dataset_factory import prepare_input_output
from models.motionseg.motion_utils import (get_parser,
                                           get_dataset,
                                           get_model,
                                           poly_lr_scheduler,
                                           get_load_convert_model)
from utils.torch_tools import init_writer
from utils.losses import jaccard_loss,dice_loss
from tqdm import trange,tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.metric.motionseg_metric import MotionSegMetric
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as td
import torch
import time,random
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

def is_main_process(config):
    return not config.use_sync_bn or (config.use_sync_bn and config.rank % config.ngpus_per_node == 0)

def get_dist_module(config):
    model=smp.__dict__[config.net_name](encoder_name=config.backbone_name,
                                     encoder_depth=config.deconv_layer,
                                     encoder_weights='imagenet',
                                     classes=2,
                                     in_channels=3)

    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.use_sync_bn:
        torch.cuda.set_device(config.gpu)
        model.cuda(config.gpu)
        model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model=DDP(model,find_unused_parameters=True,device_ids=[config.gpu])
    else:
        model.to(device)

    if config.loss_name in ['iou','dice']:
        # iou loss not support ignore_index
        assert config.dataset not in ['cdnet2014','all','all2','all3']
        assert config.ignore_pad_area==0
        if config.loss_name=='iou':
            seg_loss_fn=jaccard_loss
        else:
            seg_loss_fn=dice_loss
    elif config.net_name == 'motion_diff':
        seg_loss_fn=torch.nn.BCEWithLogitsLoss()
    else:
        seg_loss_fn=torch.nn.CrossEntropyLoss(ignore_index=255)

    if config.use_sync_bn:
        seg_loss_fn=seg_loss_fn.cuda(config.gpu)

    optimizer_params = [{'params': [p for p in model.parameters() if p.requires_grad]}]

    if config.optimizer=='adam':
        optimizer = torch.optim.Adam(
                    optimizer_params, lr=config['init_lr'], amsgrad=False)
    else:
        assert config.init_lr>1e-3
        optimizer = torch.optim.SGD(
                    optimizer_params, lr=config['init_lr'], momentum=0.9, weight_decay=1e-4)

    dataset_loaders={}
    for split in ['train','val']:
        xxx_dataset=get_dataset(config,split)

        if config.use_sync_bn and split=='train':
            xxx_sampler=torch.utils.data.DistributedSampler(xxx_dataset)
        else:
            xxx_sampler=None

        batch_size=config.batch_size if split=='train' else 1

        if split=='train':
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=(xxx_sampler is None),drop_last=True,num_workers=2,sampler=xxx_sampler,pin_memory=True)
        else:
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
        dataset_loaders[split]=xxx_loader

    return model,seg_loss_fn,optimizer,dataset_loaders

def train(config,model,seg_loss_fn,optimizer,dataset_loaders):
    if is_main_process(config):
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir = os.path.join(config['log_dir'], config['net_name'],
                               config['dataset'], config['note'], time_str)
        checkpoint_path = os.path.join(log_dir, 'model-last-%d.pkl' % config['epoch'])

        writer=init_writer(config,log_dir)

    motionseg_metric=MotionSegMetric(config.exception_value)

    if is_main_process(config):
        tqdm_epoch = trange(config['epoch'], desc='{} epochs'.format(config.note), leave=True)
    else:
        tqdm_epoch=range(config.epoch)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    step_acc=0
    for epoch in tqdm_epoch:
        for split in ['train','val']:
            if split=='train':
                model.train()
            else:
                model.eval()

            motionseg_metric.reset()

            if is_main_process(config):
                tqdm_step = tqdm(dataset_loaders[split], desc='steps', leave=False)
            else:
                tqdm_step = dataset_loaders[split]

            N=len(dataset_loaders[split])
            for step,data in enumerate(tqdm_step):
                images,origin_labels,resize_labels=prepare_input_output(data=data,config=config,device=device)

                if split=='train':
                    poly_lr_scheduler(config,optimizer,
                              iter=epoch*N+step,
                              max_iter=config.epoch*N)

                outputs=model.forward(images[0])
                mask_loss_value=seg_loss_fn(outputs,torch.squeeze(resize_labels[0].long(),dim=1))

                stn_loss_value=torch.tensor(0.0)
                total_loss_value=mask_loss_value

                origin_mask=F.interpolate(outputs, size=origin_labels[0].shape[2:4],mode='bilinear')

                motionseg_metric.update({"fmeasure":(origin_mask,origin_labels[0]),
                                         "stn_loss":stn_loss_value.item(),
                                         "mask_loss":mask_loss_value.item(),
                                         "total_loss":total_loss_value.item()})

                if split=='train':
                    total_loss_value.backward()
                    if (step_acc+1)>=config.accumulate:
                        optimizer.step()
                        optimizer.zero_grad()
                        step_acc=0
                    else:
                        step_acc+=1

            if is_main_process(config):
                motionseg_metric.write(writer,split,epoch)
                current_metric=motionseg_metric.fetch()
                fmeasure=current_metric['fmeasure'].item()
                mean_total_loss=current_metric['total_loss']
                if split=='train':
                    tqdm_epoch.set_postfix(train_fmeasure=fmeasure)
                else:
                    tqdm_epoch.set_postfix(val_fmeasure=fmeasure)

                if epoch % 10 == 0:
                    print(split,'fmeasure=%0.4f'%fmeasure,
                          'total_loss=',mean_total_loss)

    if is_main_process(config) and config['save_model']:
        torch.save(model.state_dict(),checkpoint_path)

    if is_main_process(config):
        writer.close()


def main_worker(gpu,ngpus_per_node,config):
    config.gpu=gpu

    if config.use_sync_bn:
        if config.dist_url=='env://' and config.rank==-1:
            config.rank=int(os.environ['RANK'])

        config.rank=gpu
        #config.rank=config.rank*ngpus_per_node+gpu

        dist.init_process_group(backend=config.dist_backend,
                                init_method=config.dist_url,
                                world_size=config.world_size,
                                rank=config.rank)

    model,loss_fn_dict,optimizer,dataset_loaders=get_dist_module(config)
    cudnn.benchmark=True
    train(config,model,loss_fn_dict,optimizer,dataset_loaders)


def dist_train(config):
    config.dist_backend='nccl'
    config.dist_url='tcp://127.0.0.1:9876'
    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        cudnn.deterministic=True

    ngpus_per_node=torch.cuda.device_count()
    config.ngpus_per_node=ngpus_per_node
    if config.use_sync_bn:
        config.world_size=ngpus_per_node
        mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,config))
    else:
        config.world_size=ngpus_per_node
        main_worker(config.gpu,ngpus_per_node,config)

if __name__ == '__main__':
    torch.hub.set_dir(os.path.expanduser('~/.torch/models'))
    parser=get_parser()
#    models=['Unet','DeepLabV3','FPN','PSPNet','Linknet',
#                                 'PSPNet','DeepLabV3Plus']
#    models+=[m.lower() for m in models]
#
#    parser.add_argument('--baseline',
#                        default='Unet',
#                        choices=models,
#                        help='the baseline model name(Unet)')

    args = parser.parse_args()

    config=update_default_config(args)
#    config.baseline=config.net_name=args.baseline

    if config.use_sync_bn:
        dist_train(config)
    else:
        main_worker(gpu=0,ngpus_per_node=1,config=config)

