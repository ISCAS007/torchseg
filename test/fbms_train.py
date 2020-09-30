# -*- coding: utf-8 -*-
"""
dataset loader: cdnet not support two gt file

"""
import numpy as np
import torch.utils.data as td
from models.motion_stn import stn_loss
from utils.metric.motionseg_metric import MotionSegMetric
from utils.config.motionseg_config import get_default_config
from models.motionseg.motion_utils import (get_parser,
                                           get_dataset,
                                           fine_tune_config,get_model,
                                           poly_lr_scheduler,
                                           get_load_convert_model)
from utils.torch_tools import init_writer
from utils.losses import jaccard_loss,dice_loss
import torch.nn.functional as F
import os
import torch
import time
import sys
from tqdm import trange,tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import cv2
from easydict import EasyDict as edict
from utils.davis_benchmark import benchmark
import json

def get_dist_module(config):
    model=get_model(config)

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
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=(xxx_sampler is None),drop_last=False,num_workers=2,sampler=xxx_sampler,pin_memory=True)
        else:
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=True)
        dataset_loaders[split]=xxx_loader

    return model,seg_loss_fn,optimizer,dataset_loaders

def is_main_process(config):
    return not config.use_sync_bn or (config.use_sync_bn and config.rank % config.ngpus_per_node == 0)

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
                frames=data['images']
                gt=data['labels'][0]
                images = [torch.autograd.Variable(img.to(device).float()) for img in frames]

                aux_input = []
                for c in config.input_format:
                    if c.lower()=='b':
                        assert False
                    elif c.lower()=='g':
                        origin_aux_gt=torch.autograd.Variable(data['labels'][1].to(device).long())
                        #print(origin_aux_gt.shape)
                        resize_aux_gt=F.interpolate(origin_aux_gt.float(),size=config.input_shape,mode='nearest').float()
                        aux_input.append(resize_aux_gt)
                    elif c.lower()=='n':
                        aux_input.append(torch.autograd.Variable(data['images'][1].to(device).float()))
                    elif c.lower()=='o':
                        aux_input.append(torch.autograd.Variable(data['optical_flow'].to(device).float()))
                    elif c.lower()=='-':
                        pass
                    else:
                        assert False

                if len(aux_input)>0:
                    images[1]=torch.autograd.Variable(torch.cat(aux_input,dim=1).to(device).float())

                origin_labels=torch.autograd.Variable(gt.to(device).long())
                labels=F.interpolate(origin_labels.float(),size=config.input_shape,mode='nearest').long()

                if config.use_sync_bn:
                    images=[img.cuda(config.gpu,non_blocking=True) for img in images]
                    origin_labels=origin_labels.cuda(config.gpu,non_blocking=True)
                    labels=labels.cuda(config.gpu,non_blocking=True)

                if split=='train':
                    poly_lr_scheduler(config,optimizer,
                              iter=epoch*N+step,
                              max_iter=config.epoch*N)

                outputs=model.forward(images)
                if config.net_name=='motion_anet':
                    mask_gt=torch.squeeze(labels,dim=1)
                    mask_loss_value=0
                    for mask in outputs['masks']:
                        mask_loss_value+=seg_loss_fn(mask,mask_gt)
                elif config.net_name=='motion_diff':
                    assert 'g' in config.input_format.lower()
                    gt_plus=(labels-resize_aux_gt).clamp_(min=0).float()
                    gt_minus=(resize_aux_gt-labels).clamp_(min=0).float()
                    mask_gt=torch.cat([gt_plus,gt_minus,labels.float()],dim=1)
                    ignore_index=255

                    predict=outputs['masks'][0]
                    predict[mask_gt==ignore_index]=0
                    mask_gt[mask_gt==ignore_index]=0
                    mask_loss_value=seg_loss_fn(predict.float(),mask_gt.float())
                else:
                    mask_loss_value=seg_loss_fn(outputs['masks'][0],torch.squeeze(labels,dim=1))

                if config['net_name'].find('_stn')>=0:
                    if config['stn_object']=='features':
                        stn_loss_value=stn_loss(outputs['features'],labels.float(),outputs['pose'],config['pose_mask_reg'])
                    elif config['stn_object']=='images':
                        stn_loss_value=stn_loss(outputs['stn_images'],labels.float(),outputs['pose'],config['pose_mask_reg'])
                    else:
                        assert False,'unknown stn object %s'%config['stn_object']

                    total_loss_value=mask_loss_value*config['motion_loss_weight']+stn_loss_value*config['stn_loss_weight']
                else:
                    stn_loss_value=torch.tensor(0.0)
                    total_loss_value=mask_loss_value

                if config.net_name=='motion_diff':
                    origin_mask=F.interpolate(outputs['masks'][0][:,2:3,:,:], size=origin_labels.shape[2:4],mode='bilinear')
                    origin_mask=torch.cat([1-origin_mask,origin_mask],dim=1)
                else:
                    origin_mask=F.interpolate(outputs['masks'][0], size=origin_labels.shape[2:4],mode='bilinear')
                
                motionseg_metric.update({"fmeasure":(origin_mask,origin_labels),
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

def compute_fps(config):
    batch_size=config.batch_size
    model=get_load_convert_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    split='val'
    xxx_dataset=get_dataset(config,split)
    xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=batch_size,shuffle=False,num_workers=batch_size,pin_memory=True)

    tqdm_step = tqdm(xxx_loader, desc='steps', leave=False)
    counter=0.0
    total_time=0.0

    for step,data in enumerate(tqdm_step):
        frames=data['images']
        images = [img.to(device).float() for img in frames]
        start_time=time.time()
        outputs=model.forward(images)
        total_time+=(time.time()-start_time)
        counter+=outputs['masks'][0].shape[0]

        if counter>100:
            break
    fps=counter/total_time
    print(f'fps={fps}')

    fps_summary_file=os.path.expanduser('~/tmp/result/fps.json')
    with open(fps_summary_file,'r+') as f:
        try:
            fps_summary=json.load(f)
        except:
            fps_summary=dict()
        finally:
            f.seek(0)
            fps_summary[config.note]=fps
            json.dump(fps_summary,f)
            f.truncate()

def test(config):
    model=get_load_convert_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.dataset.upper() in ['DAVIS2017','DAVIS2016']:
        if config.app=='test':
            split_set=['val']
        elif config.app=='benchmark':
            split_set=['test-dev','test-challenge']
        else:
            assert False

        for split in split_set:
            save_dir=os.path.join(os.path.expanduser('~/tmp/result'),config.dataset,split,config.note)
            xxx_dataset=get_dataset(config,split)
            xxx_loader=td.DataLoader(dataset=xxx_dataset,batch_size=1,shuffle=False,num_workers=2,pin_memory=True)
            tqdm_step = tqdm(xxx_loader, desc='steps', leave=False)
            for step,data in enumerate(tqdm_step):
                frames=data['images']
                main_path=data['main_path'][0]
                height,width,_=data['shape']
                height,width=height[0],width[0]

                save_path=xxx_dataset.get_result_path(save_dir,main_path)
                assert save_path!=main_path
                images = [img.to(device).float() for img in frames]
                outputs=model.forward(images)
                result_mask=F.interpolate(outputs['masks'][0], size=(height,width),mode='nearest')
                # print(result_mask.shape) # (batch_size,2,height,width)
                np_mask=np.squeeze(np.argmax(result_mask.data.cpu().numpy(),axis=1)).astype(np.uint8)
                # print(np_mask.shape) # (height,width)

                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                print(f'save image to {save_path}')
                cv2.imwrite(save_path,np_mask)

            if split=='val':
                args=edict()
                args.davis_path=os.path.expanduser('~/cvdataset/DAVIS')
                args.set='val'
                args.task='unsupervised'
                args.results_path=save_dir
                benchmark(args)
    else:
        assert False,'not supported dataset for test'

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
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser=get_parser()
    args = parser.parse_args()

    config=get_default_config()

    torch.hub.set_dir(os.path.expanduser('~/.torch/models'))

    if args.net_name=='motion_psp':
        if args.use_none_layer is False or args.upsample_layer<=3:
            min_size=30*config.psp_scale*2**config.upsample_layer
        else:
            min_size=30*config.psp_scale*2**3

        config.input_shape=[min_size,min_size]

    sort_keys=sorted(list(config.keys()))
    for key in sort_keys:
        if hasattr(args,key):
            print('{} = {} (default: {})'.format(key,args.__dict__[key],config[key]))
            config[key]=args.__dict__[key]
        else:
            print('{} : (default:{})'.format(key,config[key]))

    for key in args.__dict__.keys():
        if key not in config.keys():
            print('{} : unused keys {}'.format(key,args.__dict__[key]))

    # update config according to basic config
    config=fine_tune_config(config)

    if args.app=='dataset':
        for split in ['train','val']:
            xxx_dataset=get_dataset(config,split)
            dataset_size=len(xxx_dataset)
            for idx in range(dataset_size):
                xxx_dataset.__getitem__(idx)
        sys.exit(0)
    elif args.app=='summary':
        import torchsummary

        config.gpu=0
        model,seg_loss_fn,optimizer,dataset_loaders=get_dist_module(config)
        # not work for output with dict.
        torchsummary.summary(model, ((3, config.input_shape[0], config.input_shape[1]),
                                     (2, config.input_shape[0], config.input_shape[1])))
        sys.exit(0)
    elif args.app in ['test','benchmark']:
        test(config)
    elif args.app == 'fps':
        compute_fps(config)
    elif config.use_sync_bn:
        dist_train(config)
    else:
        main_worker(gpu=0,ngpus_per_node=1,config=config)
