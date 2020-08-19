# -*- coding: utf-8 -*-
import numpy as np
import torch
import math
import os
import matplotlib.pyplot as plt
import glob
import warnings

# Minimum common multiple or least common multiple
def lcm(a,b):
    return a*b//math.gcd(a,b)

def lcm_list(l):
    x=1
    for i in l:
       x=lcm(i,x)

    return x


def show_images(images,titles=None,vmin=None,vmax=None):
    fig, axes = plt.subplots(2, (len(images)+1)//2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(images)):
        ax[i].imshow(images[i],vmin=vmin,vmax=vmax)
        if titles is None:
            ax[i].set_title("image %d"%i)
        else:
            ax[i].set_title(titles[i])

    plt.show()

def show_tensor_list(images_tensor,title,normer=None):
    for idx,t in enumerate(images_tensor):
        batch_images=t.data.cpu().numpy()
        batch_images=batch_images.transpose((0,2,3,1))

        print(idx,np.mean(batch_images),np.std(batch_images),np.max(batch_images),np.min(batch_images))
        if normer is not None:
            if batch_images.shape[-1]==2:
                pass
            else:
                batch_images=normer.backward(batch_images).astype(np.uint8)
        else:
            batch_images=np.clip(batch_images,a_min=0,a_max=1)

        b,h,w,c=batch_images.shape
        image_list=np.split(batch_images,b)
        image_list=[np.squeeze(img) for img in image_list]
        for i,img in enumerate(image_list):
            if img.shape[-1]==2:
                h,w,c=img.shape
                img_3=np.zeros((h,w,3),dtype=np.float)
                img_3[:,:,0:2]=img
                image_list[i]=img_3
        show_images(image_list,[title[idx] for img in image_list])

def str2bool(s):
    if s.lower() in ['t','true','y','yes','1']:
        return True
    elif s.lower() in ['f','false','n','no','0']:
        return False
    else:
        assert False,'unexpected value for bool type'

def changed_or_not(backbone_name,param_name):
    changed_threshold={
            "vgg19_bn":39,
            "vgg19":27,
            "vgg16_bn":33,
            "vgg16":23,
            "vgg13_bn":27,
            "vgg13":19,
            "vgg11_bn":21,
            "vgg11":15,
            }

    idx=int(param_name.split('.')[0])
    return idx>=changed_threshold[backbone_name]

def get_backbone_optimizer_params(backbone_name,
                              backbone,
                              unchanged_lr_mult=1,
                              changed_lr_mult=10,
                              new_lr_mult=20):

    unchanged_backbone={'params':[],'lr_mult':unchanged_lr_mult}
    modified_backbone={'params':[],'lr_mult':changed_lr_mult}
    new_backbone={'params':[],'lr_mult':new_lr_mult}
    if backbone_name in ['vgg16','vgg16_bn','vgg19','vgg19_bn','vgg11','vgg11_bn','vgg13','vgg13_bn']:
        for param_name, p in backbone.features.named_parameters():
            if p.requires_grad:
                changed=changed_or_not(backbone_name,param_name)
                if changed:
                    modified_backbone['params'].append(p)
                else:
                    unchanged_backbone['params'].append(p)
        return [unchanged_backbone,modified_backbone]
    elif backbone_name in ['resnet50','resnet101']:
        modify_resnet_head=str2bool(os.environ['modify_resnet_head'])
        for child_name, child in backbone.named_children():
#            print(child_name,'*'*10)
            if child_name in ['avgpool','fc']:
                continue
            elif child_name in ['layer3','layer4']:
                modified_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
            elif child_name == 'prefix_net':
                if modify_resnet_head:
                    new_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
                else:
                    unchanged_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
            else:
                unchanged_backbone['params']+=[p for p in child.parameters() if p.requires_grad]

        if modify_resnet_head:
            return [unchanged_backbone,modified_backbone,new_backbone]
        else:
            return [unchanged_backbone,modified_backbone]
    else:
        assert False,'unknonw backbone name %s'%backbone_name

def save_model_if_necessary(model,config,checkpoint_path):
    save_model=False
    if hasattr(config,'save_model'):
        save_model=config.save_model

    if save_model:
        torch.save(model.state_dict(),checkpoint_path)

def get_newest_file(files):
    t=0
    newest_file=None
    for full_f in files:
        if os.path.isfile(full_f):
            file_create_time = os.path.getctime(full_f)
            if file_create_time > t:
                t = file_create_time
                newest_file = full_f

    return newest_file

def get_checkpoint_from_txt(txt_path):
    log_dir=os.path.dirname(txt_path)
    ckpt_files=glob.glob(os.path.join(log_dir,'**','model-best-*.pkl'),recursive=True)

    # use best checkpoint if available
    if len(ckpt_files)==0:
        ckpt_files=glob.glob(os.path.join(log_dir,'**','*.pkl'),recursive=True)

    if len(ckpt_files)==0:
        warnings.warn('cannot obtain checkpoint file from {}'.format(txt_path))
        return None
    else:
        return get_newest_file(ckpt_files)

def remove_file(root_path,suffix):
    files=glob.glob(os.path.join(root_path,'**','*.{}'.format(suffix)),recursive=True)
    for f in files:
        os.system('rm {}'.format(f))