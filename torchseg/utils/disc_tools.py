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

def get_special_cmap(base_cmap_name='Set3',bad_color='black'):
    cmap=plt.get_cmap(base_cmap_name)
    cmap.set_over(bad_color)
    cmap.set_bad(bad_color)
    return cmap

def get_diff(img1,img2,bg=0,out_of_roi=255):
    assert len(img1.shape)==len(img2.shape)==2,'support only label image'
    assert img1.dtype==img2.dtype==np.uint8,'support only uint8 label image'

    h,w=img1.shape
    merge_image=np.zeros((h,w,3),np.uint8)
    vmax=np.max(img1)
    merge_image[:,:,0]=img1*(255//vmax)
    diff=np.zeros_like(img1)
    if bg==0:
        diff[np.logical_and(img1==bg,img1==img2)]=0
        diff[np.logical_and(img1!=bg,img1==img2)]=1
        diff[np.logical_and(img1==bg,img1!=img2)]=2
        diff[np.logical_and(img1!=bg,img1!=img2)]=3

        merge_image[:,:,1]=diff*(255//3)
    else:
        warnings.warn('bg is not 0, there is no background?')
        diff[img1==img2]=1
        diff[img1!=img2]=2

        merge_image[:,:,1]=diff*(255//2)

    diff[img1==out_of_roi]=out_of_roi
    merge_image[:,:,2]=255*(img1==out_of_roi).astype(np.uint8)
    return diff,merge_image

def show_images(images,titles=None,vmin=None,vmax=None,cmap=None):
    if len(images)==1:
        plt.imshow(images[0],vmin=vmin,vmax=vmax,cmap=cmap)
        if titles is None:
            plt.title("image")
        else:
            plt.title(titles[0])
        plt.show()
    else:
        fig, axes = plt.subplots(2, (len(images)+1)//2, figsize=(7, 6), sharex=True, sharey=True)
        ax = axes.ravel()

        for i in range(len(images)):
            ax[i].imshow(images[i],vmin=vmin,vmax=vmax,cmap=cmap)
            if titles is None:
                ax[i].set_title("image %d"%i)
            else:
                ax[i].set_title(titles[i])

        plt.show()

def show_tensor_list(images_tensor,title,normer=None,vmin=None,vmax=None,cmap=None):
    """
    show image tensor with shape [b,c,h,w], c>=2 or [b,h,w]
    
    Parameters
    ----------
    images_tensor : pytorch tensor list 
        pytorch tensor list.
    title : string list 
        matplot figure title list.
    normer : image_normalizations, optional
        normalizations for image. The default is None.
    vmin : TYPE, optional
        DESCRIPTION. The default is None.
    vmax : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    for idx,t in enumerate(images_tensor):
        batch_images=t.data.cpu().numpy()
        
        if len(batch_images.shape)==3:
            b,h,w=batch_images.shape
            image_list=np.split(batch_images,b)
            image_list=[np.squeeze(img) for img in image_list]
            show_images(image_list,[title[idx] for img in image_list],cmap=cmap,vmax=vmax,vmin=vmin)
            continue
            
        
        b,c,h,w=batch_images.shape
        assert c>=2,'the channel must >=2'
        batch_images=batch_images.transpose((0,2,3,1))

        print(idx,np.mean(batch_images),np.std(batch_images),np.max(batch_images),np.min(batch_images))
        if normer is not None:
            if batch_images.shape[-1]==2:
                pass
            else:
                batch_images=normer.backward(batch_images).astype(np.uint8)
        else:
            if np.max(batch_images)==255:
                batch_images=batch_images.astype(np.uint8)
            #batch_images=np.clip(batch_images,a_min=0,a_max=1)

        b,h,w,c=batch_images.shape
        image_list=np.split(batch_images,b)
        image_list=[np.squeeze(img) for img in image_list]
        for i,img in enumerate(image_list):
            if img.shape[-1]==2:
                h,w,c=img.shape
                img_3=np.zeros((h,w,3),dtype=np.float)
                img_3[:,:,0:2]=img
                image_list[i]=img_3
        show_images(image_list,[title[idx] for img in image_list],cmap=cmap,vmax=vmax,vmin=vmin)

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