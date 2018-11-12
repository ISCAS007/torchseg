# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import math
import os

# Minimum common multiple or least common multiple
def lcm(a,b):
    return a*b//math.gcd(a,b)

def lcm_list(l):
    x=1
    for i in l:
       x=lcm(i,x)
       
    return x


def show_images(images,titles=None):
    fig, axes = plt.subplots(2, (len(images)+1)//2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    for i in range(len(images)):
        ax[i].imshow(images[i])
        if titles is None:
            ax[i].set_title("image %d"%i)
        else:
            ax[i].set_title(titles[i])

    plt.show()
    
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
        for child_name, child in backbone.named_children():
#            print(child_name,'*'*10)
            if child_name in ['avgpool','fc']:
                continue
            elif child_name in ['layer3','layer4']:
                modified_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
            elif child_name == 'prefix_net':
                new_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
            else:
                unchanged_backbone['params']+=[p for p in child.parameters() if p.requires_grad]
                
        return [unchanged_backbone,modified_backbone,new_backbone]
    else:
        assert False,'unknonw backbone name %s'%backbone_name
        
def save_model_if_necessary(model,config,checkpoint_path):
    save_model=False
    if hasattr(config.args,'save_model'):
        save_model=config.args.save_model
        
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