# -*- coding: utf-8 -*-
import os
import torch
import torch.utils.data as TD
from dataset.dataset_generalize import dataset_generalize, image_normalizations
import glob
from utils.disc_tools import get_newest_file
import numpy as np
from PIL import Image

def get_test_loader(config):
    if config.dataset.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.dataset.norm_ways)
        
    test_dataset=dataset_generalize(config.dataset,split='train',
                                    normalizations=normalizations)
    
    test_loader=TD.DataLoader(dataset=test_dataset,
                              batch_size=config.args.batch_size,
                              shuffle=False,
                              drop_last=False)
    
    return test_loader

def voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def save_pil_image(image,filename,palette):
    pil_img=Image.fromarray(image,mode='P')
    pil_img.putpalette(palette)
    pil_img.save(filename)

def keras_benchmark(model,test_loader,config=None,checkpoint_path=None,predict_save_path=None):
    if config is None:
        config=model.config
    
    if checkpoint_path is None:
        log_dir = os.path.join(config.args.log_dir, model.name,
                           config.dataset.name, config.args.note)
        ckpt_files=glob.glob(os.path.join(log_dir,'**','model-best-*.pkl'),recursive=True)
        
        # use best model first, then use the last model, because the last model will be the newest one if exist.
        if len(ckpt_files)==0:
            ckpt_files=glob.glob(os.path.join(log_dir,'**','*.pkl'),recursive=True)
        
        assert len(ckpt_files)>0,'no weight file found under %s, \n please specify checkpoint path'%log_dir
        checkpoint_path=get_newest_file(ckpt_files)
        print('no checkpoint file given, auto find %s'%checkpoint_path)
    else:
        assert os.path.exists(checkpoint_path),'checkpoint path %s not exist'%checkpoint_path
    
    if predict_save_path is None:
        predict_save_path=os.path.join('output',model.name,config.dataset.name,config.args.note)
        os.makedirs(predict_save_path,exist_ok=True)
        print('no predict save path given, auto use',predict_save_path)
    else:
        assert os.path.exists(predict_save_path),'predict save path %s not exist'%predict_save_path
    
    # support only voc currently, if cityscapes, need convert the image'
    assert config.dataset.name=='VOC2012','current support only voc'
    cmap=voc_color_map()
    palette=list(cmap.reshape(-1))
    
    # support for cpu/gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    state_dict=torch.load(checkpoint_path)
    if 'model_state' in state_dict.keys():
        model.load_state_dict(state_dict['model_state'])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    for step, data in enumerate(test_loader):
        # tensor with shape [b,c,h,w]
        images=data['image']
        image_names=data['filename']
        
        # tensor with shape [b,c,h,w]
        tensor_outputs=model.forward(images)
        # numpy array with shape [b,h,w]
        outputs = torch.argmax(tensor_outputs,dim=1).data.cpu().numpy()
        
        if isinstance(outputs, dict):
            main_output=outputs['seg']
        elif isinstance(outputs, (list, tuple)):
            main_output=outputs[0]
        elif isinstance(outputs, torch.Tensor):
            main_output=outputs
        else:
            assert False, 'unexcepted outputs type %s' % type(outputs)
        
        for idx,f in enumerate(image_names):
            save_filename=os.path.join(predict_save_path,os.path.basename(f)).replace('.jpg','.png')
            save_pil_image(main_output[idx],save_filename,palette)
            print('save image to',save_filename)
        