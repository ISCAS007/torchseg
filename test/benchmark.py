# -*- coding: utf-8 -*-
"""
voc benchmark
"""

import os
import torch
import torch.utils.data as TD
import glob

from torchseg.dataset.dataset_generalize import dataset_generalize, image_normalizations
from torchseg.utils.disc_tools import get_newest_file
from torchseg.utils.augmentor import Augmentations
import numpy as np
from PIL import Image
import cv2
import warnings

def get_loader(config,split):
    if config.norm_ways is None:
        normalizations = None
    else:
        normalizations = image_normalizations(config.norm_ways)



    if split=='test':
        test_dataset=dataset_generalize(config,split=split,
                                    normalizations=normalizations)
        test_loader=TD.DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  drop_last=False)

        return test_loader
    else:
        assert split in ['train','val']
        if config.augmentation:
            augmentations = Augmentations(config)
        else:
            augmentations = None

        dataset = dataset_generalize(
            config, split=split,
            augmentations=augmentations,
            normalizations=normalizations)
        loader = TD.DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=(split=='train'),
            drop_last=(split=='train'),
            num_workers=8)

        return loader


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
#    assert image.dtype==np.uint8
    if image.dtype!=np.uint8:
        warnings.warn('warning: when save numpy image to pil image, use np.uint8 as dtype')
        save_image=image.astype(np.uint8)
    else:
        save_image=image
    pil_img=Image.fromarray(save_image,mode='P')
    pil_img.putpalette(palette)
    pil_img.save(filename)

def keras_benchmark(model,test_loader=None,config=None,checkpoint_path=None,predict_save_path=None):
    if config is None:
        config=model.config

    if checkpoint_path is None:
        log_dir = os.path.join(config.log_dir, model.name,
                           config.dataset_name, config.note)
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
        predict_save_path=os.path.join('output',model.name,config.dataset_name,config.note)
        os.makedirs(predict_save_path,exist_ok=True)
        print('no predict save path given, auto use',predict_save_path)
    else:
        assert os.path.exists(predict_save_path),'predict save path %s not exist'%predict_save_path

    # support only voc currently, if cityscapes, need convert the image'
    assert config.dataset_name=='voc2012','current support only voc, not %s'%config.dataset_name
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

    if test_loader is None:
        test_loader=get_loader(config,'test')
    for step, data in enumerate(test_loader):
        # tensor with shape [b,c,h,w]
        if isinstance(data['image'],(tuple,list)):
            images=data['image'][0].to(device).float()
            image_names=data['filename'][1]
        else:
            images=data['image'].to(device).float()
            image_names=data['filename']

        # tensor with shape [b,c,h,w]
        tensor_outputs=model.forward(images)
        # numpy array with shape [b,h,w]
        outputs = torch.argmax(tensor_outputs,dim=1)

        if isinstance(outputs, dict):
            main_output=outputs['seg']
        elif isinstance(outputs, (list, tuple)):
            main_output=outputs[0]
        elif isinstance(outputs, torch.Tensor):
            main_output=outputs
        else:
            assert False, 'unexcepted outputs type %s' % type(outputs)

        main_output=main_output.data.cpu().numpy()
        for idx,f in enumerate(image_names):
            save_filename=os.path.join(predict_save_path,os.path.basename(f)).replace('.jpg','.png')
            origin_img=cv2.imread(f)
            origin_shape=origin_img.shape
            resize_img=cv2.resize(main_output[idx],
                                  dsize=(origin_shape[1],origin_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            assert resize_img.shape[0:2]==origin_img.shape[0:2],'{} vs {}'.format(resize_img.shape,origin_img.shape)
            save_pil_image(resize_img,save_filename,palette)
            print('save image to',save_filename)
