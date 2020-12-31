# -*- coding: utf-8 -*-
"""
usage:
    python demo.py --config_txt xxx/config.txt
    use config.txt to find weight/checkpoint file
"""

import argparse 
import sys
import glob
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchseg.utils.disc_tools import get_checkpoint_from_txt,show_images
from torchseg.dataset.dataset_generalize import image_normalizations

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config_txt',
                        help='path to config_txt file',
                        required=True,
                        default=None)
    parser.add_argument('--app',
                        help='semantic segmentation or motion segmentation ?',
                        required=True,
                        default=None,
                        choices=['semantic','motion'])
    parser.add_argument('--images',
                        help='the input image list for app',
                        default=[],
                        required=True,
                        nargs='*')
    
    args=parser.parse_args()
    
    if args.app=='motion':
        from torchseg.utils.configs.motionseg_config import (
            load_config)
        assert False,'waiting for implement'
    elif args.app=='semantic':
        from torchseg.utils.configs.semanticseg_config import (
            get_net,
            load_config)
    
    print('use config.txt to find weight file')
    checkpoint_path=get_checkpoint_from_txt(args.config_txt)
    config=load_config(args.config_txt)
    config.backbone_pretrained=False
    model=get_net(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    state_dict=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 
    normalizations = image_normalizations(config.norm_ways)
    for img_f in args.images:
        img=cv2.imread(img_f,cv2.IMREAD_COLOR)
        # for opencv, resize input is (w,h)
        dsize=(config.input_shape[1],config.input_shape[0])
        img = cv2.resize(src=img, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_img = img.copy()
        
        img=normalizations.forward(img)
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img,0)
        
        net_input=torch.tensor(img).to(device).float()
        outputs=model(net_input)
        
        if isinstance(outputs, dict):
            main_output=outputs['seg']
        elif isinstance(outputs, (list, tuple)):
            main_output=outputs[0]
        elif isinstance(outputs, torch.Tensor):
            main_output=outputs
        else:
            assert False, 'unexcepted outputs type %s' % type(outputs)
        
        # main_output [b,c,h,w]
        main_output = torch.argmax(main_output,dim=1)
        # main_output [b,h,w]
        main_output=main_output.data.cpu().numpy()
    
        show_images([origin_img,main_output[0]],["image","prediction"])