# -*- coding: utf-8 -*-
from utils.config import load_config
from utils.notebook import get_model_and_dataset
from models.motionseg.motion_utils import get_parser,get_default_config,get_dataset
from models.motionseg.motion_utils import fine_tune_config
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm,trange
import torch
import fire
import os
import cv2
import matplotlib.pyplot as plt
import glob
import torch.utils.data as td
from dataset.dataset_generalize import image_normalizations

class sy_dataset(td.Dataset):
    def __init__(self,config,normalizations=None,augmentations=None):
        self.config=config
        self.normalizations=normalizations
        self.augmentations=augmentations
        self.input_shape=tuple(config.input_shape)
        self.use_optical_flow=config.use_optical_flow
        
        self.frames=glob.glob(os.path.join(config.root_path,'*.jpg'))
        self.frames.sort()
        
        print('dataset size is',len(self.frames))
    
    def __len__(self):
        return len(self.frames)//2
    
    def __getitem__(self,index):
        frames=[self.frames[2*index],self.frames[2*index+1]]
        frame_images=[cv2.imread(f,cv2.IMREAD_COLOR) for f in frames]
        resize_frame_images=[cv2.resize(img,self.input_shape,interpolation=cv2.INTER_LINEAR) for img in frame_images]
        
        # normalize image
        if self.normalizations is not None:
            resize_frame_images = [self.normalizations.forward(img) for img in resize_frame_images]
        
        # bchw
        resize_frame_images=[img.transpose((2,0,1)) for img in resize_frame_images]
        
        if self.use_optical_flow:
            flow_path=frames[0].replace('jpg','flow')
            flow_file=open(flow_path,'r')
            a=np.fromfile(flow_file,np.uint8,count=4)
            b=np.fromfile(flow_file,np.int32,count=2)
            flow=np.fromfile(flow_file,np.float32).reshape((b[1],b[0],2))
            flow=np.clip(flow,a_min=-50,a_max=50)/50.0
            optical_flow=cv2.resize(flow,self.input_shape,interpolation=cv2.INTER_LINEAR).transpose((2,0,1))
            return {'paths':frames,
                    'images':[resize_frame_images[0],optical_flow]}
        else:
            return {'paths':frames,
                    'images':resize_frame_images}

def generate_flow():
    flow_execute_dir=os.path.expanduser('~/git/gnu/pytorch-liteflownet')
    os.chdir(flow_execute_dir)
    
    root_path=os.path.expanduser('~/Pictures/sy')
    frames=glob.glob(os.path.join(root_path,'*.jpg'))
    frames.sort()
    
    N=len(frames)//2
    for idx in range(N):
        main_path=frames[2*idx]
        aux_path=frames[2*idx+1]
        flow_file=main_path.replace('jpg','flow')
        
        cmd='python run.py --model default --first {} --second {} --out {}'.format(aux_path,main_path,flow_file)
        os.system(cmd)
        print(flow_file)

def test(config_file,filter_relu=False):
    if not os.path.exists(config_file):
        pattern=os.path.expanduser('~/tmp/logs/motion/**/config.txt')
        config_files=glob.glob(pattern,recursive=True)
        config_files=[f for f in config_files if f.find(config_file)>=0]
        assert len(config_files)>0
        config_file=config_files[0]
        print(config_file)
        
    model,_,normer=get_model_and_dataset(config_file,filter_relu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    config=model.config
    config.root_path=os.path.expanduser('~/Pictures/sy')
    dataset=sy_dataset(config,normalizations=normer)
    loader=td.DataLoader(dataset=dataset,batch_size=1,shuffle=False,drop_last=False,num_workers=1)
    tqdm_step = tqdm(loader, desc='steps', leave=False)
    for data_dict in tqdm_step:
        assert isinstance(data_dict,dict),'type is {}'.format(data_dict)
        images = [torch.autograd.Variable(img.to(device).float()) for img in data_dict['images']]
        gt_paths=data_dict['paths']
        assert len(gt_paths)==2
        save_path=gt_paths[0][0].replace('jpg','png')
        assert save_path!=gt_paths[0][0]
        
        outputs=model.forward(images)

        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        save_img=np.squeeze(np.argmax(outputs['masks'][0].data.cpu().numpy(),axis=1)).astype(np.uint8)*255
        cv2.imwrite(save_path,save_img)
        print(save_path)
        
if __name__ == '__main__':
    """
    python xxx/sy.py test config_file
    """
    fire.Fire()