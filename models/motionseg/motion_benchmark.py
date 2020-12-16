# -*- coding: utf-8 -*-
"""
for test mode:
    1. no gt offered
    2. no use_part_number support
for val_path mode:
    1. gt offered
    2. use_part_number support(better optical flow model support)
    3. path support
"""
from utils.configs.motionseg_config import load_config,get_default_config
from utils.notebook import get_model_and_dataset
from models.motionseg.motion_utils import get_parser,get_dataset
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

def get_save_path(gt_path,dataset_root_path,output_root_path):
    save_path=gt_path.replace(dataset_root_path,output_root_path)
    assert save_path!=gt_path,f'cannot overwrite gt path {gt_path} by {save_path}'

    return save_path

def benchmark(config_file,output_root_path='output'):
    if not os.path.exists(config_file):
        pattern=os.path.expanduser('~/tmp/logs/motion/**/config.txt')
        config_files=glob.glob(pattern,recursive=True)
        config_files=[f for f in config_files if f.find(config_file)>=0]
        assert len(config_files)>0
        config_file=config_files[0]
        print(config_file)

    model,dataset_loaders,normer=get_model_and_dataset(config_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    config=model.config
    split='val'
    tqdm_step = tqdm(dataset_loaders[split], desc='steps', leave=False)
    for data_dict in tqdm_step:
        assert isinstance(data_dict,dict),'type is {}'.format(data_dict)
        images = [torch.autograd.Variable(img.to(device).float()) for img in data_dict['images']]
        gt_paths=data_dict['gt_path']
        assert len(gt_paths)==1
        save_path=get_save_path(gt_paths[0],config.root_path,os.path.join(output_root_path,config.dataset,config.note))

        outputs=model.forward(images)
        shape=data_dict['shape']
        origin_mask=F.interpolate(outputs['masks'][0], size=shape[0:2],mode='nearest')
        os.makedirs(os.path.dirname(save_path),exist_ok=True)

        save_img=np.squeeze(np.argmax(origin_mask.data.cpu().numpy(),axis=1)).astype(np.uint8)*255
        cv2.imwrite(save_path,save_img)

def merge_images(images,wgap=5,hgap=5,col_num=9,resize_img_w=48):
    N=len(images)
    max_resize_img_h=0
    h=0
    for idx,img in enumerate(images):
        resize_img_h=int(img.shape[0]*resize_img_w/img.shape[1])
        max_resize_img_h=max(max_resize_img_h,resize_img_h)
        if (idx+1)%col_num==0:
            h+=max_resize_img_h
            max_resize_img_h=0

    merge_img=np.ones((h+int(np.ceil(N/col_num)-1)*hgap,resize_img_w*col_num+(col_num-1)*wgap,3),dtype=np.uint8)*100
    col=0
    y=0
    max_resize_img_h=0
    for img in images:
        x_left=col*resize_img_w+col*wgap
        y_top=y
        resize_img_h=int(img.shape[0]*resize_img_w/img.shape[1])

        resize_img=cv2.resize(img,(resize_img_w,resize_img_h))
    #     print(resize_img.shape,resize_img_h,resize_img_w)
    #     print(x_left,y_top,merge_img.shape)
        merge_img[y_top:y_top+resize_img_h,x_left:x_left+resize_img_w,:]=resize_img
        col=col+1
        max_resize_img_h=max(max_resize_img_h,resize_img_h)
        if col==col_num:
            col=0
            y=y+max_resize_img_h+hgap
            max_resize_img_h=0

    return merge_img

def showcase(config_file,output_root_path='output',generate_results=False,mode='best'):
    """
    run benchmark() first
    """
    if not os.path.exists(config_file):
        pattern=os.path.expanduser('~/tmp/logs/motion/**/config.txt')
        config_files=glob.glob(pattern,recursive=True)
        config_files=[f for f in config_files if f.find(config_file)>=0]
        assert len(config_files)>0
        config_file=config_files[0]
        print(config_file)

    if generate_results:
        benchmark(config_file,output_root_path)
    def get_fmeasure(gt_path,save_path):
        gt_img=cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            assert False,'invalid gt_path: {}'.format(gt_path)
        pred_img=cv2.imread(save_path,cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            assert False,'invalid save_path: {}'.format(save_path)

        tp=np.sum(np.logical_and(gt_img>0,pred_img>0))
#        tn=np.sum(np.logical_and(gt_img==0,pred_img==0))
        fp=np.sum(np.logical_and(gt_img==0,pred_img>0))
        fn=np.sum(np.logical_and(gt_img>0,pred_img==0))

        precision=tp/(tp+fp+1e-5)
        recall=tp/(tp+fn+1e-5)
        fmeasure=2*precision*recall/(precision+recall+1e-5)

        return fmeasure
    config=load_config(config_file)
    default_config=get_default_config()
    for key in default_config.keys():
        if not hasattr(config,key):
            config[key]=default_config[key]
    config=fine_tune_config(config)
    split='val'
    dataset=get_dataset(config,split)
    N=len(dataset)
    category_dict={}
    fmeasure_dict={}
    for idx in trange(N):
        img1_path,img2_path,gt_path=dataset.__get_path__(idx)

        if config.dataset in ['FBMS','FBMS-3D']:
            save_path=get_save_path(gt_path,config.root_path,os.path.join(output_root_path,config.dataset,config.note))
        elif config.dataset.upper() in ['DAVIS2017']:
            save_path=get_save_path(gt_path,
                                    config.root_path,
                                    os.path.join(output_root_path,config.dataset,config.note))
        elif config.dataset.upper() in ['DAVIS2016']:
            save_path=get_save_path(gt_path,
                                    #os.path.join(config.root_path,'Annotations/480p'),
                                    config.root_path,
                                    os.path.join(output_root_path,config.dataset,config.note))
        else:
            assert False

        category=img1_path.split('/')[-2]
        fmeasure=get_fmeasure(gt_path,save_path)
        if category not in category_dict.keys():
            category_dict[category]=(img1_path,save_path,gt_path)
            fmeasure_dict[category]=fmeasure
        elif fmeasure>fmeasure_dict[category] and mode=='best':
            category_dict[category]=(img1_path,save_path,gt_path)
            fmeasure_dict[category]=fmeasure
        elif fmeasure<fmeasure_dict[category] and mode=='worst':
            category_dict[category]=(img1_path,save_path,gt_path)
            fmeasure_dict[category]=fmeasure

    images=[]
    for key,value in category_dict.items():
        print(f"key={key},value={value[2]}")
        for idx,path in enumerate(value):
            img=cv2.imread(path)
            if img is None:
                assert False,'invalid path: {}'.format(path)
            if np.max(img)==1:
                img*=255
            images.append(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    merge_img=merge_images(images,col_num=9,resize_img_w=120)
    save_merge_path=os.path.join(output_root_path,'{}_showcase_{}_{}.jpg'.format(config.dataset,split,config.net_name))
    cv2.imwrite(save_merge_path,merge_img)
    print(fmeasure_dict)
    plt.imshow(merge_img)
    plt.show()

def evaluation(config_file,output_root_path='output',generate_results=False,dataset_name=''):
    """
    run benchmark() first
    """
    if not os.path.exists(config_file):
        pattern=os.path.expanduser('~/tmp/logs/motion/**/config.txt')
        config_files=glob.glob(pattern,recursive=True)
        config_files=[f for f in config_files if f.find(config_file)>=0]
        assert len(config_files)>0
        config_file=config_files[0]
        print(config_file)

    if generate_results:
        benchmark(config_file,output_root_path)
    config=load_config(config_file)
    default_config=get_default_config()
    for key in default_config.keys():
        if not hasattr(config,key):
            config[key]=default_config[key]
    config=fine_tune_config(config)
    split='val'
    if dataset_name!='':
        config.dataset=dataset_name

    dataset=get_dataset(config,split)
    N=len(dataset)

    sum_f=sum_p=sum_r=0
    sum_tp=sum_fp=sum_tn=sum_fn=0
    for idx in trange(N):
        img1_path,img2_path,gt_path=dataset.__get_path__(idx)
        save_path=get_save_path(gt_path,config.root_path,os.path.join(output_root_path,config.dataset,config.note))

        gt_img=cv2.imread(gt_path,cv2.IMREAD_GRAYSCALE)
        pred_img=cv2.imread(save_path,cv2.IMREAD_GRAYSCALE)

        tp=np.sum(np.logical_and(gt_img>0,pred_img>0))
        tn=np.sum(np.logical_and(gt_img==0,pred_img==0))
        fp=np.sum(np.logical_and(gt_img==0,pred_img>0))
        fn=np.sum(np.logical_and(gt_img>0,pred_img==0))

        if tp+fn==0:
            r=1
        else:
            r=tp/(tp+fn)

        if tp+fp==0:
            p=1
        else:
            p=tp/(tp+fp)

        if p+r==0:
            f=1
        else:
            f=2*p*r/(p+r)


        sum_f+=f
        sum_p+=p
        sum_r+=r

        sum_tp+=tp
        sum_fp+=fp
        sum_tn+=tn
        sum_fn+=fn
    overall_precision=sum_tp/(sum_tp+sum_fp+1e-5)
    overall_recall=sum_tp/(sum_tp+sum_fn+1e-5)
    overall_fmeasure=2*overall_precision*overall_recall/(overall_precision+overall_recall+1e-5)

    print('tp={},tn={},fp={},fn={}'.format(sum_tp,sum_tn,sum_fp,sum_fn))
    print('precision={},recall={}'.format(overall_precision,overall_recall))
    print('overall fmeasure is {}'.format(overall_fmeasure))

    print('mean precision={}, recall={}, fmeasure={}'.format(sum_p/N,sum_r/N,sum_f/N))
if __name__ == '__main__':
    """
    # for FBMS
    python xxx/motion_benchmark.py benchmark xxx/xxx.txt
    python xxx/motion_benchmark.py showcase xxx/xxx.txt
    python xxx.py evaluation xxx/xxx.txt

    # for DAVIS2017
    python models/motionseg/motion_benchmark.py showcase ~/tmp/logs/motion/motion_attention/DAVIS2017/test_inn_attd/2020-01-08___15-56-32/config.txt ~/tmp/result/DAVIS2017/val/test_inn_attd
    """
    fire.Fire()