# -*- coding: utf-8 -*-

"""
similar to motion_benchmark, but for the result of other papers
compute f-measure for foreground segmentation like davis
currently support dataset
1. davis2016, davis2017

"""

from models.motionseg.motion_benchmark import get_save_path
from models.motionseg.motion_utils import get_dataset,get_default_config,fine_tune_config
from tqdm import trange
import cv2
import os
import numpy as np
import fire

def evaluation_davis(result_root_path,dataset_name='DAVIS2017',split='val'):
    config=get_default_config()
    config.dataset=dataset_name
    config=fine_tune_config(config)
    dataset=get_dataset(config,split)
    N=len(dataset)

    sum_f=sum_p=sum_r=0
    sum_tp=sum_fp=sum_tn=sum_fn=0
    for idx in trange(N):
        img1_path,img2_path,gt_path=dataset.__get_path__(idx)
        save_path=dataset.get_result_path(result_root_path,img1_path)

        assert os.path.exists(gt_path),gt_path
        assert os.path.exists(save_path),save_path

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
    fire.Fire()
