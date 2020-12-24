import numpy as np
import glob
import pandas as pd
from tabulate import tabulate
import fire
import os
import cv2

def merge_image(images,wgap=5,hgap=5,col_num=9,resize_img_w=48):
    N=len(images)
    max_resize_img_h=0
    h=0
    for idx,img in enumerate(images):
        resize_img_h=int(img.shape[0]*resize_img_w/img.shape[1])
        max_resize_img_h=max(max_resize_img_h,resize_img_h)
        if (idx+1)%col_num==0:
            h+=max_resize_img_h
            max_resize_img_h=0

    merge_img=np.ones((h+int(np.ceil(N/col_num)-1)*hgap,resize_img_w*col_num+(col_num-1)*wgap,3),dtype=np.uint8)
    #     print('merge_img',merge_img.shape)
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

def motionseg_generate_table(note=''):
    """
    summary the davis benchmark value
    """
    root_path=os.path.expanduser('~/tmp/result/DAVIS2017/val')
    csv_files=glob.glob(os.path.join(root_path,'**','global_results-*.csv'),recursive=True)
    csv_files=[f for f in csv_files if f.find(note)>=0]

    assert len(csv_files)>0

    if len(csv_files)>0:
        tables=[]
        for f in csv_files:
            d=pd.read_csv(f)
            note=f.split(os.sep)[-2]
            d.insert(0,column='note',value=note)
            tables.append(d)

        results=pd.concat(tables)
        print(tabulate(results.sort_values('J&F-Mean'),tablefmt='pipe',headers='keys'))

if __name__ == '__main__':
    fire.Fire()