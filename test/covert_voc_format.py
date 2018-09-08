# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
from PIL import Image
import glob
import os
from tqdm import tqdm

def color_map(N=256, normalized=False):
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

def im2index(im,rgb2idx):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            b, g, r = im[h, w, :]
            m_lable[h, w, :] = rgb2idx[(r, g, b)]
    return np.squeeze(m_lable)

def convert(source_filename,target_filename,cmap,rgb2idx):
    """
    convert a RGB format image to P 
    """
    palette=list(cmap.reshape(-1))
    cv_img_rgb=cv2.imread(source_filename)
    idx_img=im2index(cv_img_rgb,rgb2idx)
#    print(np.unique(idx_img))
    pil_img=Image.fromarray(idx_img,mode='P')
    pil_img.putpalette(palette)
    pil_img.save(target_filename)
    
if __name__ == '__main__':
    src_files=glob.glob('test/comp6_test_cls/*.png')
    target_dir='test/comp6_test_cls_voc'
    des_files=[os.path.join(target_dir,os.path.basename(f)) for f in src_files]
    
    print(src_files[0:3])
    print(des_files[0:3])
    
    cmap=color_map()
    rgb2idx={tuple(v):idx for idx,v in enumerate(cmap)}
    for src,des in tqdm(zip(src_files,des_files)):
        convert(src,des,cmap,rgb2idx)
    