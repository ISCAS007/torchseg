# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pylab import get_cmap

def AddImpulseNoise(img,noise_density=0.2,noise_type="salt_pepper",rho=0.5):
    if noise_density==0:
        noise_img=img
        mask=(noise_img!=img)
        
    if len(img.shape)<3:
        assert False,'support only rgb image'
    
    width,height,channle=img.shape
    noise_arr=np.random.rand(*img.shape)
    noise_img=img.copy()
    
    if noise_type=="salt_pepper":
        img[noise_arr<noise_density/2]=0
        img[noise_arr>=1-noise_density/2]=255
        mask=(noise_img!=img)
        
        noise_arr2=np.random.rand(*img.shape)
        noise_arr3=np.random.rand(*img.shape)
        # sum(mask)=1 or 2, apply to mask==0
        # equals to sum(mask) > 0, apply to mask==0
        # [h,w,1] can broadcast to [h,w,3]
        noise_area=np.logical_and(mask==0,np.sum(mask,axis=-1,keepdims=True)>0)
        noise_area=np.logical_and(noise_arr2<rho,noise_area)
        noise_255=np.logical_and(noise_area,noise_arr3<0.5)
        noise_0=np.logical_and(noise_area,noise_arr3>=0.5)
        
        mask=np.logical_or(noise_area,mask)
        noise_img[noise_255]=255
        noise_img[noise_0]=0

#        for x in range(width):
#            for y in range(height):
#                noisy=mask[x,y,:]
#                if sum(noisy)==1 or sum(noisy)==2:
#                    for k in range(3):
#                        if noisy[k]==0 and np.random.rand()<rho:
#                            noise_img[x,y,k] = 255 if np.random.rand()<0.5 else 0
#                            mask[x,y,k]=1
                        
    elif noise_type=="random_pulse":
        N=noise_arr.copy()
        N[N>=noise_density]=0
        N1=N.copy()
        #N1 = [x for x in N1 if x>0]
        N1= N1[N1>0]
        n_max=N1.max()
        n_min=N1.min()
        # similar to N/noise_density
        N=((N-n_min)*255/(n_max-n_min)).astype(np.uint8)
        
        
        noise_img[noise_arr<noise_density]=N[noise_arr<noise_density]
        mask=(noise_img!=img)
        
        noise_arr2=np.random.rand(*img.shape)
        noise_arr3=np.random.rand(*img.shape)
        noise_area=np.logical_and(mask==0,np.sum(mask,axis=-1,keepdims=True)>0)
        noise_area=np.logical_and(noise_arr2<rho,noise_area)
        noise_value=(255*noise_arr3).astype(np.uint8)
        noise_img[noise_area]=noise_value[noise_area]
        mask=np.logical_or(noise_area,mask)
#        for x in range(width):
#            for y in range(height):
#                noisy=mask[x,y,:]
#                if sum(noisy)==1 or sum(noisy)==2:
#                    for k in range(3):
#                        if noisy[k]==0 and np.random.rand()<rho:
#                            noise_img[x,y,k] = 255 * np.random.rand()
#                            mask[x,y,k]=1
    else:
        assert False,'unknown noise type %s'%noise_type
        
        
    return noise_img.astype(np.uint8),mask.astype(np.uint8)

if __name__ == '__main__':
    np.random.seed(np.random.seed(int(time.time())))
    img=cv2.imread('image.png')
    noise_img,mask=AddImpulseNoise(img,noise_type='random_pulse')
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('img')
    plt.imshow(img)
    
    
    plt.subplot(1,3,2)
    plt.imshow(noise_img)
    
    plt.subplot(1,3,3)
    plt.imshow(255*mask)
    plt.show()