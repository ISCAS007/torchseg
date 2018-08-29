# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import math

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
