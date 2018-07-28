import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
import torchvision
import torchvision.transforms.functional as ttf

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomOrderApply(object):
    def __init__(self,transforms):
        self.transforms=transforms
        
    def __call__(self,img,mask):
        n=len(self.transforms)
        if n==0:
            return img,mask
        else:
            order=[i for i in range(n)]
            np.random.shuffle(order)
            for i in order:
                img,mask=self.transforms[i](img,mask)
                
            return img,mask
    
class ToPILImage():
    def __init__(self):
        pass
    
    def __call__(self, img, mask):
        img=torchvision.transforms.ToPILImage(img)
        mask=torchvision.transforms.ToPILImage(mask)
        return img, mask

class ToTensor():
    def __init__(self):
        pass
        
    def __call__(self,img,mask):
        img=torchvision.transforms.ToTensor(img)
        mask=torchvision.transforms.ToTensor(mask)
        return img, mask

class ToNumpy():
    """convert PILImage, Tensor to Numpy Array"""
    def __init__(self):
        pass
    
    def __call__(self,img,mask=None):
        img=np.array(img, dtype=np.uint8)
        if mask is None:
            return img
        else:
            mask=np.array(mask,dtype=np.uint8)
            return img,mask

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ttf.pad(img, padding=self.padding, fill=0)
            mask = ttf.pad(mask, padding=self.padding, fill=0)
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return ttf.resize(img,(th,tw), Image.BILINEAR), ttf.resize(mask,(th,tw), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return ttf.crop(img,y1,x1,th,tw), ttf.crop(mask,y1,x1,th,tw)


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return ttf.crop(img,y1,x1,th,tw), ttf.crop(mask,y1,x1,th,tw)


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        return ttf.hflip(img), ttf.hflip(mask)

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        return ttf.resize(img,self.size, Image.BILINEAR), ttf.resize(mask,self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return ttf.resize(img,(oh,ow), Image.BILINEAR), ttf.resize(mask,(oh,ow), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return ttf.resize(img,(oh,ow), Image.BILINEAR), ttf.resize(mask,(oh,ow), Image.NEAREST)

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return ttf.rotate(img=img,angle=rotate_degree,resample=Image.BILINEAR), ttf.rotate(img=mask,angle=rotate_degree, resample=Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = ttf.resize(img,(h,w), Image.BILINEAR), ttf.resize(mask,(h,w), Image.NEAREST)

        return self.crop(*self.scale(img, mask))