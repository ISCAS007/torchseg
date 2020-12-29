# -*- coding: utf-8 -*-
"""
ImageAugmenter: apply to input image
ImageTransformer: apply to input image and annotations

imgaug: use library img_aug to augment image and mask
pillow: use the library in pillow_transform
semseg: use the library in semseg_transform to augment iamge and mask

todo: add aug_library in {imgaug,pillow,semseg,album} to augment image and mask
reference: 
    - https://github.com/AgaMiko/data-augmentation-review
    - https://github.com/albumentations-team/albumentations
    - https://github.com/aleju/imgaug

"""
from imgaug import augmenters as iaa
import numpy as np
import cv2
import random
import warnings

from easydict import EasyDict as edict
from torchvision import transforms as TT
from .augmentation import pillow_transform as pillow
from .augmentation import semseg_transform as semseg
from .disc_tools import show_images
from .augmentation.custom import get_crop_size,rotate_while_keep_size
from functools import partial


class ImageAugmenter:
    def __init__(self, config):
        propability=config.propability
        self.aug_library = config.aug_library
#        self.print=True        
        # use imgaug to do data augmentation
        if self.aug_library =='imgaug':
            sometimes = lambda aug: iaa.Sometimes(propability, aug)
            blur = iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),
                # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)),
                # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11))
                # blur image using local medians with kernel sizes between 2 and 7
            ])
            noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
            # add gaussian noise to images
            dropout = iaa.Dropout((0.01, 0.1), per_channel=0.5)
            bright = iaa.Add((-10, 10), per_channel=0.5)
            # change brightness of images (by -10 to 10 of original value)
            # randomly remove up to 10% of the pixels
            self.iaa_seq = iaa.Sequential(
                [sometimes(blur),
                 sometimes(noise),
                 sometimes(dropout),
                 sometimes(bright)
                 ],
                random_order=True)
        elif self.aug_library=='pillow':
            # use torchvision transform to do data augmentation
            tt_aug = TT.RandomApply([
                TT.ColorJitter(brightness=10, contrast=0.05, saturation=0.05, hue=0.01),
            ], p=propability)

            self.tt_seq = TT.Compose([
                TT.ToPILImage(),
                tt_aug,
                pillow.ToNumpy(),
            ])
        else:
            assert False,'unsupported augmentation library {}'.format(self.aug_library)

    def augument_image(self, image):
        # use zzl noise
#        use_zzl_noise=True
#        if use_zzl_noise:
#            if self.print:
#                print('use zzl noise: '+'*'*30)
#                self.print=False
#            from utils.aug.zzl_noise import AddImpulseNoise
#            noise_img,mask=AddImpulseNoise(image,noise_density=0.2,noise_type="salt_pepper",rho=0.5)
#            return noise_img
        
        if self.aug_library=='imgaug':
            return self.iaa_seq.augment_image(image)
        elif self.aug_library=='pillow':
            return self.tt_seq(image)
        else:
            assert False,'unsupported augmentation library {}'.format(self.aug_library)

class ImageTransformer(object):
    def __init__(self, config):
        self.config = config
        self.aug_library = config.aug_library

    def transform_image_and_mask_pillow(self, image, mask, angle=None, crop_size=None):
        assert self.aug_library == 'pillow'
        transforms = [pillow.RandomHorizontallyFlip()]
        if crop_size is not None:
            transforms.append(pillow.RandomCrop(size=crop_size))
        if angle is not None:
            transforms.append(pillow.RandomRotate(degree=angle))
        pillow_random = pillow.RandomOrderApply(transforms)

        pillow_transforms = pillow.Compose([
            pillow.ToPILImage(),
            pillow_random,
            pillow.ToNumpy(),
        ])

        return pillow_transforms(image, mask)

    def transform_image_and_mask_imgaug(self, image, mask, angle=None, crop_size=None, hflip=False, vflip=False):
        assert self.aug_library == 'imgaug'
        transforms = []

        if crop_size is not None:
            if self.config.pad_for_crop:
                transforms.append(partial(self.padding_transform,crop_size=crop_size,
                padding_image=[123,116,103],padding_mask=self.config.ignore_index))
            transforms.append(partial(self.crop_transform, crop_size=crop_size))
        if angle is not None:
            transforms.append(partial(self.rotate_transform, rotate_angle=angle))
        if hflip:
            transforms.append(self.horizontal_flip_transform)
        if vflip:
            transforms.append(self.vertical_flip_transform)

        n = len(transforms)

        if n == 0:
            return image, mask
        else:
            order = [i for i in range(n)]
            # np.random.shuffle(order)
            for i in order:
                image, mask = transforms[i](image, mask)
                assert image is not None
                assert mask is not None
            return image, mask

    def transform_image_and_mask(self, image, mask):
        config=self.config

        if self.config.use_rotate:
            a = np.random.rand()
            angle = a * config.rotate_max_angle
        else:
            angle = None

        # image_size = height, width , channel
        image_size=image.shape
        # crop_size <= image_size
        if self.config.use_crop:
            crop_size = get_crop_size(config, image_size)
        else:
            crop_size=None
        
        a = np.random.rand()
        hflip = False
        if config.horizontal_flip and a<0.5:
            hflip = True

        a = np.random.rand()
        vflip = False
        if config.vertical_flip and a<0.5:
            vflip = True

        if config.debug:
            print('angle is',angle)
            print('crop_size is',crop_size)

        if self.aug_library=='imgaug':
            return self.transform_image_and_mask_imgaug(image,
                                                     mask,
                                                     angle=angle,
                                                     crop_size=crop_size,
                                                     hflip=hflip,
                                                     vflip=vflip)
        elif self.aug_library=='pillow':
            return self.transform_image_and_mask_pillow(image,
                                                    mask,
                                                    angle=angle,
                                                    crop_size=crop_size)
        else:
            assert False,'unsupported augmentation library {}'.format(self.aug_library)

    @staticmethod
    def rotate_transform(image, mask, rotate_angle):
        new_image = rotate_while_keep_size(image, rotate_angle, cv2.INTER_CUBIC)
        new_mask = rotate_while_keep_size(mask, rotate_angle, cv2.INTER_NEAREST)
        return new_image, new_mask

    @staticmethod
    def crop_transform(image, mask, crop_size):
        th, tw = crop_size
        h, w = mask.shape
        assert h >= th, 'crop size (%d,%d) should small than image size (%d,%d)' % (th, tw, h, w)
        assert w >= tw, 'crop size (%d,%d) should small than image size (%d,%d)' % (th, tw, h, w)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        # image[y1:y1+th,x1:x1+tw] == image[y1:y1+th,x1:x1+tw,:]
        new_image = image[y1:y1 + th, x1:x1 + tw]
        new_mask = mask[y1:y1 + th, x1:x1 + tw]
        return new_image, new_mask

    @staticmethod
    def horizontal_flip_transform(image, mask):
        new_image = cv2.flip(image, 1)
        new_mask = cv2.flip(mask, 1)
        return new_image, new_mask

    @staticmethod
    def vertical_flip_transform(image, mask):
        new_image = cv2.flip(image, 0)
        new_mask = cv2.flip(mask, 0)
        return new_image, new_mask

    @staticmethod
    def padding_transform(image,mask,crop_size,padding_image=0,padding_mask=0):
        if image.shape[0] >= crop_size[0] and image.shape[1] >=crop_size[1]:
            return image,mask
        else:
            new_image=np.zeros(shape=crop_size,dtype=np.uint8)
            new_image=padding_image
            new_mask=np.zeros(shape=crop_size,dtype=np.uint8)
            new_mask=padding_mask
            new_image[0:image.shape[0],0:image.shape[1],:]=image
            new_mask[0:image.shape[0],0:image.shape[1],:]=mask
            return new_image, new_mask

def get_default_augmentor_config(config=None):
    def get_default_config():
        config = edict()
        config.input_shape=(224,224)
        config.pad_for_crop=True
        config.propability=0.25
        config.use_rotate=True
        config.use_crop=True
        config.rotate_max_angle=15
        config.keep_crop_ratio=True
        config.horizontal_flip = True
        config.vertical_flip = True
        config.debug = False
        config.aug_library="imgaug"
        return config
    
    default_config=get_default_config()
    if config is None:
        config=default_config
    else:
        for key in default_config:
            if not hasattr(config,key):
                config[key]=default_config[key]
            
    if not hasattr(config,'min_crop_size') or config.min_crop_size is None:
        config.min_crop_size=config.input_shape
    elif not isinstance(config.min_crop_size,(tuple,list)):
        assert config.min_crop_size>0
        config.min_crop_size=[config.min_crop_size]*2
    else:
        assert min(config.min_crop_size)>0
        
    if not hasattr(config,'max_crop_size') or config.max_crop_size is None:
        config.max_crop_size=[2*i for i in config.input_shape]
    elif not isinstance(config.max_crop_size,(tuple,list)):
        assert config.max_crop_size>0
        config.max_crop_size=[config.max_crop_size]*2
    else:
        assert min(config.max_crop_size)>0
    
    # image size != network input size != crop size        
    if not config.keep_crop_ratio:
        if not hasattr(config,'crop_size_step') or config.crop_size_step is None:
            config.crop_size_step=0
            
        print('min_crop_size is',config.min_crop_size)
        print('max_crop_size is',config.max_crop_size)
        print('crop_size_step is',config.crop_size_step)
    else:            
        warnings.warn('ignore min_crop_size when keep_crop_ratio is True')        
        if not hasattr(config,'crop_ratio') or config.crop_ratio is None:
            config.crop_ratio = [0.4, 1.0]
        
        print('max_crop_size is',config.max_crop_size)
        print('crop_ratio is',config.crop_ratio)
        # height: image height * random(0.85,1.0)
        # width : image width * random(0.85,1.0)
    
    return config


class Augmentations(object):
    def __init__(self, config=None):
        config = get_default_augmentor_config(config)
        self.aug_library=config.aug_library

        if self.aug_library=='semseg':
            value_scale = 255
            mean = [0.485, 0.456, 0.406]
            mean = [item * value_scale for item in mean]
            std = [0.229, 0.224, 0.225]
            std = [item * value_scale for item in std]
            
            transforms=[semseg.RandScale([0.5, 2.0])]
            
            if config.use_rotate:
                transforms.append(semseg.RandRotate([-10,10], padding=mean, ignore_label=255))
            
            transforms.append(semseg.RandomGaussianBlur())
            transforms.append(semseg.RandomHorizontalFlip())
            if config.use_crop:
                transforms.append(semseg.Crop(config.max_crop_size, crop_type='rand', padding=mean, ignore_label=255))
                
            self.tran = semseg.Compose(transforms)
            
        elif self.aug_library in ['imgaug','pillow']:
            # augmentation for image
            self.aug = ImageAugmenter(config=config)
            # augmentation for image and mask
            self.tran = ImageTransformer(config=config)
        else:
            assert False,'unsupported augmentation library {}'.format(self.aug_library)

    def transform(self, image, mask=None):
        if self.aug_library=='semseg':
            if mask is None:
                return image
            else:
                return self.tran(image,mask)
        elif self.aug_library in ['imgaug','pillow']:
            if mask is None:
                return self.aug.augument_image(image)
            else:
                return self.tran.transform_image_and_mask(image, mask)
        else:
            assert False,'unsupported augmentation library {}'.format(self.aug_library)


if __name__ == '__main__':
    config = get_default_augmentor_config()
    aug = Augmentations(config)
    img = cv2.imread('test/image.png', cv2.IMREAD_COLOR)
    mask = cv2.imread('test/mask.png', cv2.IMREAD_GRAYSCALE)
    #    img=np.random.rand(60,80)
    #    mask=np.random.rand(60,80)

    assert img is not None
    assert mask is not None

    tran_img, tran_mask = aug.transform(img, mask)
    imgs = [cv2.resize(img, (224, 224), interpolation=cv2.INTER_NEAREST) for img in [img, mask, tran_img, tran_mask]]
    show_images(imgs, ['img', 'mask', 'tran_img', 'tran_mask'])
