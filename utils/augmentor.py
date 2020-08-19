# -*- coding: utf-8 -*-
"""
ImageAugmenter,TorchAugmenter: apply to input image
ImageTransformer,TorchTransformer: apply to input image and annotations
"""
from imgaug import augmenters as iaa
import numpy as np
import cv2
import math
import random
from utils.disc_tools import show_images
from easydict import EasyDict as edict
from torchvision import transforms as TT
from utils import joint_transforms as JT
from utils import semseg_transform as ST
from dataset.dataset_generalize import image_normalizations
from functools import partial


class ImageAugmenter:
    def __init__(self, config):
        propability=config.propability
        use_imgaug=config.use_imgaug

        self.use_imgaug = use_imgaug

        # use imgaug to do data augmentation
        if self.use_imgaug:
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
        else:
            # use torchvision transform to do data augmentation
            tt_aug = TT.RandomApply([
                TT.ColorJitter(brightness=10, contrast=0.05, saturation=0.05, hue=0.01),
            ], p=propability)

            self.tt_seq = TT.Compose([
                TT.ToPILImage(),
                tt_aug,
                JT.ToNumpy(),
            ])

    def augument_image(self, image):
        if self.use_imgaug:
            return self.iaa_seq.augment_image(image)
        else:
            return self.tt_seq(image)

class ImageTransformer(object):
    def __init__(self, config):
        self.config = config
        self.use_iaa = config.use_imgaug
        self.crop_size_list=None

    def transform_image_and_mask_tt(self, image, mask, angle=None, crop_size=None):
        assert self.use_iaa == False
        transforms = [JT.RandomHorizontallyFlip()]
        if crop_size is not None:
            transforms.append(JT.RandomCrop(size=crop_size))
        if angle is not None:
            transforms.append(JT.RandomRotate(degree=angle))
        jt_random = JT.RandomOrderApply(transforms)

        jt_transform = JT.Compose([
            JT.ToPILImage(),
            jt_random,
            JT.ToNumpy(),
        ])

        return jt_transform(image, mask)

    def transform_image_and_mask_iaa(self, image, mask, angle=None, crop_size=None, hflip=False, vflip=False):
        assert self.use_iaa == True
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
        crop_size = None
        if not config.keep_crop_ratio:
            # may not keep image height:width ratio
            # make sure crop size <= image size
            min_crop_size = config.min_crop_size
            if not isinstance(min_crop_size,(list,tuple)):
                min_crop_size=[min_crop_size]*2
            if len(min_crop_size)==1:
                min_crop_size=min_crop_size*2

            max_crop_size = config.max_crop_size
            if not isinstance(max_crop_size,(list,tuple)):
                max_crop_size=max_crop_size*2
            if len(max_crop_size)==1:
                max_crop_size=[max_crop_size[0],max_crop_size[0]]

            crop_size_step=config.crop_size_step
            if crop_size_step>0:
                if self.crop_size_list is None:
                    crop_size_list=[[min_crop_size[0]],
                                    [min_crop_size[1]]]
                    for i in range(2):
                        while crop_size_list[i][-1]+crop_size_step<max_crop_size[i]:
                            crop_size_list[i].append(crop_size_list[i][-1]+crop_size_step)
                    self.crop_size_list=crop_size_list

                assert len(self.crop_size_list)==2
                crop_size=[random.choice(self.crop_size_list[0]),
                          random.choice(self.crop_size_list[1])]
            else:
                # just like crop_size_step=1
                a = np.random.rand()
                th = (min_crop_size[0] + (max_crop_size[0] - min_crop_size[0]) * a)
                tw = (min_crop_size[1] + (max_crop_size[1] - min_crop_size[1]) * a)
                crop_size = [int(th), int(tw)]

            # change crop_size to make sure crop_size* <= image_size
            # note in deeplabv3_plus, the author padding the image_size with mean+ignore_label
            # to make sure crop_size <= image_size*
            if crop_size[0] <= image_size[0] and crop_size[1] <= image_size[1]:
                pass
            else:
                y=int(image_size[0]*crop_size[1]/float(crop_size[0]))
                if y<=image_size[1]:
                    crop_size[0]=image_size[0]
                    crop_size[1]=y
                else:
                    crop_size[0]=int(image_size[1]*crop_size[0]/float(crop_size[1]))
                    crop_size[1]=image_size[1]
                    assert crop_size[0]<=image_size[0]
        else:
            # keep image height:width ratio
            # make sure crop size <= image size
            crop_ratio = config.crop_ratio
            h, w = mask.shape

            # use random crop ratio
            if type(crop_ratio) == list or type(crop_ratio) == tuple:
                ratio_max = max(crop_ratio)
                ratio_min = min(crop_ratio)

                assert ratio_max<=1.0
                a = np.random.rand()
                th = (ratio_min + (ratio_max - ratio_min) * a) * h
                tw = (ratio_min + (ratio_max - ratio_min) * a) * w
                crop_size = (int(th), int(tw))
            else:
                assert crop_ratio<=1.0
                crop_size = (int(crop_ratio * h), int(crop_ratio * w))

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

        if self.use_iaa:
            return self.transform_image_and_mask_iaa(image,
                                                     mask,
                                                     angle=angle,
                                                     crop_size=crop_size,
                                                     hflip=hflip,
                                                     vflip=vflip)
        else:
            return self.transform_image_and_mask_tt(image,
                                                    mask,
                                                    angle=angle,
                                                    crop_size=crop_size)

    @staticmethod
    def rotate_while_keep_size(image, rotation, interpolation):
        def rotateImage(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interpolation)
            return result

        # Get size before we rotate
        y, x = image.shape[0:2]
        #        rotation=rotation*2
        #        image_2a=rotateImage(image,rotation)
        image_a = rotateImage(image, rotation / 2)
        #        Y_2a,X_2a=image_2a.shape[0:2]
        Y_a, X_a = image_a.shape[0:2]
        assert y == Y_a and x == X_a, 'rotate image has different size'

        #        tan_angle=math.tan(math.radians(rotation))
        tan_angle_half = math.tan(math.radians(rotation / 2))
        cos_angle = math.cos(math.radians(rotation))
        cos_angle_half = math.cos(math.radians(rotation / 2))

        width_new_float = 2 * (cos_angle_half * (x / 2 - tan_angle_half * y / 2) / cos_angle)
        height_new_float = 2 * (cos_angle_half * (y / 2 - tan_angle_half * x / 2) / cos_angle)

        assert width_new_float > 0, 'half of the angle cannot bigger than arctan(width/height)'
        assert height_new_float > 0, 'half of the angle cannot bigger than arctan(height/width)'
        #        height_new=2*int(cos_angle_half*(x/2-tan_angle_half*y/2)/cos_angle)
        #        width_new=2*int(cos_angle_half*(y/2-tan_angle_half*x/2)/cos_angle)
        #        print('old height is',y)
        #        print('old width is',x)
        #        print('new_height is',height_new_float)
        #        print('new_width is',width_new_float)

        x_new = int(math.ceil((x - width_new_float) / 2))
        y_new = int(math.ceil((y - height_new_float) / 2))
        x_new_end = int(math.floor(width_new_float + (x - width_new_float) / 2))
        y_new_end = int(math.floor(height_new_float + (y - height_new_float) / 2))

        new_image = image_a[y_new:y_new_end, x_new:x_new_end]
        #        print(y,x)
        # Return the image, re-sized to the size of the image passed originally
        return cv2.resize(src=new_image, dsize=(x, y), interpolation=interpolation)

    @staticmethod
    def rotate_transform(image, mask, rotate_angle):
        new_image = ImageTransformer.rotate_while_keep_size(image, rotate_angle, cv2.INTER_CUBIC)
        new_mask = ImageTransformer.rotate_while_keep_size(mask, rotate_angle, cv2.INTER_NEAREST)
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
    if config is None:
        config = edict()

    config.propability=0.25
    config.use_rotate=True
    config.rotate_max_angle=15

    config.keep_crop_ratio=True
    # height:480, width: 480-720
    config.min_crop_size=480
    config.max_crop_size=[480,720]

    # height: image height * random(0.85,1.0)
    # width : image width * random(0.85,1.0)
    config.crop_ratio = [0.85, 1.0]
    config.horizontal_flip = True
    config.vertical_flip = False
    config.debug = False
    config.use_imgaug=True
    return config


class Augmentations(object):
    def __init__(self, config=None):
        if config is None:
            config = get_default_augmentor_config()

        if hasattr(config,'use_semseg'):
            self.use_semseg=config.use_semseg
        else:
            self.use_semseg=False

        if self.use_semseg:
            value_scale = 255
            mean = [0.485, 0.456, 0.406]
            mean = [item * value_scale for item in mean]
            std = [0.229, 0.224, 0.225]
            std = [item * value_scale for item in std]

            self.tran = ST.Compose([
                ST.RandScale([0.5, 2.0]),
                ST.RandRotate([-10,10], padding=mean, ignore_label=255),
                ST.RandomGaussianBlur(),
                ST.RandomHorizontalFlip(),
                ST.Crop(config.input_shape, crop_type='rand', padding=mean, ignore_label=255)])
        else:
            # augmentation for image
            self.aug = ImageAugmenter(config=config)
            # augmentation for image and mask
            self.tran = ImageTransformer(config=config)

    def transform(self, image, mask=None):
        if self.use_semseg:
            if mask is None:
                return image
            else:
                return self.tran(image,mask)
        else:
            if mask is None:
                return self.aug.augument_image(image)
            else:
                return self.tran.transform_image_and_mask(image, mask)


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
