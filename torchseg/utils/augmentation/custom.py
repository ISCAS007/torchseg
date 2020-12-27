# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2
import math

def get_crop_size(config,image_size,crop_size_list=None):
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
            if crop_size_list is None:
                crop_size_list=[[min_crop_size[0]],
                                [min_crop_size[1]]]
                for i in range(2):
                    while crop_size_list[i][-1]+crop_size_step<max_crop_size[i]:
                        crop_size_list[i].append(crop_size_list[i][-1]+crop_size_step)

            assert len(crop_size_list)==2
            crop_size=[random.choice(crop_size_list[0]),
                      random.choice(crop_size_list[1])]
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
        h, w = image_size[0:2]

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
            
    return crop_size,crop_size_list


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