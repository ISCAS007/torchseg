# -*- coding: utf-8 -*-

import unittest
import os
from torchseg.dataset.dataset_generalize import image_normalizations
from torchseg.utils.disc_tools import show_images
import cv2
import numpy as np
def print_img_info(img):
    print('{} [{},{}]'.format(img.dtype,np.min(img),np.max(img)))
    
class Test(unittest.TestCase):
    def test_backward(self):
        img_path=os.path.expanduser('~/Pictures/sketch_app.png')
        img=cv2.imread(img_path)
        
        normer=image_normalizations()
        new_img=normer.forward(img)
        origin_img=normer.backward(new_img)
        origin_img=origin_img.astype(np.uint8)
        for x in [img,new_img,origin_img]:
            print_img_info(x)
        show_images([img,new_img,origin_img],['img','new img','origin img'])
        
        self.assertTrue(True)
        
if __name__ == '__main__':
    unittest.main()
    