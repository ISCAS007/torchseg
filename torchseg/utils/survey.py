# -*- coding: utf-8 -*-
"""
survey the image size
class numbers
"""
import numpy as np
from PIL import Image

class dataset_survey():
    def __init__(self,class_number):
        self.class_number=class_number
        self.size_survey={}
        self.class_survey=[0 for i in range(class_number)]
    
    def update_survey(self,label_file):
        label_img_pil=Image.open(label_file)
        label_img = np.array(label_img_pil, dtype=np.uint8)
        size=label_img.shape
        
        if size not in self.size_survey.keys():
            self.size_survey[size]=1
        else:
            self.size_survey[size]+=1
            
        for i in np.unique(label_img):
            if i<self.class_number:
                self.class_survey[i]+=np.count_nonzero(label_img==i)
                
    def update_survey_image(self,label_img):
        for i in np.unique(label_img):
            if i<self.class_number:
                self.class_survey[i]+=np.count_nonzero(label_img==i)
        
    def summary(self):
        print(self.size_survey)
        print(self.class_survey)