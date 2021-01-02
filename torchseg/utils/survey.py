# -*- coding: utf-8 -*-
"""
dataset_survey: 
    survey the image size
    class numbers
dataset_mean, dataset_std:
    compute mean and std for dataset
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
        
class dataset_class_count():
    def __init__(self,class_number):
        self.reset(class_number)
        
    def update(self,image):
        for i in np.unique(image):
            if i<self.class_number:
                self.class_survey[i]+=np.count_nonzero(image==i)
    
    def summary(self):
        return self.class_survey
    
    def reset(self,class_number):
        self.class_number=class_number
        self.class_survey=[0]*class_number
        
class dataset_mean():
    def __init__(self):
        self.reset()
        
    def update(self,image):
        assert len(image.shape)==3 and image.shape[2]==3,"require rgb image with shape [h,w,3]"
        
        self.mean_sum+=np.mean(image,axis=(0,1))
        self.count+=1
        
    def summary(self):
        return self.mean_sum/self.count
    
    def reset(self):
        self.mean_sum=np.zeros(3,np.float64)
        self.count=0
        
class dataset_std():
    def __init__(self,mean,mode='n-1'):
        """
        

        Parameters
        ----------
        mean : TYPE
            DESCRIPTION.
        mode : String, optional
            DESCRIPTION. The default is 'n'.
            support mode: {n,n-1}
        Returns
        -------
        None.

        """
        self.reset(mean,mode)
        
    def update(self,image):
        assert len(image.shape)==3 and image.shape[2]==3,"require rgb image with shape [h,w,3]"
        
        h,w,c=image.shape
        image=image.astype(np.float64)
        for i in range(3):
            image[:,:,i]-=self.mean[i]
        
        self.std_sum+=np.sum(image*image,axis=(0,1))
        self.count+=h*w
        
    def summary(self):
        if self.mode=='n':
            return np.sqrt(self.std_sum/self.count) 
        else:
            return np.sqrt(self.std_sum/(self.count-1))
            
    
    def reset(self,mean,mode):
        self.mean=mean
        self.std_sum=np.zeros(3,np.float64)
        self.count=0
        self.mode=mode