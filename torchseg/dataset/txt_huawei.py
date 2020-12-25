# -*- coding: utf-8 -*-

"""
ls root_path

*.png
*.json
*.PNG
"""

import os
import glob
import random
import shutil
from tqdm import tqdm

root_path=os.path.expanduser('~/cvdataset/huawei/segmentation')

img_files=glob.glob(os.path.join(root_path,'*.png'))+glob.glob(os.path.join(root_path,'*.PNG'))
json_files=glob.glob(os.path.join(root_path,'*.json'))

other_files=[f for f in os.listdir(root_path) if not f.endswith(('.png','.PNG','.json'))]
if len(other_files)>0 :
   print('other files:',other_files)     

matched_imgs=[]
unmatched_imgs=[]
for img_f in img_files:
    img_f_to_json=img_f.replace(".png", ".json")
    img_f_to_json=img_f_to_json.replace(".PNG",".json")
    if img_f_to_json not in json_files:
        unmatched_imgs.append(img_f)
    else:
        matched_imgs.append(img_f)

if len(unmatched_imgs)>0 :
   print('unmatched_imgs:',unmatched_imgs)
   
unmatched_jsons=[]
for json_f in json_files:
    json_f_to_img=json_f.replace(".json", ".png")
    if json_f_to_img not in img_files:
        unmatched_jsons.append(json_f)

if len(unmatched_jsons)>0 :
   print('unmatched_jsons:',unmatched_jsons)

# assert len(matched_imgs)<10000
# for idx,img in enumerate(tqdm(matched_imgs)):
#     dirname=os.path.dirname(img)
#     target_dir=os.path.join(dirname,'post')
    
#     new_img='huawei%04d'%(idx)+'.png'
#     new_json='huawei%04d'%(idx)+'.json'
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
    
#     if not os.path.exists(os.path.join(target_dir,new_img)):
#         shutil.copy(os.path.join(dirname,img),os.path.join(target_dir,new_img))
#         json_f=img.replace('.png', '.json')
#         shutil.copy(os.path.join(dirname,json_f),os.path.join(target_dir,new_json))
        
splits=['train','val']

random.shuffle(matched_imgs)
train_size=int(len(matched_imgs)*0.7)
for split in splits:
    if split=='train':
        split_imgs=matched_imgs[0:train_size]
    elif split=='val':
        split_imgs=matched_imgs[train_size:]
    elif split=='test':
        split_imgs=unmatched_imgs
    else:
        assert False
    
    write_file=open('dataset/txt/huawei_'+split+'.txt','w')
    
    for g_img in split_imgs:
        img_f=g_img.replace(root_path+'/', '')
        ann_f=img_f.replace('.png', '.json')
        assert os.path.join(root_path,img_f) in img_files,'%s not exist'%img_f
        assert os.path.join(root_path,ann_f) in json_files,'%s not exist'%ann_f
        write_file.write(img_f+' '+ann_f+'\n')
    write_file.close()