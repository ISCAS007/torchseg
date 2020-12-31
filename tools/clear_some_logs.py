# -*- coding: utf-8 -*-
"""
clear old time logs, but keep the newest.
"""

import os
import glob
import shutil
import argparse
from pprint import pprint
from torchseg.utils.disc_tools import get_newest_file

def clear_some_logs(logdir=os.path.expanduser('~/tmp/logs/pytorch')):
    """
    logdir/psp_caffe/cityscapes/caffe/2018-07-23___17-51-43
    logdir/model_name/dataset_name/note/timestr
    """
    
    choice='y'
    dirs_to_note=glob.glob(os.path.join(logdir,'*','*','*'))
    for p in dirs_to_note:
        dirs_to_timestr=glob.glob(os.path.join(p,'*'))
        dirs_to_timestr.sort()
        
        if len(dirs_to_timestr) > 1:
            for d in dirs_to_timestr[:-1]:
                print('remove dir?',d)
                if(choice=='Y'):
                    pass
                elif(choice=='N'):
                    break
                else:
                    prompt='Y(yes, and delete all) \n N(no, and skip the rest) \n n(no, skip this) \n y(yes, delete this): '
                    print(prompt)
                    choice=input()
                
                if(choice=='Y' or choice=='y'):
                    shutil.rmtree(d)
                    print('remove dir:'+d+"*"*30)
                    
            print('keep dir',dirs_to_timestr[-1])

def clear_some_weight_files(logdir):
    choice='y'
    dirs_to_note=glob.glob(os.path.join(logdir,'*','*','*'))
    for p in dirs_to_note:
        dirs_to_timestr=glob.glob(os.path.join(p,'*'))
        dirs_to_timestr.sort()
        
        for d in dirs_to_timestr:
            weight_files=glob.glob(os.path.join(d,'*.pkl'))
            
            if(len(weight_files)>0):
                if len(weight_files)==1 and choice =='KK':
                    continue
                
                weight_files.sort()
                best_weight_file=get_newest_file(weight_files)
                print('find weight file: \n')
                pprint(weight_files)
                print('best weight file is \n',best_weight_file)
            else:
                continue
            
            print('remove weight files in this dir?',d)
            
            prompt="""
                KK(yes, and delete all weight files in other dirs 
                  except for the last one(best one)) \n 
                YY(yes, and delete all weight files in other dirs) \n 
                NN(no, and skip the weight files in other dirs) \n 
                K(yes, and delete all weight files in this dirs 
                  except for the last one(best one)) \n 
                Y(yes, and delete all weight files in this dirs) \n 
                N(no, and skip the weight files in this dirs) \n 
                k(yes, and delete all except for the last one(best one)) \n 
                n(no, skip this) \n 
                y(yes, delete this): 
                """
            
            
            if choice not in ['KK','YY','NN']:
                print(prompt)
                choice=input()
                
            if(choice=='NN'):
                break
            
            if choice in ['Y','YY']:
                for f in weight_files:
                    os.remove(f)
                print('remove all in ',d)
            elif choice in ['K','KK']:
                for f in weight_files:
                    if f!=best_weight_file:
                        os.remove(f)
                
                print('remove all in ',d)
                print('except for ',best_weight_file)
            elif choice in ['N','NN']:
                continue
            else: 
                raise NotImplementedError
                    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--logdir',default=os.path.expanduser('~/tmp/logs/motion'),help='the root directory for log')
    parser.add_argument('--app',choices=['remove_dir','remove_weight'],default='remove_dir')
    args=parser.parse_args()
    
    if args.app=='remove_dir':
        clear_some_logs(args.logdir)
    else:
        clear_some_weight_files(args.logdir)
