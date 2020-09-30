# -*- coding: utf-8 -*-
"""
clear old time logs, but keep the newest.
"""

import os
import glob
import shutil
import argparse

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
                    choice=input(prompt='Y(yes, and delete all) \n N(no, and skip the rest) \n n(no, skip this) \n y(yes, delete this): ')
                
                if(choice=='Y' or choice=='y'):
                    shutil.rmtree(d)
                    print('remove dir:'+d+"*"*30)
                    
            print('keep dir',dirs_to_timestr[-1])
        

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--logdir',default=os.path.expanduser('~/tmp/logs/motion'),help='the root directory for log')
    args=parser.parse_args()
    clear_some_logs(args.logdir)