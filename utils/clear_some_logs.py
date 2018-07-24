# -*- coding: utf-8 -*-
"""
clear old time logs, but keep the newest.
"""

import os
import glob
import shutil

def clear_some_logs(logdir='/home/yzbx/tmp/logs/pytorch'):
    """
    logdir/psp_caffe/cityscapes/caffe/2018-07-23___17-51-43
    logdir/model_name/dataset_name/note/timestr
    """
    
    dirs_to_note=glob.glob(os.path.join(logdir,'*','*','*'))
    for p in dirs_to_note:
        dirs_to_timestr=glob.glob(os.path.join(p,'*'))
        dirs_to_timestr.sort()
        
        if len(dirs_to_timestr) > 1:
            for d in dirs_to_timestr[:-1]:
                print('remove dir',d)
                shutil.rmtree(d)
            print('keep dir',dirs_to_timestr[-1])
        

if __name__ == '__main__':
    clear_some_logs()