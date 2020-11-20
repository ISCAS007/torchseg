
# coding: utf-8

import pandas as pd
from glob import glob
import numpy as np
from easydict import EasyDict as edict
import os
import time
import sys
from tabulate import tabulate
import argparse
sys.path.insert(0,os.path.expanduser('~/git/torchseg'))
print(sys.path)

from utils.configs.semanticseg_config import load_config
from utils.summary_to_csv import config_to_log,load_log,edict_to_pandas,today_log,recent_log,get_actual_step
import warnings

from utils.summary_to_jupyter import summary,dump

def dump_tasks(notes,
    delete_nan=False,
    dump_group=True,
    dump_dir=True,
    tags=['train/fmeasure','val/fmeasure'],
    rootpath=os.path.expanduser('~/tmp/logs/motion'),
    invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']):

    dump(tags=tags,rootpath=rootpath,notes=notes,
        descripe=['note','epoch'],
             invalid_param_list=invalid_param_list,
        delete_nan=delete_nan,
        dump_group=dump_group,
        dump_dir=dump_dir)


# In[63]:

def dump_recent(
    tags=['train/fmeasure','val/fmeasure'],
    rootpath=os.path.expanduser('~/tmp/logs/motion'),
    notes=['today','recent'],
    dump_dir=True,
    recent_log_number=100,
    note_gtags=[['dataset','log_time','net_name'],
               ['dataset','log_time','net_name']]):
    dump(tags=tags,rootpath=rootpath,notes=notes,
        note_gtags=note_gtags,sort_tags=['dataset','val/fmeasure'],
             descripe=['note','epoch'],delete_nan=False,dump_group=False,
        dump_dir=dump_dir,recent_log_number=recent_log_number)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--delete_nan',
                action='store_true')
    parser.add_argument('--dump_group',
                 action='store_true')
    parser.add_argument('--notes',
                nargs='*')
    parser.add_argument('--ignore_params',
                nargs='*',
                default=[])
    parser.add_argument('--recent_log_number',
                type=int,
                default=100)
    parser.add_argument('--dump_dir',
                action='store_true')
    parser.add_argument('--app',
                default='dump_recent',
                choices=['dump_tasks','dump_recent'])

    parser.add_argument('--tags',
                       choices=['f','iou'],
                       default='f')

    args=parser.parse_args()

    if args.tags=='f':
        tags=['train/fmeasure','val/fmeasure']
        rootpath=os.path.expanduser('~/tmp/logs/motion')
    elif args.tags=='iou':
        tags=['train/iou','val/iou']
        rootpath=os.path.expanduser('~/tmp/logs/motion')

    if args.app=='dump_recent':
        recent_log_number=args.recent_log_number
        dump_recent(dump_dir=args.dump_dir,rootpath=rootpath,tags=tags,recent_log_number=recent_log_number)
    else:
        invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']+args.ignore_params
        print('notes={},delete_nan={},dump_group={}'.format(args.notes,args.delete_nan,args.dump_group))
        dump_tasks(notes=args.notes,
            delete_nan=args.delete_nan,
            dump_group=args.dump_group,
            dump_dir=args.dump_dir,
            invalid_param_list=invalid_param_list,
            tags=tags,
            rootpath=rootpath)
