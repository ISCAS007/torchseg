
# coding: utf-8

# In[1]:


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

from utils.config import load_config
from utils.summary_to_csv import config_to_log,load_log,edict_to_pandas,today_log,recent_log
recent_log_number=100
def summary(rootpath,tags,filter_str=None,):
    config_files=glob(os.path.join(rootpath,'**','config.txt'),recursive=True)
    if filter_str == 'today':
        config_files=today_log(config_files)
    elif filter_str == 'recent':
        config_files=recent_log(config_files,recent_log_number)
    elif filter_str is not None:
        config_files=[f for f in config_files if f.find(filter_str)>=0]
    
    tasks=pd.DataFrame()
    for cfg in config_files:
        log=config_to_log(cfg)
        if log is not None:
            ed=load_config(cfg)
            metrics=load_log(log,tags)
            for key,value in metrics.items():
                ed[key]=value
            
            ed['dir']=cfg
            log_time=''
            for s in cfg.split(os.sep):
                if s.find('___')>=0:
                    log_time=s
                    break
            ed['log_time']=log_time
            
            task=edict_to_pandas(ed)
            tasks=tasks.append(task,ignore_index=True,sort=False)
        else:
            print('cannot find log file for',cfg)
            
    return tasks

def dump(tags=['train/fmeasure','val/fmeasure'],
        rootpath=os.path.expanduser('~/tmp/logs/motion'),
        notes=['motion_fcn','motion_sparse','motion_psp'],
        note_gtags=None,
        descripe=['note','epoch'],
        sort_tags=[],
        invalid_param_list=['dir','log_time','root_path',
                            'test_path', 'train_path', 'val_path'],
        delete_nan=False,
        dump_group=False,
        dump_dir=True):
    dir_tags=[tags[1],'dir']
    
    for idx,note in enumerate(notes):
        show_tags=[]
        show_tags=tags.copy()
        tasks=summary(rootpath,tags,note)
        if tasks.empty:
            print(note,'task is empty')
            continue
        
        if note_gtags is None:
            param_list=[]
            for col in tasks.columns:
                if not isinstance(tasks[col][0],(tuple,list)):
                    if len(set(tasks[col]))>1 and (col not in invalid_param_list) and (col not in tags):
                        param_list.append(col)

            print(note,','.join(param_list))
            group_tags=param_list
        else:
            if idx<len(note_gtags):
                group_tags=note_gtags[idx]
            else:
                group_tags=[]
            
        show_tags+=group_tags
        show_tags+=descripe
        show_tags=list(set(show_tags))
        
        # remove empty dirs
        if delete_nan:
            dirs=tasks[tasks[tags[1]].isna()]['dir'].tolist()
            tasks=tasks[tasks[tags[1]].notna()]
            for dir in dirs:
                print(os.path.dirname(dir))
                os.system('rm -rf {}'.format(os.path.dirname(dir)))

        #print(tasks[show_tags].groupby(group_tags).max().to_string())
        if dump_group:
            if len(group_tags)==0:
                print(group_tags,param_list,tags,invalid_param_list)
            print(tabulate(tasks[show_tags].groupby(group_tags).mean().sort_values(tags[1]),tablefmt='pipe',headers='keys'))
            print('\n')
            print(tabulate(tasks[[tags[1]]+group_tags].groupby(group_tags).agg([np.mean,np.std,np.max]),tablefmt='pipe',headers='keys'))
            print('\n')
        
        if note=='recent':
            sort_tags=['log_time'] if sort_tags is None or len(sort_tags)==0 else sort_tags
            print(tabulate(tasks[show_tags].sort_values(sort_tags),tablefmt='pipe',headers='keys'))
        else:
            print(tabulate(tasks[show_tags].sort_values(sort_tags+[tags[1]]),tablefmt='pipe',headers='keys'))
        print('\n')
        if dump_dir:
            print(tabulate(tasks[dir_tags].sort_values(tags[1]),tablefmt='pipe',headers='keys'))
            print('\n')


# In[77]:

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
	note_gtags=[['dataset','log_time','net_name'],
          	 ['dataset','log_time','net_name']]):
	dump(tags=tags,rootpath=rootpath,notes=notes,
		note_gtags=note_gtags,sort_tags=['dataset','val/fmeasure'],
     		descripe=['note','epoch'],delete_nan=False,dump_group=False,
		dump_dir=dump_dir)


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

	args=parser.parse_args()
	if args.app=='dump_recent':
		recent_log_number=args.recent_log_number
		dump_recent(dump_dir=args.dump_dir)
	else:
		invalid_param_list=['dir','log_time','root_path',
                    'test_path', 'train_path', 'val_path',
                    'use_optical_flow']+args.ignore_params
		print('notes={},delete_nan={},dump_group={}'.format(args.notes,args.delete_nan,args.dump_group))
		dump_tasks(notes=args.notes,
			delete_nan=args.delete_nan,
			dump_group=args.dump_group,
			dump_dir=args.dump_dir,
			invalid_param_list=invalid_param_list)	
