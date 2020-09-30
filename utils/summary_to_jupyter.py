import pandas as pd
from glob import glob
import numpy as np
import os
from tabulate import tabulate

from utils.configs.semanticseg_config import load_config
from utils.summary_to_csv import config_to_log,load_log,edict_to_pandas,today_log,recent_log,get_actual_step
import warnings

def summary(rootpath,tags,filter_str=None,recent_log_number=100):
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
            actual_step=get_actual_step(log)
            for key,value in metrics.items():
                ed[key]=value

            ed['dir']=cfg
            ed['actual_step']=actual_step
            log_time=''
            for s in cfg.split(os.sep):
                if s.find('___')>=0:
                    log_time=s
                    break
            ed['log_time']=log_time
            if 'dataset_name' not in ed.keys():
                ed['dataset_name']=cfg.split(os.sep)[-4]
                warnings.warn('obtain dataset name from config file path')

            task=edict_to_pandas(ed)
            tasks=tasks.append(task,ignore_index=True,sort=False)
        else:
            warnings.warn('cannot find log file for {}'.format(cfg))

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
        dump_dir=True,
        recent_log_number=100):
    dir_tags=[tags[1],'dir']

    for idx,note in enumerate(notes):
        show_tags=[]
        show_tags=tags.copy()

        tasks=summary(rootpath,tags,note,recent_log_number)
        if tasks.empty:
            warnings.warn('{} task is empty'.format(note))
            continue

        check_ok=True
        for key in show_tags:
            if key not in tasks.columns:
                warnings.warn('key {} not in task'.format(key))
                check_ok=False

        if not check_ok:
            continue

        if note_gtags is None:
            param_list=[]
            for col in tasks.columns:
                if not isinstance(tasks[col][0],(tuple,list)):
                    try:
                        col_set=set(tasks[col])
                        if len(col_set)>1 and (col not in invalid_param_list) and (col not in tags):
                            has_nan=pd.isna(tasks[col]).any()
                            has_null=pd.isnull(tasks[col]).any()
                            if (has_nan or has_null) and len(col_set)<=2:
                                warnings.warn('no group None column {}'.format(col))
                            else:
                                param_list.append(col)
                    except:
                        warnings.warn('exception in task')
                        print(type(tasks[col][0]),col)

            print('###',note,','.join(param_list))
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
                warnings.warn('remove nan log')
                print(os.path.dirname(dir))
                os.system('rm -rf {}'.format(os.path.dirname(dir)))

        #print(tasks[show_tags].groupby(group_tags).max().to_string())
        if dump_group:
            if len(group_tags)==0:
                warnings.warn('no group tags, check tags and invalid_param_list')
                print('tags is',tags)
                print('invalid_param_list is',invalid_param_list)
            else:
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

    return tasks