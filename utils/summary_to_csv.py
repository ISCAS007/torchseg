# -*- coding: utf-8 -*-
"""
low level tensorboard logs processing library
"""
from glob import glob
from utils.configs.semanticseg_config import load_config
from easydict import EasyDict as edict
import pandas as pd
import os
import tensorflow as tf
import time

def edict_to_pandas(ed):
    def edict_to_dict(ed):
        d=dict()
        for k,v in ed.items():
            if isinstance(v,edict):
                d_child=edict_to_dict(v)
                d.update(d_child)
#            elif isinstance(v,(list,tuple)):
#                d[k+'_max']=max(v)
            else:
                d[k]=v

        return d

    d=edict_to_dict(ed)
    task= pd.DataFrame(data=[d.values()], columns=d.keys())
    return task

def config_to_log(config_file):
    """
    convert config file to log file
    """
    dirname=os.path.dirname(config_file)
    log_files=glob(os.path.join(dirname,'events.out.*'))

    log_files.sort()
    if len(log_files)>0:
        return log_files[-1]
    else:
        return None

def load_log(log_file,tags=['train/acc','val/acc','train/iou','val/iou']):
    """
    return best metrics
    """
    best_metrics={}
    for e in tf.train.summary_iterator(log_file):
        for v in e.summary.value:
            if v.tag in tags:
                if v.tag not in best_metrics.keys():
                    best_metrics[v.tag]=v.simple_value
                elif v.tag.find('loss')>=0:
                    best_metrics[v.tag]=min(best_metrics[v.tag],v.simple_value)
                else:
                    best_metrics[v.tag]=max(best_metrics[v.tag],v.simple_value)

#    print(best_metrics)
    return best_metrics

def get_actual_step(log_file):
    """
    returna actual step
    """
    actual_step=0
    for e in tf.train.summary_iterator(log_file):
        if hasattr(e,'step'):
            actual_step=e.step

    return actual_step+1

def summary(rootpath):
    config_files=glob(os.path.join(rootpath,'**','config.txt'),recursive=True)
    tasks=pd.DataFrame()
    for cfg in config_files:
        log=config_to_log(cfg)
        if log is not None:
            ed=load_config(cfg)
            metrics=load_log(log)
            for key,value in metrics.items():
                ed[key]=value

            task=edict_to_pandas(ed)
            tasks=tasks.append(task,ignore_index=True)
        else:
            print('cannot find log file for',cfg)

    return tasks

def today_log(log_files):
    today_str=time.strftime('%Y-%m-%d',time.localtime())
    today_log_files=[f for f in log_files if f.find(today_str)>=0]
    return today_log_files

def recent_log(log_files,log_number=100):
    log_files_tuple=[]
    for f in log_files:
        for s in f.split(os.sep):
            if s.find('___')>=0:
                log_files_tuple.append((s,f))
    log_files_tuple.sort()
    log_number=min(log_number,len(log_files_tuple))

    recent_files=[f[1] for f in log_files_tuple[-log_number:]]
    return recent_files

if __name__ == '__main__':
    rootpath=os.path.expanduser('~/tmp/logs/pytorch')
    tasks=summary(rootpath)
    print(tasks.sort_values(by='val/iou'))
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    tab_file=os.path.join(rootpath,time_str+'.tab')
    tasks.to_csv(tab_file,sep='\t')