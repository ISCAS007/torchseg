# -*- coding: utf-8 -*-

import subprocess
import psutil
from utils.config import get_parser
import time
import pandas as pd
#|    0      1357      C   /usr/bin/python                              157MiB |
#|    0     24038      C   python                                      1863MiB |
#|    0     28136      C   python                                      2875MiB |
#|    1     25081      C   python                                     11713MiB |
#|    2     29140      C   ...yzbx/bin/miniconda3/envs/new/bin/python   203MiB |
#|    2     29268      C   python                                     11521MiB |
#|    3      9469      C   python                                      6473MiB |

def process_line(line):
    contents=line.split(' ')
    valid_con=[]
    for c in contents:
        if c not in [' ','\n','|','']:
            valid_con.append(c)
    if len(valid_con)==5:
#        print(valid_con)
        valid_con[-1]=valid_con[-1].replace('MiB','')
        return valid_con
    else:
        return None

def process_pid(pid):
#    ps_out=subprocess.check_output(["pmap","%d"%pid], shell=True)
#    return ps_out.replace('%d:'%pid,'').strip()
    for process in psutil.process_iter():
        if process.pid == pid:
            return {'cmdline':process.cmdline(),
                    'username':process.username(),
                    'create_time':process.create_time()}
    
    return {}

def process_time(create_time):
    t=time.localtime(create_time)
    return {'tm_year':t.tm_year,
            'tm_mon':t.tm_mon,
            'tm_mday':t.tm_mday,
            'tm_hour':t.tm_hour,
            'tm_min':t.tm_min,
            'tm_sec':t.tm_sec}
    
def process_cmd(parser,cmdline):
    if 'test/pspnet_test.py' in cmdline:
        args=parser.parse_args(cmdline[2:])
        return args.__dict__
    else:
        return {}

nv_out=subprocess.check_output("nvidia-smi", shell=True)
print("nvidia-smi",nv_out)
lines_nv_out=str(nv_out).split('\\n')

tasks=pd.DataFrame()
start=False
parser=get_parser()
for line in lines_nv_out:
    if line.find('PID')>=0:
        start=True
        print('*'*50)
    elif start == True:
        task=process_line(line)
        if task is not None:
            pid=int(task[1])
            gpu=int(task[-1])
            cmd=process_pid(pid)
            dicts={'gpu':gpu,
                   'username':cmd['username'],
                   'train_miou':0.0,
                   'val_miou':0.0,
                   'train_acc':0.0,
                   'val_acc':0.0}
            t=process_time(cmd['create_time'])
            for k,v in t.items():
                dicts[k]=v
            
            args=process_cmd(parser,cmd['cmdline'])
            for k,v in args.items():
                dicts[k]=v
                
            if len(args) > 0:
                dicts['model_file']=cmd['cmdline'][1]
                task= pd.DataFrame(data=[dicts.values()], columns=dicts.keys())
                tasks=tasks.append(task)
                
print(tasks)
