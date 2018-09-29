# -*- coding: utf-8 -*-

import fire
import glob
import os
import pandas

class experiment_summary(object):
    """A simple calculator class."""

    def __init__(self, log_dir=os.path.expanduser('~/tmp/logs/pytorch'), **kwargs):
        self.log_dir = log_dir

        for k, v in kwargs.iteritems():
            print("%s = %s" % (k, v))
        
        config_files=glob.glob(os.path.join(self.log_dir,'**/config.txt'),recursive=True)
        
        summarys=[]
        for cfg_file in config_files:
            cfg_dir=os.path.dirname(cfg_file)
            log_files=glob.glob(os.path.join(cfg_dir,'*.dp'))
            
            #use the newest log file
            if len(log_files)>0:
                log_files.sort()
                summarys.append((cfg_file,log_files[-1]))
        
if __name__ == '__main__':
    fire.Fire(experiment_summary)
