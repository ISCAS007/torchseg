# -*- coding: utf-8 -*-

from hyperopt import fmin,tpe,hp
from bayes_opt import BayesianOptimization as bayesopt
import argparse
import pandas as pd
from utils.torch_tools import keras_fit
from skopt.space import Real,Integer
from skopt import gp_minimize
import torch
from tqdm import tqdm,trange
from time import sleep
import random
import time

class psp_opt():
    def __init__(self,psp_model,config,train_loader,val_loader):
        self.psp_model=psp_model
        self.config=config
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        self.n_calls=config.args.n_calls
        self.current_call=0
    
    def loop(self):
        config=self.config
        train_loader=self.train_loader
        val_loader=self.val_loader
        psp_model=self.psp_model
        self.current_call=0
        results={}
        
        def fn_loop(xyz):
            """
            random for single variale
            """
            config.dataset.norm_ways=xyz
            net = psp_model(config)
            best_val_miou=keras_fit(net, train_loader, val_loader)
            
            cols=['norm_ways','val_miou']
            for col,value in zip(cols,xyz+(best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='loop_%s_%s.tab'%(config.args.note,self.time_str),sep='\t')
            score=best_val_miou
            self.current_call+=1
            print('%s/%s calls, score=%0.3f'%(self.current_call,self.n_calls,score))
            return score
        
        best_score=0
        best_call=0
        for t in range(self.n_calls):
            norm_ways=random.choice(['caffe','pytorch','cityscapes','-1,1','0,1'])
            score=fn_loop(norm_ways)
            if score > best_score:
                best_score=score
                best_call=t
        
        print('best params is'+'*'*30)
        for k,v in results.items():
            print(k,v[best_call])
        print('best score is %0.3f'%best_score+'*'*30)
        
        
    def tpe(self):
        """
        random for servel discrete variable
        """
        config=self.config
        train_loader=self.train_loader
        val_loader=self.val_loader
        psp_model=self.psp_model
        self.current_call=0
        results={}
        def fn_tpe(xyz):
            assert False
#            base_lr,l1_reg,l2_reg,bf,bp,bn=xyz
            norm_ways=xyz
            config.dataset.norm_ways=norm_ways
            net = psp_model(config)
            best_val_miou=keras_fit(net, train_loader, val_loader)
            
            cols=['norm_ways','val_miou']
            for col,value in zip(cols,xyz+(best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='tpe_%s_%s.tab'%(config.args.note,self.time_str),sep='\t')
            score=1-best_val_miou
            self.current_call+=1
            print('%s/%s calls, score=%0.3f'%(self.current_call,self.n_calls,score))
            return score
        
        best = fmin(fn=fn_tpe,
                        space=[hp.uniform('base_lr',0.01,1e-4),
                               hp.uniform('l1_reg',1e-3,1e-7),
                               hp.uniform('l2_reg',1e-3,1e-7),
                               hp.choice('backbone_freeze',[True,False]),
                               hp.choice('backbone_pretrained',[True,False]),
                               hp.choice('backbone_name',['resnet50','resnet101'])],
                        algo=tpe.suggest,
                        max_evals=self.n_calls)
        print('*'*50)
        print(best)
    def bayes(self):
        """
        bayes for serveral continuous variables
        """
        results={}
        self.current_call=0
        def target(base_lr,l1_reg,l2_reg):
            config=self.config
            train_loader=self.train_loader
            val_loader=self.val_loader
            psp_model=self.psp_model
            config.model.use_reg=True
            config.model.learning_rate=base_lr
            config.model.l1_reg=l1_reg
            config.model.l2_reg=l2_reg
            
            net = psp_model(config)
            best_val_miou=keras_fit(net, train_loader, val_loader)
            
            cols=['base_lr','l1_reg','l2_reg','val_miou']
            for col,value in zip(cols,(base_lr,l1_reg,l2_reg,best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='bayes_%s_%s.tab'%(config.args.note,self.time_str),sep='\t')
            self.current_call+=1
            print('%s/%s calls,score=%0.3f'%(self.current_call,self.n_calls,best_val_miou))
            return best_val_miou
        
        bo=bayesopt(target,{'base_lr':[1e-4,0.01],'l1_reg':[1e-7,1e-3],'l2_reg':[1e-7,1e-3]})
        bo.maximize(init_points=5,n_iter=self.n_calls,kappa=2)
        best=bo.res['max']
        
        print('*'*50)
        print(best)
    
    def skopt(self):
        """
        bayes for serveral continuous variables
        """
        results={}
        self.current_call=0
        def target(param):
            base_lr,l1_reg,l2_reg=param
            config=self.config
            train_loader=self.train_loader
            val_loader=self.val_loader
            psp_model=self.psp_model
            config.model.use_reg=True
            config.model.learning_rate=base_lr
            config.model.l1_reg=l1_reg
            config.model.l2_reg=l2_reg
            
            net = psp_model(config)
            best_val_miou=keras_fit(net, train_loader, val_loader)
            
            cols=['base_lr','l1_reg','l2_reg','val_miou']
            for col,value in zip(cols,(base_lr,l1_reg,l2_reg,best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='skopt_%s_%s.tab'%(config.args.note,self.time_str),sep='\t')
            self.current_call+=1
            print('%s/%s calls'%(self.current_call,self.n_calls))
            return 1-best_val_miou
        
        res_gp=gp_minimize(func=target,
                         dimensions=[Real(1e-4,0.01,'log-uniform',name='base_lr'),
                         Real(1e-7,1e-3,'log-uniform',name='l1_reg'),
                         Real(1e-7,1e-3,'log-uniform',name='l2_reg')],
                         n_calls=self.n_calls,
                         random_state=0)
        print('*'*50)
        print("minimize score=%.4f" % res_gp.fun)
        print('minimize param',res_gp.x)
            
if __name__ == '__main__':
    choices = ['fn_demo', 'fn_test','fn_bayes', 'fn_skopt']
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn",
                        help="function to optimizer",
                        choices=choices,
                        default='fn_bayes')

    args = parser.parse_args()
    
    results={}
    def fn_demo(xyz):
        x,y,z=xyz
        if z=='hello':
            score= x+y
        else:
            
            score= x-y
          
        print(x,y,z,score)
        cols={'x','y','z','score'}
        for col,value in zip(cols,xyz+(score,)):
            if col in results.keys():
                results[col].append(value)
            else:
                results[col]=[value]
            
        return score
    
    def fn_test(xyz):
        x,y=xyz
        return x+y
    
    def fn_bayes(x,y,z):
        px=torch.tensor(x,device='cpu',requires_grad=True)
        py=torch.tensor(y,device='cpu',requires_grad=True)
        s=torch.tensor(0.5,device='cpu',requires_grad=True)
        for i in trange(10,leave=False):
            s=s+0.5*px+py
            sleep(0.1)
        return s.data.cpu().numpy()
    
    def fn_skopt(params):
        x,y,z=params

        px=torch.tensor(x,device='cpu',requires_grad=True)
        py=torch.tensor(y,device='cpu',requires_grad=True)
        s=torch.tensor(0.5,device='cpu',requires_grad=True)
        for i in trange(10,leave=False):
            s=s+0.5*px+py
            sleep(0.1)
        return float(s.data.cpu().numpy())

    if args.fn=='fn_demo':
        best = fmin(fn=fn_demo,
                    space=[hp.uniform('x', -10, 10),
                           hp.uniform('y', -10, 10),
                           hp.choice('z',['hello','world'])],
                    algo=tpe.suggest,
                    max_evals=50)
        tasks=pd.DataFrame(results,columns=['x','y','z','score'])
        print(tasks)
    elif args.fn=='fn_test':
        best = fmin(fn=fn_test,
                    space=[hp.uniform('x', -10, 10),
                           hp.uniform('y', -10, 10)],
                    algo=tpe.suggest,
                    max_evals=30)
    elif args.fn=='fn_skopt':
        res_gp=gp_minimize(func=fn_skopt,
                         dimensions=[Real(-10,10,'uniform',name='x'),
                         Real(-10,10,'uniform',name='y'),
                         Integer(-10,10,name='z')],
                         n_calls=15,
                         random_state=0)
        print("Best score=%.4f" % res_gp.fun)
        print('best param',res_gp.x)
        best=res_gp.fun
    else:
        bo=bayesopt(fn_bayes,{'x':[-10,10],'y':[-10,10],'z':[-10,10]})
        bo.maximize(init_points=5,n_iter=10,kappa=2)
        best=bo.res['max']
        print(bo.res['all'])
    print(best)
    
    