# -*- coding: utf-8 -*-

from hyperopt import fmin,tpe,hp
from bayes_opt import BayesianOptimization as bayesopt
import argparse
import pandas as pd
from utils.torch_tools import do_train_or_val
import torch
from tqdm import tqdm,trange
from time import sleep

class psp_opt():
    def __init__(self,psp_model,config,train_loader,val_loader):
        self.psp_model=psp_model
        self.config=config
        self.train_loader=train_loader
        self.val_loader=val_loader
    
    def tpe(self):
        config=self.config
        train_loader=self.train_loader
        val_loader=self.val_loader
        psp_model=self.psp_model
        results={}
        def fn_tpe(xyz):
            """
            optimizer should be sgd or adam
            must use reg
            bf=backbone_freeze?
            bp=backbone_pretrained?
            bn=backbone_name?
            """
            config.model.optimizer='sgd'
            config.model.use_reg=True
            assert config.model.optimizer=='sgd','must use sgd for hyperopt here'
            assert config.model.use_reg==True,'must use reg for hyperopt here'
            base_lr,l1_reg,l2_reg,bf,bp,bn=xyz
            config.model.learning_rate=base_lr
            config.model.l1_reg=l1_reg
            config.model.l2_reg=l2_reg
            config.model.backbone_freeze=bf
            config.model.backbone_pretrained=bp
            config.model.backbone_name=bn
            net = psp_model(config)
            best_val_miou=do_train_or_val(net, config.args, train_loader, val_loader)
            
            cols=['base_lr','l1_reg','l2_reg','backbone_freeze','backbone_pretrained','backbone_name','val_miou']
            for col,value in zip(cols,xyz+(best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='hyperopt_%s.tab'%args.note,sep='\t')
            score=1-best_val_miou
            return score
        
        best = fmin(fn=fn_tpe,
                        space=[hp.uniform('base_lr',0.01,1e-4),
                               hp.uniform('l1_reg',1e-3,1e-7),
                               hp.uniform('l2_reg',1e-3,1e-7),
                               hp.choice('backbone_freeze',[True,False]),
                               hp.choice('backbone_pretrained',[True,False]),
                               hp.choice('backbone_name',['resnet50','resnet101'])],
                        algo=tpe.suggest,
                        max_evals=10)
        print('*'*50)
        print(best)
    def bayes(self):
        results={}
        
        def target(base_lr,l1_reg,l2_reg):
            config=self.config
            train_loader=self.train_loader
            val_loader=self.val_loader
            psp_model=self.psp_model
            config.model.learning_rate=base_lr
            config.model.l1_reg=l1_reg
            config.model.l2_reg=l2_reg
            
            net = psp_model(config)
            best_val_miou=do_train_or_val(net, config.args, train_loader, val_loader)
            
            cols=['base_lr','l1_reg','l2_reg','val_miou']
            for col,value in zip(cols,(base_lr,l1_reg,l2_reg,best_val_miou)):
                if col in results.keys():
                    results[col].append(value)
                else:
                    results[col]=[value]
                
            tasks=pd.DataFrame(results,columns=cols)
            tasks.to_csv(path_or_buf='hyperopt_%s.tab'%config.args.note,sep='\t')
            return best_val_miou
        
        bo=bayesopt(target,{'base_lr':[0.01,1e-4],'l1_reg':[1e-3,1e-7],'l2_reg':[1e-3,1e-7]})
        bo.maximize(init_points=1,n_iter=1,kappa=2)
        best=bo.res['max']
        
        print('*'*50)
        print(best)
        
            
if __name__ == '__main__':
    choices = ['fn_demo', 'fn_test','fn_bayes']
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
        return s
    
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
    else:
        bo=bayesopt(fn_bayes,{'x':[-10,10],'y':[-10,10],'z':[-10,10]})
        bo.maximize(init_points=5,n_iter=10,kappa=2)
        best=bo.res['max']
        print(bo.res['all'])
    print(best)
    
    