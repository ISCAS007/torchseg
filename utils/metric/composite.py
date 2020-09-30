# -*- coding: utf-8 -*-
"""
design pattern: composite
"""
from abc import ABC, abstractmethod 
import torch
import warnings 

class Metric(ABC):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("you should implement this!")
        
    @abstractmethod
    def update(self,value) -> None:
        raise NotImplementedError("you should implement this!")
        
    @abstractmethod
    def fetch(self):
        raise NotImplementedError("you should implement this!")
        
class MetricAcc(Metric):
    def __init__(self,exception_value=1):
#        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype=torch.int64
#        self.tp=torch.tensor(0,dtype=self.dtype,device=device)
#        self.fp=torch.tensor(0,dtype=self.dtype,device=device)
#        self.tn=torch.tensor(0,dtype=self.dtype,device=device)
#        self.fn=torch.tensor(0,dtype=self.dtype,device=device)
#        self.count=torch.tensor(0,dtype=self.dtype,device=device)
        self.tp=0
        self.fp=0
        self.tn=0
        self.fn=0
        self.count=0

        ## compute avg_p,avg_r,avg_f
        self.sum_p=0
        self.sum_r=0
        self.sum_f=0
        self.img_count=0

        ## when no gt, f=0/1???
        self.exception_value=exception_value

    def update(self,value):
        assert isinstance(value,tuple)
        predicts,labels=value
        # print(labels.shape,predicts.shape)
        if labels.shape != predicts.shape:
            pred=torch.argmax(predicts,dim=1,keepdim=True).type_as(labels)
        else:
            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pred=(predicts>0.5).type_as(labels)

        self.tp+=torch.sum(((pred==1) & (labels==1)).to(self.dtype))
        self.fp+=torch.sum(((pred==1) & (labels==0)).to(self.dtype))
        self.tn+=torch.sum(((pred==0) & (labels==0)).to(self.dtype))
        self.fn+=torch.sum(((pred==0) & (labels==1)).to(self.dtype))

        self.count+=torch.sum(((labels<=1)).to(self.dtype))

        assert self.tp+self.fp+self.tn+self.fn==self.count, \
        'tp={}; fp={}; tn={}; fn={}; count={} \n pred {}, labels {}'.format(self.tp,
            self.fp,self.tn,self.fn,self.count,torch.unique(pred),torch.unique(labels))

        b=pred.size(0)
        for i in range(b):
            tp=torch.sum(((pred[i]==1) & (labels[i]==1)).to(torch.float32))
            fp=torch.sum(((pred[i]==1) & (labels[i]==0)).to(torch.float32))
#            tn=torch.sum(((pred[i]==0) & (labels[i]==0)).to(torch.float32))
            fn=torch.sum(((pred[i]==0) & (labels[i]==1)).to(torch.float32))

            if tp+fn==0:
                warnings.warn('tp+fn==0')
                r=self.exception_value
            else:
                r=tp/(tp+fn)

            if tp+fp==0:
                warnings.warn('tp+fp==0')
                p=self.exception_value
            else:
                p=tp/(tp+fp)

            if p+r==0:
                warnings.warn('p+r==0')
                f=self.exception_value
            else:
                f=2*p*r/(p+r)

            self.sum_p+=p
            self.sum_r+=r
            self.sum_f+=f
        self.img_count+=b

    def get_avg_metric(self):
        return self.sum_p/self.img_count,self.sum_r/self.img_count,self.sum_f/self.img_count

    def get_acc(self):
        return (self.tp+self.tn).to(torch.float32)/(self.count.to(torch.float32)+1e-5)

    def get_precision(self):
        return self.tp.to(torch.float32)/((self.tp+self.fp).to(torch.float32)+1e-5)

    def get_recall(self):
        return self.tp.to(torch.float32)/((self.tp+self.fn).to(torch.float32)+1e-5)

    def get_fmeasure(self):
        p=self.get_precision()
        r=self.get_recall()
        return 2*p*r/(p+r+1e-5)
    
    def fetch(self):
        return self.get_fmeasure()

    def reset(self):
        self.tp=0
        self.fp=0
        self.tn=0
        self.fn=0
        self.count=0

        self.sum_p=0
        self.sum_r=0
        self.sum_f=0
        self.img_count=0


class MetricMean(Metric):
    def __init__(self):
        self.total=0
        self.count=0

    def update(self,value):
        self.total+=value
        self.count+=1.0

    def get_mean(self):
        return self.total/self.count

    def reset(self):
        self.total=0
        self.count=0
        
    def fetch(self):
        return self.get_mean()
        
class CompositeMetric(Metric):
    def __init__(self):
        self.metrics={}
        
    def update(self,value: dict):
        assert len(value)==len(self.metrics)
        for k,v in value.items():
            self.metrics[k].update(v)
            
    def fetch(self):
        ret={}
        for k,v in self.metrics.items():
            ret[k]=v.fetch()
        return ret

    def reset(self):
        for m in self.metrics.values():
            m.reset()
            
    def add(self,k: str, m: Metric) -> None:
        self.metrics[k]=m
    

if __name__ == '__main__':
    a=MetricMean()
    b=MetricMean()
    c=CompositeMetric()
    c.add('acc',a)
    c.add('loss',b)
    
    c.update({"acc":1,"loss":2})
    c.update({"acc":2,"loss":3})
    c.update({"acc":3,"loss":4})
    
    result=c.fetch()
    print(result)
    c.reset()
    c.update({"acc":2,"loss":3})
    c.update({"acc":3,"loss":4})
    c.update({"acc":4,"loss":5})
    result=c.fetch()
    print(result)
    
    a_result=a.fetch()
    print(a_result)