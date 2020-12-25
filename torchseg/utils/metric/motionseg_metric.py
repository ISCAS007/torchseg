# -*- coding: utf-8 -*-

from .composite import Metric, MetricAcc, MetricMean, CompositeMetric

class MotionSegMetric(Metric):
    def __init__(self,exception_value=0):
        self.metric_acc=MetricAcc(exception_value)
        self.metric_stn_loss=MetricMean()
        self.metric_mask_loss=MetricMean()
        self.metric_total_loss=MetricMean()
        
        self.metrics=CompositeMetric()
        
        self.metrics.add("fmeasure",self.metric_acc)
        self.metrics.add("stn_loss",self.metric_stn_loss)
        self.metrics.add("mask_loss",self.metric_mask_loss)
        self.metrics.add("total_loss",self.metric_total_loss)
        
    def update(self,value):
        self.metrics.update(value)
        
    def reset(self):
        self.metrics.reset()
        
    def fetch(self):
        return self.metrics.fetch()
        
    def write(self,writer,split,epoch):
        acc=self.metric_acc.get_acc()
        precision=self.metric_acc.get_precision()
        recall=self.metric_acc.get_recall()
        fmeasure=self.metric_acc.get_fmeasure()
        avg_p,avg_r,avg_f=self.metric_acc.get_avg_metric()
        mean_stn_loss=self.metric_stn_loss.get_mean()
        mean_mask_loss=self.metric_mask_loss.get_mean()
        mean_total_loss=self.metric_total_loss.get_mean()
        writer.add_scalar(split+'/acc',acc,epoch)
        writer.add_scalar(split+'/precision',precision,epoch)
        writer.add_scalar(split+'/recall',recall,epoch)
        writer.add_scalar(split+'/fmeasure',fmeasure,epoch)
        writer.add_scalar(split+'/avg_p',avg_p,epoch)
        writer.add_scalar(split+'/avg_r',avg_r,epoch)
        writer.add_scalar(split+'/avg_f',avg_f,epoch)
        writer.add_scalar(split+'/stn_loss',mean_stn_loss,epoch)
        writer.add_scalar(split+'/mask_loss',mean_mask_loss,epoch)
        writer.add_scalar(split+'/total_loss',mean_total_loss,epoch)
