# -*- coding: utf-8 -*-

import keras
import torch
import keras.layers as KL
import torch.utils.data as TD
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import os
import sys
from tensorboardX import SummaryWriter
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from utils.metrics import runningScore,get_scores
from dataset.dataset_generalize import dataset_generalize,get_dataset_generalize_config
from models.keras.semantic_segmentation import SS


class empty_net(SS):
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__
        self.version = self.get_version(self.name)
        self.class_number = self.config.model.class_number
        self.dataset_name = self.config.dataset.name
        self.model = self.get_model()

    @staticmethod
    def get_default_config():
        config = SS.get_default_config()
        config.model.activation='softmax'
        config.training.log_dir='/home/yzbx/tmp/logs/keras'
        config.dataset.resize_shape=(32,32)
        config.model.input_shape=(32,32)
        config.dataset.batch_size=2
        config.dataset=get_dataset_generalize_config(config.dataset,'Cityscapes')
        return config

    def get_model(self):
        input_size=self.config.model.input_shape.copy()
        if len(input_size)==2:
            input_size.append(3)
        model=Sequential([
            KL.Conv2D(filters=self.class_number,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      activation='softmax',
                     input_shape=input_size)])
        return model

    def train(self,args=None,train_loader=None,val_loader=None):
        if args is None:
            args=self.config.training
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['acc'])
        
        if train_loader is None and val_loader is None:
            train_dataset=dataset_generalize(config.dataset,split='train',bchw=False)
            train_loader=TD.DataLoader(dataset=train_dataset,batch_size=self.config.dataset.batch_size, shuffle=True,drop_last=False)
            
            val_dataset=dataset_generalize(config.dataset,split='val',bchw=False)
            val_loader=TD.DataLoader(dataset=val_dataset,batch_size=self.config.dataset.batch_size, shuffle=True,drop_last=False)
        
        running_metrics = runningScore(self.class_number)
        
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir=os.path.join(args.log_dir,self.name,self.dataset_name,args.note,time_str)
        checkpoint_path=os.path.join(log_dir,"{}_{}_best_model.pkl".format(self.name, self.dataset_name))
        os.makedirs(log_dir,exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        best_iou=0.6
        
        loaders=[train_loader,val_loader]
        loader_names=['train','val']
        for epoch in range(args.n_epoch):
            for loader,loader_name in zip(loaders,loader_names):
                if loader is None:
                    continue
                
                if loader_name == 'val':
                    if epoch % 10 != 0:
                        continue
            
                print(loader_name+'.'*50)
                n_step=len(loader)
                losses=[]
                
                for i, (images, labels) in enumerate(loader):   
                    x=images.data.numpy()
                    trues=labels.data.numpy()
                    y=to_categorical(trues,self.class_number)
                    if loader_name=='train':
                        outputs = self.model.train_on_batch(x,y)
                    else:
                        outputs = self.model.test_on_batch(x,y)
                        
                    predict_outputs = self.model.predict_on_batch(x)
                    predicts=np.argmax(predict_outputs,axis=-1)
                    
                    losses.append(outputs[0])
                    if i % 5 == 0:
                        print("%s Epoch [%d/%d] Step [%d/%d]" % (loader_name,epoch+1, args.n_epoch, i, n_step))
                        for name,value in zip(self.model.metrics_names,outputs):
                            print(name,value)
                        
                        running_metrics.update(trues,predicts)
                        score, class_iou = running_metrics.get_scores()
                        for k, v in score.items():
                            print(k, v)
                        
                        
                writer.add_scalar('%s/loss'%loader_name, np.mean(losses), epoch)
                writer.add_scalar('%s/acc'%loader_name, score['Overall Acc: \t'], epoch)
                writer.add_scalar('%s/iou'%loader_name, score['Mean IoU : \t'], epoch)
                
                running_metrics.reset()
                if loader_name == 'val':
                    if score['Mean IoU : \t'] >= best_iou:
                        best_iou = score['Mean IoU : \t']
                        self.model.save(checkpoint_path)
                    
                    if epoch % (1+args.n_epoch//10) == 0:
                        print('write image to tensorboard'+'.'*50)
                        idx=np.random.choice(predicts.shape[0])
                        writer.add_image('val/images',x[idx,:,:,:],epoch)
                        writer.add_image('val/predicts', torch.from_numpy(predicts[idx,:,:]), epoch)
                        writer.add_image('val/trues', torch.from_numpy(trues[idx,:,:]), epoch)
                        diff_img=(predicts[idx,:,:]==trues[idx,:,:]).astype(np.uint8)
                        writer.add_image('val/difference', torch.from_numpy(diff_img), epoch)
        
        writer.close()
                
if __name__ == '__main__':
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)
    KTF.set_session(session)

    config = empty_net.get_default_config()
    m = empty_net(config)
    m.train()
