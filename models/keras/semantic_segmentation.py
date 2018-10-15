# -*- coding: utf-8 -*-
from easydict import EasyDict as edict
import keras
import json
import os
import sys
import time
import glob
import matplotlib.pyplot as plt
import torch
import torch.utils.data as TD
from keras.utils import to_categorical
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

from utils.metrics import runningScore
from dataset.dataset_generalize import dataset_generalize
from utils.config import get_config
from models.keras import metrics_fmeasure
from models.keras.miou import MeanIoU
from models.keras.BackBone import BackBone_Standard

def get_newest_file(files):
    t=0
    newest_file=None
    for full_f in files:
        if os.path.isfile(full_f):
            file_create_time = os.path.getctime(full_f)
            if file_create_time > t:
                t = file_create_time
                newest_file = full_f

    return newest_file

def do_train_or_val(net,args=None,train_loader=None,val_loader=None):
    gpu_config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    gpu_config.gpu_options.allow_growth=True
    session = tf.Session(config=gpu_config)
    KTF.set_session(session)
    session.run(tf.global_variables_initializer())
    
    if args is None:
        args=net.config.training
    metrics = net.get_metrics(net.class_number)
    opt = net.get_optimizer(args)
    net.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=metrics)
    
    if train_loader is None and val_loader is None:
        train_dataset=dataset_generalize(net.config.dataset,split='train',bchw=False)
        train_loader=TD.DataLoader(dataset=train_dataset,batch_size=net.config.dataset.batch_size, shuffle=True,drop_last=False)
        
        val_dataset=dataset_generalize(net.config.dataset,split='val',bchw=False)
        val_loader=TD.DataLoader(dataset=val_dataset,batch_size=net.config.dataset.batch_size, shuffle=True,drop_last=False)
    
    running_metrics = runningScore(net.class_number)
    
    time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
    log_dir=os.path.join(args.log_dir,net.name,args.dataset_name,args.note,time_str)
    checkpoint_path=os.path.join(log_dir,"{}_{}_best_model.pkl".format(net.name, args.dataset_name))
    os.makedirs(log_dir,exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    config=net.config
    config_str = json.dumps(config, indent=2, sort_keys=True).replace(
        '\n', '\n\n').replace('  ', '\t')
    writer.add_text(tag='config', text_string=config_str)

    # write config to config.txt
    config_path = os.path.join(log_dir, 'config.txt')
    config_file = open(config_path, 'w')
    json.dump(config, config_file, sort_keys=True)
    config_file.close()
    
    
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
                y=to_categorical(trues,net.class_number)
                if loader_name=='train':
                    outputs = net.model.train_on_batch(x,y)
                else:
                    outputs = net.model.test_on_batch(x,y)
                    
                predict_outputs = net.model.predict_on_batch(x)
                predicts=np.argmax(predict_outputs,axis=-1)
                
                losses.append(outputs[0])
                if i % 5 == 0:
                    print("%s Epoch [%d/%d] Step [%d/%d]" % (loader_name,epoch+1, args.n_epoch, i, n_step))
                    for name,value in zip(net.model.metrics_names,outputs):
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
                    net.model.save(checkpoint_path)
                
                if epoch % (1+args.n_epoch//10) == 0:
                    print('write image to tensorboard'+'.'*50)
                    idx=np.random.choice(predicts.shape[0])
                    writer.add_image('val/images',x[idx,:,:,:],epoch)
                    writer.add_image('val/predicts', torch.from_numpy(predicts[idx,:,:]), epoch)
                    writer.add_image('val/trues', torch.from_numpy(trues[idx,:,:]), epoch)
                    diff_img=(predicts[idx,:,:]==trues[idx,:,:]).astype(np.uint8)
                    writer.add_image('val/difference', torch.from_numpy(diff_img), epoch)
    
    writer.close()
    
class SS():
    def __init__(self,config):
        self.config=config
    
    @staticmethod
    def get_default_config():
        config=get_config()
        
        return config

    def get_base_model(self):
        h, w = self.config.model['input_shape']

        assert h is not None
        if self.config.model.backbone_type=='standard':
            bb_config=edict()
            bb_config.application=self.config.model.backbone
            h,w=self.config.model.input_shape[0:2]
            bb_config.input_shape=(h,w,3)
            if self.config.model.load_imagenet_weights:
                bb_config.weights='imagenet'
            else:
                bb_config.weights=None
            bb_config.layer_preference=self.config.model.layer_preference
        
            main_backbone = BackBone_Standard(bb_config)
            main_layers = main_backbone.get_layers()
            main_base_model = main_backbone.model
        else:
            print('unknown backbone type',self.config.model.backbone_type)
            assert False
        
        if self.config.model['trainable_ratio']>0:
            ratio=1.0-self.config.model['trainable_ratio']
            min_trainable_idx=int(ratio*len(main_base_model.layers))
            for idx,layer in enumerate(main_base_model.layers):
                layer.trainable=(idx>min_trainable_idx)
                layer.name = 'main_'+layer.name
        else:
            for layer in main_base_model.layers:
                if self.config.model.load_imagenet_weights:
                    layer.trainable = False
                layer.name = 'main_'+layer.name

        return main_base_model,main_layers

    @staticmethod
    def merge(inputs, mode='concatenate'):
        if mode == "add" or mode == 'sum':
            return keras.layers.add(inputs)
        elif mode == "subtract":
            return keras.layers.subtract(inputs)
        elif mode == "multiply":
            return keras.layers.multiply(inputs)
        elif mode == "max":
            return keras.layers.maximum(inputs)
        elif mode == "min":
            return keras.layers.minimum(inputs)
        elif mode in ["concatenate",'concat','concate']:
            return keras.layers.concatenate(inputs)
        else:
            print('warning: unknown merge type %s' % mode)
            assert False

    @staticmethod
    def get_optimizer(config):
        optimizer_str=config['optimizer']
        learning_rate=config['learning_rate']

        if optimizer_str == 'sgd':
            optimizer = keras.optimizers.SGD(lr=learning_rate,
                                             decay=1e-6,
                                             momentum=0.9,
                                             nesterov=True)
        elif optimizer_str == 'adam':
            optimizer = keras.optimizers.Adam(lr=learning_rate,
                                              beta_1=0.9,
                                              beta_2=0.999,
                                              epsilon=1e-08,
                                              decay=0.0)
        else:
            print('maybe unknown optimizer, no learning rate support')
            optimizer = optimizer_str

        return optimizer

    @staticmethod
    def get_metrics(class_num=None):
        if class_num is None:
            return [metrics_fmeasure.precision,
                    metrics_fmeasure.recall,
                    metrics_fmeasure.fmeasure]
        else:
            miou_metric = MeanIoU(class_num)
            return [miou_metric.mean_iou,
                    metrics_fmeasure.precision,
                    metrics_fmeasure.recall,
                    metrics_fmeasure.fmeasure]

    @staticmethod
    def get_callbacks(config):
        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())

        log_dir=os.path.join(config['log_dir'],
                             config['dataset_name'],
                             config['model_name'],
                             config['note'],
                             time_str)

        checkpoint_dir=os.path.join(config['checkpoint_dir'],
                                     config['dataset_name'],
                                     config['model_name'],
                                     config['note'],
                                     time_str)

        os.makedirs(log_dir,exist_ok=True)
        os.makedirs(checkpoint_dir,exist_ok=True)

        # write config to config.txt
        config_path=os.path.join(checkpoint_dir,'config.txt')
        config_file=open(config_path,'w')
        json.dump(config,config_file,sort_keys=True)

        tensorboard_log = keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
        checkpoint = keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    verbose=1,
                    save_best_only=False,
                    save_weights_only=False,
                    mode='auto',
                    period=config.training['epoches']//3)

        return [tensorboard_log,checkpoint]

    def show_version(self):
        print('config is'+'*'*100+'\n')
        print(self.config)
        self.model.summary()

    @staticmethod
    def show_images(images,titles=None):
        fig, axes = plt.subplots(2, (len(images)+1)//2, figsize=(7, 6), sharex=True, sharey=True,
                                 subplot_kw={'adjustable': 'box-forced'})
        ax = axes.ravel()

        for i in range(len(images)):
            ax[i].imshow(images[i])
            if titles is None:
                ax[i].set_title("image %d"%i)
            else:
                ax[i].set_title(titles[i])

        plt.show()

    @staticmethod
    def get_weight_path(checkpoint_path):
        if os.path.isdir(checkpoint_path):
            files = glob.glob(os.path.join(checkpoint_path,'*.hdf5'))
            if len(files)==0:
                files = glob.glob(os.path.join(checkpoint_path,'*','*.hdf5'))
            newest_file = get_newest_file(files)

            if newest_file is None:
                print('no weight file find in path',checkpoint_path)
                return None
            else:
                return newest_file
        elif os.path.isfile(checkpoint_path):
            return checkpoint_path
        else:
            print('checkpoint_path is not a dir or a file!!!')
            sys.exit(-1)

#    @staticmethod
#    def get_default_config():
#        return Config.get_default_config()

if __name__ == '__main__':
    c=SS.get_default_config()
    m=SS(c)
    print('config is'+'*'*100)
    print(c)