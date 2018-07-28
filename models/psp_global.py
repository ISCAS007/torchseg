# -*- coding: utf-8 -*-

import torch
import torch.nn as TN
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as TD
import random
from dataset.cityscapes import cityscapes
from models.backbone import backbone
from utils.metrics import runningScore
from utils.torch_tools import freeze_layer
from models.upsample import upsample_duc, upsample_bilinear, transform_psp, transform_global
from easydict import EasyDict as edict
import numpy as np
from tensorboardX import SummaryWriter
from utils.augmentor import Augmentations
import time
import os


class psp_global(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__
        self.backbone = backbone(config.model)

        if hasattr(self.config.model, 'backbone_lr_ratio'):
            backbone_lr_raio = self.config.model.backbone_lr_ratio
            if backbone_lr_raio == 0:
                freeze_layer(self.backbone)

        self.upsample_type = self.config.model.upsample_type
        self.upsample_layer = self.config.model.upsample_layer
        self.class_number = self.config.model.class_number
        self.input_shape = self.config.model.input_shape
        self.dataset_name = self.config.dataset.name
#        self.midnet_type = self.config.model.midnet_type
        self.midnet_pool_sizes = self.config.model.midnet_pool_sizes
        self.midnet_scale = self.config.model.midnet_scale

        self.midnet_in_channels = self.backbone.get_feature_map_channel(
            self.upsample_layer)
        self.midnet_out_channels = self.config.model.midnet_out_channels
        self.midnet_out_size = self.backbone.get_feature_map_size(
            self.upsample_layer, self.input_shape[0:2])

        self.midnet = transform_psp(self.midnet_pool_sizes,
                                    self.midnet_scale,
                                    self.midnet_in_channels,
                                    self.midnet_out_channels,
                                    self.midnet_out_size)

        # psp net will output channels with 2*self.midnet_out_channels
        if self.upsample_type == 'duc':
            r = 2**self.upsample_layer
            self.seg_decoder = upsample_duc(
                2*self.midnet_out_channels, self.class_number, r)
        elif self.upsample_type == 'bilinear':
            self.seg_decoder = upsample_bilinear(
                2*self.midnet_out_channels, self.class_number, self.input_shape[0:2])
        else:
            assert False, 'unknown upsample type %s' % self.upsample_type

        self.gnet_dilation_sizes = self.config.model.gnet_dilation_sizes
        self.global_decoder = transform_global(
            self.gnet_dilation_sizes, self.class_number)

    def forward(self, x):
        feature_map = self.backbone.forward(x, self.upsample_layer)
        feature_mid = self.midnet(feature_map)
        seg = self.seg_decoder(feature_mid)
        g_seg = self.global_decoder(seg)
        return seg, g_seg

    def do_train_or_val(self, args, train_loader=None, val_loader=None):
        # use gpu memory
        self.cuda()
        self.backbone.model.cuda()
        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad], lr=0.0001)
#        loss_fn=random.choice([torch.nn.NLLLoss(),torch.nn.CrossEntropyLoss()])
        loss_fn = torch.nn.CrossEntropyLoss()

        # metrics
        running_metrics = runningScore(self.class_number)
        g_running_metrics = runningScore(self.class_number)

        time_str = time.strftime("%Y-%m-%d___%H-%M-%S", time.localtime())
        log_dir = os.path.join(args.log_dir, self.name,
                               self.dataset_name, args.note, time_str)
        checkpoint_path = os.path.join(
            log_dir, "{}_{}_best_model.pkl".format(self.name, self.dataset_name))
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        best_iou = 0.4

        loaders = [train_loader, val_loader]
        loader_names = ['train', 'val']
        for epoch in range(args.n_epoch):
            for loader, loader_name in zip(loaders, loader_names):
                if loader is None:
                    continue

                if loader_name == 'val':
                    if epoch % (1+args.n_epoch//10) == 0:
                        val_image = True
                    else:
                        val_image = False

                    if val_image or epoch % 10 == 0:
                        val_log = True
                    else:
                        val_log = False

                    if not val_log:
                        continue

                    self.eval()
                else:
                    self.train()

                print(loader_name+'.'*50)
                n_step = len(loader)
                losses = []
                gseg_losses = []
                seg_losses = []
                running_metrics.reset()
                g_running_metrics.reset()
                for i, (images, labels) in enumerate(loader):
                    images = Variable(images.cuda().float())
                    labels = Variable(labels.cuda().long())

                    if loader_name == 'train':
                        optimizer.zero_grad()
                    seg_output, gseg_output = self.forward(images)
                    seg_loss = loss_fn(input=seg_output, target=labels)
                    gseg_loss = loss_fn(input=gseg_output, target=labels)
                    loss = 0.9*seg_loss + 0.1*gseg_loss

                    if loader_name == 'train':
                        loss.backward()
                        optimizer.step()

                    losses.append(loss.data.cpu().numpy())
                    seg_losses.append(seg_loss.data.cpu().numpy())
                    gseg_losses.append(gseg_loss.data.cpu().numpy())

                    predicts = seg_output.data.cpu().numpy().argmax(1)
                    g_predicts = gseg_output.data.cpu().numpy().argmax(1)
                    trues = labels.data.cpu().numpy()
                    running_metrics.update(trues, predicts)
                    g_running_metrics.update(trues, g_predicts)
                    score, class_iou = running_metrics.get_scores()

                    if (i+1) % 5 == 0:
                        print("%s, Epoch [%d/%d] Step [%d/%d] Total Loss: %.4f" %
                              (loader_name, epoch+1, args.n_epoch, i, n_step, loss.data))
                        for k, v in score.items():
                            print(k, v)

                g_score, g_class_iou = g_running_metrics.get_scores()
                writer.add_scalar('%s/loss' % loader_name,
                                  np.mean(seg_losses), epoch)
                writer.add_scalar('%s/gseg_loss' %
                                  loader_name, np.mean(gseg_losses), epoch)
                writer.add_scalar('%s/total_loss' %
                                  loader_name, np.mean(losses), epoch)
                writer.add_scalar('%s/acc' % loader_name,
                                  score['Overall Acc: \t'], epoch)
                writer.add_scalar('%s/iou' % loader_name,
                                  score['Mean IoU : \t'], epoch)
                writer.add_scalar('%s/gseg_acc' % loader_name,
                                  g_score['Overall Acc: \t'], epoch)
                writer.add_scalar('%s/gseg_iou' % loader_name,
                                  g_score['Mean IoU : \t'], epoch)

                if loader_name == 'val':
                    if score['Mean IoU : \t'] >= best_iou:
                        best_iou = score['Mean IoU : \t']
                        state = {'epoch': epoch+1,
                                 'miou': best_iou,
                                 'model_state': self.state_dict(),
                                 'optimizer_state': optimizer.state_dict(), }

                        torch.save(state, checkpoint_path)

                    if val_image:
                        print('write image to tensorboard'+'.'*50)
                        idx = np.random.choice(predicts.shape[0])
                        writer.add_image(
                            'val/images', images[idx, :, :, :], epoch)
                        writer.add_image(
                            'val/predicts', torch.from_numpy(predicts[idx, :, :]), epoch)
                        writer.add_image(
                            'val/trues', torch.from_numpy(trues[idx, :, :]), epoch)
                        diff_img = (predicts[idx, :, :] ==
                                    trues[idx, :, :]).astype(np.uint8)
                        writer.add_image('val/difference',
                                         torch.from_numpy(diff_img), epoch)

        writer.close()