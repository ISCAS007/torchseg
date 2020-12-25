# -*- coding: utf-8 -*-
import torch.nn as TN
from ..backbone import backbone
from ..upsample import get_midnet, get_suffix_net
from ...utils.disc_tools import get_backbone_optimizer_params

class psp_aux(TN.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = self.__class__.__name__

        self.backbone = backbone(config, use_none_layer=config.use_none_layer)
        self.upsample_layer = self.config.upsample_layer
        self.class_number = self.config.class_number
        self.input_shape = self.config.input_shape
        self.dataset_name = self.config.dataset.name
        self.ignore_index = self.config.dataset.ignore_index


        self.auxnet_layer = self.config.auxnet_layer
        assert self.auxnet_layer >= 3
        assert self.upsample_layer >= 3
        assert self.upsample_layer >= self.auxnet_layer
        assert config.use_none_layer==True
        # use modified backbone, the output shape is the same for upsample_layer=[3,4,5]
        self.auxnet_input_shape = self.backbone.get_output_shape(self.auxnet_layer, self.input_shape)
        self.midnet_input_shape = self.backbone.get_output_shape(self.upsample_layer, self.input_shape)

        self.midnet_out_channels = 2*self.midnet_input_shape[1]
        self.auxnet_out_channels = self.auxnet_input_shape[1]

        self.midnet = get_midnet(self.config,
                                 self.midnet_input_shape,
                                 self.midnet_out_channels)

        self.auxnet = get_suffix_net(self.config,
                                     self.auxnet_out_channels,
                                     self.class_number,
                                     aux=True)

        self.decoder = get_suffix_net(self.config,
                                      self.midnet_out_channels,
                                      self.class_number)

        if config.use_lr_mult:
            if config.use_none_layer and config.backbone_pretrained and self.upsample_layer >= 4:
                backbone_optmizer_params = get_backbone_optimizer_params(config.backbone_name,
                                                                         self.backbone.model,
                                                                         unchanged_lr_mult=1,
                                                                         changed_lr_mult=config.changed_lr_mult,
                                                                         new_lr_mult=config.new_lr_mult)
            else:
                backbone_optmizer_params = [{'params': [
                    p for p in self.backbone.parameters() if p.requires_grad], 'lr_mult': 1}]

            self.optimizer_params = backbone_optmizer_params + [{'params': self.midnet.parameters(),
                                                                 'lr_mult':config.new_lr_mult},
                                                                 {'params': self.auxnet.parameters(),
                                                                  'lr_mult':config.new_lr_mult},
                                                                {'params': self.decoder.parameters(),
                                                                 'lr_mult': config.new_lr_mult}]
        else:
            self.optimizer_params = [{'params': [p for p in self.backbone.parameters() if p.requires_grad],
                                                 'lr_mult': 1},
                                 {'params': self.midnet.parameters(), 'lr_mult': 1},
                                 {'params': self.auxnet.parameters(), 'lr_mult': 1},
                                 {'params': self.decoder.parameters(), 'lr_mult': 1}]

        if not hasattr(config, 'aux_base_weight'):
            config.aux_base_weight = 0.4

    def forward(self, x):
        main, aux = self.backbone.forward_aux(
            x, self.upsample_layer, self.auxnet_layer)
#        print('main,aux shape is',main.shape,aux.shape)
        aux = self.auxnet(aux)

        main = self.midnet(main)
        main = self.decoder(main)


        return main, aux