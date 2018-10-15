# -*- coding: utf-8 -*-

from dataset.dataset_generalize import dataset_generalize, get_dataset_generalize_config
from models.keras.semantic_segmentation import SS,do_train_or_val
from models.keras import layers
from utils.disc_tools import lcm,lcm_list
from easydict import EasyDict as edict
from keras import backend as K
import keras.layers as KL
import keras
import os

class keras_pspnet(SS):
    def __init__(self,config):
        super().__init__(config)
        self.name=self.__class__.__name__
        self.class_number=self.config.model.class_number
        upsample_type = self.config.model.upsample_type
        upsample_layer = self.config.model.upsample_layer
        output_shape = self.config.model.input_shape
        levels=self.config.model.midnet_pool_sizes
        scale=self.config.model.midnet_scale
        data_format = self.config.model.data_format
        
        if K.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1
        
        base_model, base_layers = self.get_base_model()
        assert len(base_layers) == 6,'unexpected model structure with base layer number of %d'%len(base_layers)
        
        base_layers.reverse()
        # make base_layers from big to small
        # eg: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        
        x = feature = base_layers[upsample_layer].output
        
        filters=x.shape[bn_axis].value
        
        # may change feature map shape
        min_feature_size=lcm_list(levels)*scale
        assert feature.shape[2].value<=min_feature_size,'backbone output size %d should <= duc input size %d'%(x.sahpe[2].value,min_feature_size)
        if feature.shape[2].value<min_feature_size:
            feature=layers.BilinearUpsampling(output_size=(min_feature_size,min_feature_size))(feature)
        
        psp_feature = layers.pyramid_pooling_module(feature,
                                                 feature,
                                                 num_filters=filters,
                                                 levels=levels,
                                                 scale=scale)
        
        if upsample_type == 'duc':    
            x = layers.duc(psp_feature,
                    factor=2**upsample_layer,
                    output_shape=tuple(output_shape)+(self.class_number,))
        elif upsample_type == 'bilinear':
            x = layers.BilinearUpsampling(output_size=output_shape)(psp_feature)
        else:
            print('unknown upsample type', upsample_type)
            assert False
        
        outputs = KL.Conv2D(filters=self.class_number,
                            kernel_size=(3, 3),
                            activation=self.config.model.activation,
                            padding='same',
                            data_format=data_format)(x)
        self.model = keras.models.Model(inputs=base_model.inputs,
                                   outputs=[outputs])

if __name__ == '__main__':
    config=edict()
    config.model=edict()
    config.dataset=edict()
    config.training=edict()
    
    config.dataset=get_dataset_generalize_config(config.dataset,'Cityscapes')
    config.dataset.with_edge=False
    
    config.model.class_number=len(config.dataset.foreground_class_ids)+1
    config.dataset.ignore_index=0
    config.model.upsample_type='bilinear'
    config.model.upsample_layer=3
    config.model.midnet_pool_sizes = [6,3,2,1]
    config.model.midnet_scale = 15
    config.model.trainable_ratio=1.0
    config.model.load_imagenet_weights=True
    config.model.backbone_type='standard'
    config.model.layer_preference='first'
    config.model.data_format='channels_last'
    config.model.backbone='vgg19'
    config.model.merge_type='concat'
    config.model.activation='softmax'
    
    config.training.optimizer='adam'
    config.training.learning_rate=0.01
    config.training.n_epoch=100
    config.training.log_dir=os.path.join(os.getenv('HOME'),'tmp','logs','pytorch')
    config.training.note='keras'
    config.training.dataset_name='cityscapes'
    
    count_size=max(config.model.midnet_pool_sizes)*config.model.midnet_scale*2**config.model.upsample_layer
    input_shape=(count_size,count_size)
    config.model.input_shape=input_shape
    config.dataset.resize_shape=input_shape
    
    batch_size=2
    config.training.batch_size=batch_size
    config.dataset.batch_size=batch_size
    m = keras_pspnet(config)
    m.model.summary()
    do_train_or_val(m)