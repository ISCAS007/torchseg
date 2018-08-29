# -*- coding: utf-8 -*-

"""
use the code from:
https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py
"""

from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras import backend as K
from keras import layers as KL
import keras

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format='channels_last', **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)
        
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def relu6(x):
    return K.relu(x, max_value=6)

def duc(x, factor=8, output_shape=(224, 224, 1),name=None):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1
    
    if name is None:
        name='duc_%s'%factor
    H, W, c, r = output_shape[0], output_shape[1], output_shape[2], factor
    h = H // r
    w = W // r
    x = KL.Conv2D(
            c*r*r,
            (3, 3),
            padding='same',
            name='conv_%s'%name)(x)
    x = KL.BatchNormalization(axis=bn_axis,name='bn_%s'%name)(x)
    x = KL.Activation('relu')(x)
    x = KL.Permute((3, 1, 2))(x)
    x = KL.Reshape((c, r, r, h, w))(x)
    x = KL.Permute((1, 4, 2, 5, 3))(x)
    x = KL.Reshape((c, H, W))(x)
    x = KL.Permute((2, 3, 1))(x)

    return x

def deconv(base_layers,upsample_layer,class_num):
    # make sure base_layers from big to small
    # eg: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    
    layers=[i+1 for i in range(upsample_layer)]
    layers.reverse()
    for layer in layers:
        if layer == upsample_layer:
            x=base_layers[layer].output
        
        target_filters=max(base_layers[layer-1].output_shape[-1],class_num)
        output_shape=base_layers[layer-1].output_shape[1:-1]
        x=BilinearUpsampling(output_size=output_shape)(x)
        x=KL.Conv2D(filters=target_filters, 
                    kernel_size=(3,3), 
                    strides=(1, 1), 
                    padding='same')(x)
        
    return x

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
            
def unet(base_layers,upsample_layer,class_num,merge_mode):
    layers=[i+1 for i in range(upsample_layer)]
    layers.reverse()
    for layer in layers:
        if layer == upsample_layer:
            x=base_layers[layer].output
        
        target_filters=max(base_layers[layer-1].output_shape[-1],class_num)
        output_shape=base_layers[layer-1].output_shape[1:-1]
        x=BilinearUpsampling(output_size=output_shape)(x)
        x=KL.Conv2D(filters=target_filters, 
                    kernel_size=(3,3),
                    activation=None,
                    strides=(1, 1), 
                    use_bias=False,
                    padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.2)(x)

        x=merge([x,base_layers[layer-1].output],merge_mode)
    
    return x

def Interp(x, shape):
    from keras.backend import tf as ktf
    new_height, new_width = shape
    resized = ktf.image.resize_images(x, [int(new_height), int(new_width)], align_corners=True)
    return resized

# interpolation block
def interp_block(x, feature_map_shape, num_filters=512, level=1, scale=7):
    # compute dataformat
    if K.image_data_format() == 'channels_last':
        bn_axis = -1  
    else:
        bn_axis = 1
    
    kernel = (level*scale, level*scale)
    strides = (level*scale, level*scale)
    
    global_feat = KL.AveragePooling2D(kernel, strides=strides, name='pool_level_%s'%level,padding='valid')(x)
    global_feat = KL.Conv2D(
            filters=num_filters,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            name='conv_level_%s'%level)(global_feat)
    global_feat = KL.BatchNormalization(axis=bn_axis, name='bn_level_%s'%level)(global_feat)
    global_feat = KL.Lambda(Interp, arguments={'shape': feature_map_shape})(global_feat)

    return global_feat

# pyramid pooling function
def pyramid_pooling_module(x,resize_x,num_filters=512, levels=[6, 3, 2, 1], scale=7):
    """
    input channels: c
    psp model output channels: c+num_filters
    final output channels: num_filters
    """
    # compute data format
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
        feature_map_shape=(x.shape[1].value,x.shape[2].value)
    else:
        bn_axis = 1
        feature_map_shape=(x.shape[2].value,x.shape[3].value)
    
    path_out_c_list=[]
    N=len(levels)
    out_c = num_filters//N
    for i in range(N-1):
        path_out_c_list.append(out_c)

    path_out_c_list.append(num_filters+out_c-out_c*N)
        
    pyramid_pooling_blocks = [x]
    for level,out_c in zip(levels,path_out_c_list):
        pyramid_pooling_blocks.append(
            interp_block(
                resize_x,
                feature_map_shape,
                num_filters=out_c,
                level=level,
                scale=scale))

    y = KL.concatenate(pyramid_pooling_blocks)
    #y = merge(pyramid_pooling_blocks, mode='concat', concat_axis=3)
    y = KL.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False,
            name='pyramid_out_%s'%scale)(y)
    y = KL.BatchNormalization(axis=bn_axis, name='bn_pyramid_out_%s'%scale)(y)
    y = KL.Activation('relu')(y)
    
    return y