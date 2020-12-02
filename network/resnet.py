import numpy as np
import six

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, AveragePooling2D ,Add, ZeroPadding2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

global ROW_AXIS 
global COL_AXIS 
global CHANNEL_AXIS

ROW_AXIS=1
COL_AXIS=2
CHANNEL_AXIS=3

def _bn_relu(input):
    """
     BN -> relu block
    """
    norm = BatchNormalization()(input)
    return Activation("relu")(norm)

def _shortcut(input, residual,init_strides):
    """
    Adds a shortcut between input and residual block and merges them with "sum"
    """

    input_shape = np.shape(input)
    residual_shape = np.shape(residual)
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if init_strides != 1  or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=1,
                          strides=init_strides,
                          padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input)


    out=Add()([shortcut, residual])

    return Activation("relu")(out)

def basic_block(filters, init_strides=1, is_first_block_of_first_layer=False):
    """
    Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    """
    def f(input):

        out = Conv2D(filters=filters, kernel_size=3, strides=init_strides,padding="same",kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input)
        out= _bn_relu(out)

        out = Conv2D(filters=filters, kernel_size=3, strides=1,padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(out)
        residual = BatchNormalization()(out)

        return _shortcut(input, residual,init_strides)

    return f


def bottleneck(filters, init_strides=1, is_first_block_of_first_layer=False):
    """
    Bottleneck architecture for > 34 layer resnet.
    A final conv layer of filters * 4
    """
    def f(input):
        
        out = Conv2D(filters=filters, kernel_size=1, strides=1,padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input)
        out = _bn_relu(out)

        out = Conv2D(filters=filters, kernel_size=3, strides=init_strides,padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(out)
        out = _bn_relu(out)

        out = Conv2D(filters=filters*4, kernel_size=1, strides=1,padding="same", kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(out)
        residual = BatchNormalization()(out)
        
        return _shortcut(input, residual,init_strides)

    return f


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """
    Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
                
            init_strides = 1
            if i == 0 and not is_first_layer:
                init_strides = 2
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResNet(object):
    def build(input_shape, num_outputs, block_fn, repetitions):

        # Load block function
        block_fn = _get_block(block_fn)

        filters_num = 16

        input = Input(shape=input_shape)
        conv1 = Conv2D(filters=filters_num, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(input)
        norm1 = _bn_relu(conv1)
        
        block= norm1
        for i, r in enumerate(repetitions):
                
            block = _residual_block(block_fn, filters=filters_num, repetitions=r, is_first_layer=(i == 0))(block)
            filters_num*= 2

        block= _bn_relu(block)
        block_shape = np.shape(block)
        pool2 = AveragePooling2D(pool_size=block_shape[ROW_AXIS])(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs,kernel_initializer='he_normal')(flatten1)
        out =  Activation("softmax")(dense)
        model = Model(inputs=input, outputs=out)
        return model



def resnet18(input_shape, num_outputs):
    return ResNet.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

def resnet34(input_shape, num_outputs):
    return ResNet.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

def resnet50(input_shape, num_outputs):
    return ResNet.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

def resnet101(input_shape, num_outputs):
    return ResNet.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

def resnet152(input_shape, num_outputs):
    return ResNet.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])