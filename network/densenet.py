from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, AveragePooling2D ,Add, ZeroPadding2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation


def dense_block(x, filter_num, blocks, name):

    for i in range(blocks):
        x = conv_block(x, filter_num, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      padding='same',
                      name=name + '_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


def conv_block(x, filter_num, name):

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(filter_num, 3,
                       use_bias=False,
                       padding='same',
                       name=name + '_1_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
            filter_num=None,
            input_shape=None,
            classes=1000,
            **kwargs):


    img_input = layers.Input(shape=input_shape)


    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.Conv2D(filter_num, 3, use_bias=False, padding='same',name='conv1/conv')(img_input)

    x = dense_block(x, filter_num, blocks[0], name='conv2')
    x = transition_block(x, 1.0, name='pool2')
    x = dense_block(x, filter_num, blocks[1], name='conv3')
    x = transition_block(x, 1.0, name='pool3')
    x = dense_block(x, filter_num, blocks[2], name='conv4')


    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(classes, activation='softmax', name='fc1000')(x)


    # Create model.
    if blocks == [12, 12, 12] and filter_num == 12:
        model = Model(img_input, x, name='densenet40_f12')
    elif blocks == [32, 32, 32] and filter_num == 12 :
        model = Model(img_input, x, name='densenet100_f12')
    elif blocks == [32, 32, 32] and filter_num == 24:
        model = Model(img_input, x, name='densenet100_f24')
    else:
        model = Model(img_input, x, name='densenet')

    model.summary()
    return model


def densenet40_f12(input_shape=None,classes=10):
    return DenseNet(blocks = [12, 12, 12], 
                    input_shape = input_shape, 
                    filter_num = 12,
                    classes = classes
                    )

def densenet100_f12(input_shape=None,classes=10):
    return DenseNet(blocks = [32, 32, 32], 
                    input_shape = input_shape, 
                    filter_num = 12,
                    classes = classes
                    )
def densenet100_f24(input_shape = None,classes=10):
    return DenseNet(blocks = [32, 32, 32],
                    input_shape = input_shape,
                    filter_num = 24,
                    classes = classes
                    )

