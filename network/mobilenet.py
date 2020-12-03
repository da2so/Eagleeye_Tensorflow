import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, AveragePooling2D ,Add, ZeroPadding2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.ReLU(6., name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

class Mobilenet(object):
    def build(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, pooling=None, classes=10):
        rows = input_shape[0]
        cols = input_shape[1]

        img_input = layers.Input(shape=input_shape)


        x = _conv_block(img_input, 32, alpha, strides=(2, 2))
        x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                                strides=(2, 2), block_id=2)
        x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                                strides=(2, 2), block_id=4)
        x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                                strides=(2, 2), block_id=6)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
        x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                                strides=(2, 2), block_id=12)
        x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)


        shape = (1, 1, int(1024 * alpha))

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(dropout, name='dropout')(x)
        x = layers.Conv2D(classes, (1, 1),
                            padding='same',
                            name='conv_preds')(x)
        x = layers.Reshape((classes,), name='reshape_2')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)


        # Create model.
        model = Model(img_input, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

        return model

def mobilenet(input_shape, classes):
    return Mobilenet.build(input_shape=input_shape, classes=classes)
