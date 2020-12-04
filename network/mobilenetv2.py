import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, AveragePooling2D ,Add, DepthwiseConv2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers



def _make_divisible( v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    prefix = 'block_{}_'.format(block_id)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels =  backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs

    # Expand
    if block_id:
        x = Conv2D(expansion * in_channels, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
        x = ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False, padding='same', kernel_initializer="he_normal", depthwise_regularizer=regularizers.l2(4e-5), name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)
    x = ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, kernel_size=1, strides=1, padding='same', use_bias=False, activation=None, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)


    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

class MobileNetV2(object):
    def build(input_shape=None,alpha=1.0, classes=10,**kwargs):
        rows = input_shape[0]
        cols = input_shape[1]
        print(input_shape)
        img_input = layers.Input(shape=input_shape)

        first_block_filters =_make_divisible(32 * alpha, 8)

        x = Conv2D(first_block_filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(4e-5), name='Conv1')(img_input)
        #x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        #x = ReLU(6., name='Conv1_relu')(x)

        x = _inverted_res_block(x, filters=16,  alpha=alpha, stride=1, expansion=1, block_id=0 )

        x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=1 )
        x = _inverted_res_block(x, filters=24,  alpha=alpha, stride=1, expansion=6, block_id=2 )

        x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=2, expansion=6, block_id=3 )
        x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=4 )
        x = _inverted_res_block(x, filters=32,  alpha=alpha, stride=1, expansion=6, block_id=5 )

        x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=2, expansion=6, block_id=6 )
        x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=7 )
        x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=8 )
        x = _inverted_res_block(x, filters=64,  alpha=alpha, stride=1, expansion=6, block_id=9 )
        x = Dropout(rate=0.25)(x)

        x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=10)
        x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=11)
        x = _inverted_res_block(x, filters=96,  alpha=alpha, stride=1, expansion=6, block_id=12)
        x = Dropout(rate=0.25)(x)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)
        x = Dropout(rate=0.25)(x)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)
        x = Dropout(rate=0.25)(x)

        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280

        x = layers.Conv2D(last_block_filters,
                            kernel_size=1,
                            use_bias=False,
                            name='Conv_1')(x)
        x = layers.BatchNormalization( epsilon=1e-3,
                                        momentum=0.999,
                                        name='Conv_1_bn')(x)
        x = layers.ReLU(6., name='out_relu')(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax',
                            use_bias=True, name='Logits')(x)
        # Create model.
        model = Model(img_input, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))


        return model



def mobilenetv2(input_shape, classes):
    return MobileNetV2.build(input_shape=input_shape, classes=classes)