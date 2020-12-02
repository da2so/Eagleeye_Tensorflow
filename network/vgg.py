import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, ReLU, AveragePooling2D ,Add, ZeroPadding2D
from tensorflow.keras.layers import Input, Flatten, BatchNormalization, Activation


class VGG(object):
    def build(input_shape, num_classes, cfg, batch_norm):
        rows = input_shape[0]
        cols = input_shape[1]

        img_input = layers.Input(shape=input_shape)

        x=img_input

        for v in cfg:
            if v == 'M':
                x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
            else:
                x = layers.Conv2D(v, (3, 3),
                      padding='same')(x)
                if batch_norm == True:
                    x = layers.BatchNormalization(axis=-1)(x)
                x = Activation("relu")(x)


        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(img_input, x)
        
        return model

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def vgg11(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['A'], batch_norm=False)

def vgg11_bn(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['A'], batch_norm=True)

def vgg13(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['B'], batch_norm=False)

def vgg13_bn(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['B'], batch_norm=True)

def vgg16(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['C'], batch_norm=False)

def vgg16_bn(input_shape, classes):
    return VGG.build(input_shape=input_shape, num_classes=classes, cfg=cfgs['C'], batch_norm=True)





                    