#from keras_flops import get_flops

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)
from tensorflow.keras import Sequential, Model

def load_network(model_path):
    if '.h5' in model_path:
        model= tf.keras.models.load_model(model_path)
    else:
        if 'resnet50v2' in model_path:
            model = ResNet50V2(weights='imagenet')
        elif 'mobilenet' in model_path:
            model = MobileNet(weights='imagenet')
        elif 'mobilenetv2' in model_path:
            model = MobileNetV2(weights='imagenet')
        elif 'vgg19' in model_path:
            model = VGG19(weights='imagenet')
        else:
            raise ValueError('Invalid model name : {}'.format(model_path))
    return model


def get_flops(model, batch_size):

    if not isinstance(model, (Sequential, Model)):
        raise KeyError(
            "model arguments must be tf.keras.Model or tf.keras.Sequential instanse"
        )

    if batch_size is None:
        batch_size = 1

    inputs = [
        tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs
    ]
    real_model = tf.function(model).get_concrete_function(inputs)
    frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

    # Calculate FLOPS with tf.profiler
    run_meta = tf.compat.v1.RunMetadata()
    opts = (tf.compat.v1.profiler.ProfileOptionBuilder()
                                .with_empty_output()
                                .build())
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
    )

    return flops.total_float_ops

def count_flops(model):
    return get_flops(model, batch_size=1)