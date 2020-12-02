import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

import network


def load_dataset(dataset_name, batch_size):

    if dataset_name =='cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes=10
        img_shape=[32,32,3]

    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        num_classes=100
        img_shape=[32,32,3]
    elif dataset_name == 'imagenet':
        raise ValueError('Not yet implemented')
    else:
        raise ValueError('Invalid dataset name : {}'.format(dataset_name))
    
    img_shape=[32,32,3]
    normalize = [ [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.
    x_test /= 255.

    mean= normalize[0]
    std= normalize[1]

    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]


    y_train  = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train ,y_train , x_test, y_test,num_classes,img_shape 

def load_model_arch(model_name,img_shape,num_classes):

    try:
        model_func = getattr(network, model_name) 
    except:
        raise ValueError('Invalid model name : {}'.format(model_name))
    
    model = model_func(img_shape,num_classes)
    return model
def lr_scheduler(epoch):
    lr = 1e-3
    if epoch >180:
        lr*=0.001
    elif epoch >120:
        lr*=0.01
    elif epoch >80:
        lr*=0.1

    return lr