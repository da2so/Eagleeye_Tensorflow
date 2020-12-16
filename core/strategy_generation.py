import numpy as np
import time
from collections import defaultdict

from tensorflow.python.keras import layers

def get_strartegy_generation( model,min_rate, max_rate ):
    np.random.seed(int(time.time()))
    channel_config= defaultdict()
    layer_info= defaultdict()
    
    #Random strategy
    for i, layer in enumerate(model.layers):
        
        layer_info[id(layer.output)]= {'layer_input':layer.input , 'idx':i}
        if isinstance(layer, layers.convolutional.Conv2D):
            random_rate = (max_rate-min_rate)*(np.random.rand(1) ) +min_rate
            layer_info[id(layer.output)].update( {'prune_rate':random_rate})
            channel_config[i]=random_rate
            

    #In case of some architectures (such as resnet)
    for i, layer in enumerate(model.layers):

        if isinstance(layer, layers.Add) or \
                isinstance(layer, layers.Subtract) or \
                isinstance(layer, layers.Multiply) or \
                isinstance(layer, layers.Average) or \
                isinstance(layer, layers.Maximum) or \
                isinstance(layer, layers.Minimum):

            idx_list=list()
            prune_list=list()
            is_specific_layer=-1

            for idx, in_layer in enumerate(layer.input):
                #Find conv layer or specific layers (i.e. Add, Substract, ...)
                while 'prune_rate' not in layer_info[id(in_layer)]:
                    in_layer=layer_info[id(in_layer)]['layer_input']
                
                #Find specific layers that mentioned in abvoe comments
                if layer_info[id(in_layer)]['idx'] not in channel_config:
                    is_specific_layer=idx
                
                idx_list.append(layer_info[id(in_layer)]['idx'])
                prune_list.append(layer_info[id(in_layer)]['prune_rate'])

            #if one of the input layers is a specific layer, change the minimum rate of pruning 
            if is_specific_layer != -1:
                min_prune_rate=prune_list[is_specific_layer]
            else:
                min_prune_rate=min(prune_list)
            
            #Rearrange the prune rate for conv layers
            for idx in idx_list:
                if isinstance(model.layers[idx], layers.convolutional.Conv2D):
                    channel_config[idx]=min_prune_rate
            layer_info[id(layer.output)]['prune_rate']=min_prune_rate

    return channel_config
