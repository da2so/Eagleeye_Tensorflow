import numpy as np

from tensorflow.keras import layers
import tensorflow as tf

from core.pruner.graph_wrapper import GraphWrapper


def l1_pruning(model, channel_config):
    del_layer_dict={}

    for idx, prune_rate in channel_config.items():
        w_copy=model.layers[idx].get_weights()[0]

        #l1 norm for weights of a layer
        w_copy = tf.math.abs(w_copy)
        w_copy = tf.math.reduce_sum(w_copy, axis=[0,1,2])

        num_delete = int(np.shape(w_copy)[0]*(prune_rate))
        del_list = tf.math.top_k(tf.reshape(-w_copy, [-1]), num_delete)[1].numpy()

        del_layer_dict[idx]=del_list
    
    #Reconstruct the base model uisng graph wrapper
    graph_obj=GraphWrapper(model)
    pruned_model=graph_obj.build(del_layer_dict)
    
    return pruned_model
