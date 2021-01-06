from collections import defaultdict
import sys
import numpy as np
import inspect

from tensorflow.python.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

class GraphWrapper(object):
    def __init__(self,model):
        self.model=model
        self.layer_info=defaultdict()
        self.input_shape= list(model.layers[0].input.shape[1:])
        
        #construct 
        for idx, layer in enumerate(self.model.layers):
            if idx ==0:
                self.layer_info[id(layer.output)]={'count': 0}
                continue
            
            self.layer_info[id(layer.output)]={'layer_input': layer.input, 'count': 0}

            if isinstance(layer.input , list):
                for in_layer in layer.input:
                    self.layer_info[id(in_layer)]['count'] += 1
            else:   
                self.layer_info[id(layer.input)]['count' ]+=1
        
        #check if a conv layer determine the number of classes (ouput classes)
        for idx, layer in enumerate(reversed(self.model.layers)):

            if type(layer) is layers.Dense:
                self.is_lastConv2D=False
                break
            elif type(layer) is layers.Conv2D:
                self.is_lastConv2D=True
                self.lastConv2D=len(self.model.layers)-(idx+1)
                break
            else:
                continue
        

    def build(self,del_layer_dict):
        prev_prune_idx=None
        first_dense=True
        #output processing for keeping the number of classes
        if self.is_lastConv2D:
            del_layer_dict[self.lastConv2D]=[]

        #initialize input
        input=layers.Input(shape=self.input_shape)
        x=input
        
        for idx, layer in enumerate(self.model.layers):
            if idx ==0:
                self.layer_info[id(layer.output)].update({'out':x})
                continue
                
            # reconstruct each layer
            if type(layer) is layers.convolutional.Conv2D:
                prev_prune_idx=self.get_prev_pruned_idx(layer.input, self.layer_info)
                
                pruned_weights = self.get_conv_pruned_weights(layer, del_layer_dict[idx],prev_prune_idx)
                cutted_channl_num= len (del_layer_dict[idx])
                
                attr_name =inspect.getargspec(layers.convolutional.Conv2D.__init__)[0]
                attr_name.pop(0)
                attr_value= {attr: getattr(layer ,attr) for attr in attr_name}
                attr_value['filters']-= cutted_channl_num
                recon_layer = layers.convolutional.Conv2D(**attr_value)

                self.layer_info[id(layer.output)].update({'pruned_idx':del_layer_dict[idx]})
            elif type(layer) is layers.convolutional.DepthwiseConv2D:
                prev_prune_idx=self.get_prev_pruned_idx(layer.input, self.layer_info)
                
                pruned_weights=self.get_depthwiseconv_pruned_weights(layer, prev_prune_idx)
                
                recon_layer = layer.__class__.from_config(layer.get_config())

            
            elif type(layer) is layers.normalization_v2.BatchNormalization:
                prev_prune_idx=self.get_prev_pruned_idx(layer.input, self.layer_info)

                pruned_weights = self.get_batchnorm_pruned_weights(layer, prev_prune_idx)

                cutted_channl_num= len(prev_prune_idx)
                config = layer.get_config()
                config['gamma_regularizer'] = None
                recon_layer = layers.normalization_v2.BatchNormalization.from_config(config)

            
            elif type(layer) is layers.Dense:
                if first_dense ==True:
                    prev_prune_idx=self.get_prev_pruned_idx(layer.input, self.layer_info)

                    pruned_weights = self.get_dense_pruned_weights(layer,prev_prune_idx)
                    

                first_dense =False
                recon_layer = layer.__class__.from_config(layer.get_config())
            elif type(layer) is layers.Reshape:
                prev_prune_idx=self.get_prev_pruned_idx(layer.input, self.layer_info)
                
                prev_prune_num =len( prev_prune_idx)
                
                config = layer.get_config()
                original_shape = config['target_shape']
                original_shape =list(original_shape)
                original_shape[-1]= original_shape[-1]-prev_prune_num
                new_shape= tuple(original_shape)
                config['target_shape']=new_shape

                recon_layer = layer.__class__.from_config(config)
            else:

                if type(layer) is layers.Add or \
                type(layer) is layers.Subtract or \
                type(layer) is layers.Multiply or \
                type(layer) is layers.Average or \
                type(layer) is layers.Maximum or \
                type(layer) is layers.Minimum:
                    pruned_idx=self.set_pruned_idx(layer.input, self.layer_info)
                    self.layer_info[id(layer.output)].update({'pruned_idx':pruned_idx})

                
                if type(layer) is layers.Concatenate:
                    pruned_idx=self.set_pruned_idx_for_concat(layer.input, self.layer_info)
                    self.layer_info[id(layer.output)].update({'pruned_idx':pruned_idx})
                
                recon_layer = layer.__class__.from_config(layer.get_config())

          
            # connect layers
            if isinstance( layer.input, list):
                input_list=[]
                for in_layer in layer.input:
                    input_list.append(self.layer_info[id(in_layer)]['out'])
                    self.layer_info[id(in_layer)]['count']-=1
                    self.del_key(in_layer , self.layer_info)
                x=input_list
            else:
                x=self.layer_info[id(layer.input)]['out']
                self.layer_info[id(layer.input)]['count']-=1
                self.del_key(layer.input , self.layer_info)
            
            x=recon_layer(x)    
            self.layer_info[id(layer.output)]['out']=x
                        
            try:
                recon_layer.set_weights(pruned_weights)   
            except:
                pass
        

        model = Model(inputs=input, outputs=x)
        return model


    def get_conv_pruned_weights(self, layer, prune_idx, prev_pruned_idx=None):
        weights=layer.get_weights()

        #prune kernel
        kernel = weights[0] 
        pruned_kernel= np.delete(kernel, prune_idx, axis=-1)
        
        try:
            pruned_kernel = np.delete(pruned_kernel, prev_pruned_idx, axis=-2)  
        except:
            pass

        #prune bias
        prunned_bias = None
        if layer.use_bias:
            bias = weights[1]
            prunned_bias = np.delete(bias, prune_idx)
        
        if layer.use_bias == True:
            return [pruned_kernel, prunned_bias]
        else:
            return [pruned_kernel]

    def get_dense_pruned_weights(self, layer,prev_pruned_idx=None):
        weights=layer.get_weights()
        kernel = weights[0]
        pruned_kernel= np.delete(kernel, prev_pruned_idx, axis=-2)

        bias= weights[1]

        return [pruned_kernel,bias]


    def get_depthwiseconv_pruned_weights(self, layer , prev_pruned_idx=None):
        weights=layer.get_weights()
        kernel = weights[0]
        pruned_kernel= np.delete(kernel, prev_pruned_idx, axis=-2)

        #prune bias
        prunned_bias = None
        if layer.use_bias:
            bias = weights[1]
        
        if layer.use_bias == True:
            return [pruned_kernel, bias]
        else:
            return [pruned_kernel]

    def get_batchnorm_pruned_weights(self,layer, prune_idx):
        weights = layer.get_weights()
        pruned_weights = [np.delete(w, prune_idx) for w in weights]

        return pruned_weights

    def get_prev_pruned_idx(self, in_layer , layer_info):

        while 1:
            if 'layer_input' not in layer_info[id(in_layer)]:
                return None
            if 'pruned_idx' not in layer_info[id(in_layer)]:
                in_layer=layer_info[id(in_layer)]['layer_input']
            else:
                prev_pruned_idx =layer_info[id(in_layer)]['pruned_idx']
                return prev_pruned_idx
    
    def set_pruned_idx_for_concat(self, layer_list , layer_info):
        pruned_idx_set=set()
        is_second_input=False
        for in_layer in layer_list:
            
            while 1:
                if 'pruned_idx' not in layer_info[id(in_layer)]:
                    in_layer=layer_info[id(in_layer)]['layer_input']
                else:
                    if is_second_input==False:
                        prev_layer_len=in_layer.shape[3]
                        pruned_idx_set.update(layer_info[id(in_layer)]['pruned_idx'])
                        is_second_input=True
                    else:
                        pruned_idx= layer_info[id(in_layer)]['pruned_idx']+prev_layer_len
                        pruned_idx_set.update(pruned_idx)
                        prev_layer_len=in_layer.shape[3]
                    break

        pruned_idx_list=list( pruned_idx_set) 
        return pruned_idx_list
                    

    def set_pruned_idx(self, layer_list , layer_info):
        pruned_idx_set=set()
        for in_layer in layer_list:
            
            while 1:
                if 'pruned_idx' not in layer_info[id(in_layer)]:
                    in_layer=layer_info[id(in_layer)]['layer_input']
                else:
                    pruned_num=len(layer_info[id(in_layer)]['pruned_idx'])
                    pruned_idx_set.update(layer_info[id(in_layer)]['pruned_idx'])
                    break

        pruned_idx_list=list( pruned_idx_set) [:pruned_num]
        return pruned_idx_list
                    
    def del_key(self, layer_input , layer_info):
        if layer_info[id(layer_input)]['count'] ==0:
            del layer_info[id(layer_input)]['out']