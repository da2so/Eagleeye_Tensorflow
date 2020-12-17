import numpy as np

import tensorflow as tf
import tensorflow.keras as k
import tensorflow.keras.backend as b
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from core.utils import load_network ,count_flops
from network.utils import load_dataset, lr_scheduler
from core.strategy_generation import get_strartegy_generation
from core.pruner.l1norm  import l1_pruning

class EagleEye(object):
    def __init__(self,dataset_name,model_path,bs,epochs,lr,\
        min_rate,max_rate,flops_target,num_candidates,result_dir,data_augmentation):

        self.dataset_name=dataset_name
        self.bs=bs
        self.epochs=epochs
        self.lr=lr
        self.min_rate=min_rate
        self.max_rate=max_rate
        self.flops_target=flops_target
        self.result_dir=result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        self.model_path=model_path
        self.num_candidates=num_candidates
        self.data_augmentation=data_augmentation

        #Load a base model
        self.net=load_network(model_path)

        #Load a dataset
        self.x_train,self.y_train,self.x_test, self.y_test,self.num_classes , self.img_shape = load_dataset(self.dataset_name, self.bs)
    
    def build(self):

        pruned_model_list=list()
        val_acc_list=list()
        for i in range(self.num_candidates):
            # Pruning strategy
            channel_config= get_strartegy_generation(self.net, self.min_rate, self.max_rate) 
            
            #Get pruned model using l1 pruning
            pruned_model= l1_pruning(self.net, channel_config)
            pruned_model_list.append(pruned_model)
            
            # Adpative BN-statistics 
            slice_idx=int(np.shape(self.x_train)[0]/30)
            sliced_x_train=self.x_train[:slice_idx,:,:,:]
            sliced_y_train=self.y_train[:slice_idx,:]

            sliced_train_dataset = tf.data.Dataset.from_tensor_slices((sliced_x_train, sliced_y_train)).batch(64)

            max_iters=10
            for j in range(max_iters):
                for x_batch, y_batch in sliced_train_dataset:
                    output=pruned_model(x_batch,training=True)


            #Evaluate top-1 accuracy for prunned model
            small_val_datsaet= tf.data.Dataset.from_tensor_slices((sliced_x_train,sliced_y_train)).batch(64)
            small_val_acc= k.metrics.CategoricalAccuracy()
            for x_batch, y_batch in small_val_datsaet:
                output=pruned_model(x_batch)

                small_val_acc.update_state(y_batch, output)
            
            small_val_acc=small_val_acc.result().numpy()
            print(f'Adaptive-BN-based accuracy for {i}-th prunned model: {small_val_acc}')
        
            val_acc_list.append(small_val_acc)
        
        #Select the best candidate model
        val_acc_np=np.array(val_acc_list)
        best_candidate_idx=np.argmax(val_acc_np)
        best_model=pruned_model_list[best_candidate_idx]
        print(f'\n The best candidate is {best_candidate_idx}-th prunned model (Acc: {val_acc_np[best_candidate_idx]})')
        
        #Fine tuning
        metrics='accuracy'
        optimizer = Adam(learning_rate=self.lr)
        loss = tf.keras.losses.CategoricalCrossentropy()
        
                
        def get_callback_list(save_path,early_stop=True, lr_reducer=True):
            callback_list=list()
            
            if lr_reducer == True:
                callback_list.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10,mode='auto', epsilon=0.0001, min_lr=0))
            if early_stop == True:
                callback_list.append(tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, verbose=2, mode='auto'))
        
            return callback_list
        

        best_model.compile(loss= loss,optimizer=optimizer,metrics=[metrics])
        self.net.compile(loss= loss,optimizer=optimizer,metrics=[metrics])
        callback_list=get_callback_list(self.result_dir)

        if self.data_augmentation ==True:

            datagen = ImageDataGenerator(   featurewise_center=False,  # set input mean to 0 over the dataset
                                            samplewise_center=False,  # set each sample mean to 0
                                            featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                            samplewise_std_normalization=False,  # divide each input by its std
                                            zca_whitening=False,  # apply ZCA whitening
                                            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                                            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                                            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                                            horizontal_flip=True,  # randomly flip images
                                            vertical_flip=False)  # randomly flip images

            datagen.fit(self.x_train)

            best_model.fit(datagen.flow(self.x_train, self.y_train, batch_size=self.bs), \
                steps_per_epoch=len(self.x_train) / self.bs,epochs=self.epochs, \
                validation_data=(self.x_test,self.y_test), callbacks=callback_list)

        else:
            best_model.fit(self.x_train, self.y_train, batch_size=self.bs, epochs=self.epochs,\
                validation_data=(self.x_test,self.y_test), callbacks=callback_list)
       


        #Get flops and parameters of the base model and pruned model
        params_prev = count_params(self.net.trainable_weights)
        flops_prev = count_flops(self.net)
        scores_prev= self.net.evaluate(self.x_test, self.y_test, batch_size=self.bs,verbose=0)
        print(f'\nTest loss (on base model): {scores_prev[0]}')
        print(f'Test accuracy (on base model): {scores_prev[1]}')
        print(f'The number of parameters (on base model): {params_prev}')
        print(f'The number of flops (on base model): {flops_prev}')
        params_after = count_params(best_model.trainable_weights)
        flops_after = count_flops(best_model)
        scores_after= best_model.evaluate(self.x_test, self.y_test, batch_size=self.bs,verbose=0)
        print(f'\nTest loss (on pruned model): {scores_after[0]}')
        print(f'Test accuracy (on pruned model): {scores_after[1]}')
        print(f'The number of parameters (on pruned model): {params_after}')
        print(f'The number of flops (on prund model): {flops_after}')
        
        #save the best candidate model
        slash_idx=self.model_path.rfind('/')
        ext_idx=self.model_path.rfind('.')
        save_name=self.model_path[slash_idx:ext_idx]
        tf.keras.models.save_model(
                model=best_model, 
                filepath=self.result_dir+save_name+'_pruned.h5',
                include_optimizer=False
                )



