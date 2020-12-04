import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import argparse

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from network.utils import load_dataset, load_model_arch, lr_scheduler
class TrainTeacher(object):
    def __init__(self,dataset_name, model_name, batch_size, epochs, save_dir ,data_augmentation,metrics='accuracy'):
        self.dataset_name=dataset_name
        self.batch_size=batch_size
        self.model_name=model_name
        self.data_augmentation=data_augmentation
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path=f'{save_dir}{self.dataset_name}_{self.model_name}.h5'
        self.epochs=epochs
        self.metrics=metrics
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        
        
        #Load train and validation dataset
        self.x_train,self.y_train,self.x_test, self.y_test,self.num_classes , self.img_shape = load_dataset(self.dataset_name, self.batch_size)

        #Load model architecture
        self.model=load_model_arch(self.model_name,self.img_shape,self.num_classes)
    

    #Define callback function 
    def get_callback_list(self,save_path,early_stop=True ,lr_reducer=True):

        callback_list=list()

        callback_list.append(tf.keras.callbacks.ModelCheckpoint(filepath=save_path, save_best_only=True, period=1))
        
        if early_stop== True:
            callback_list.append(tf.keras.callbacks.EarlyStopping(min_delta=0, patience=20, verbose=2, mode='auto'))
        if lr_reducer == True:
            callback_list.append(tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, cooldown=0, patience=10, min_lr=0.5e-6))
        
        
        return callback_list

    def train(self):
        self.model.compile(loss= self.loss,optimizer=self.optimizer,metrics=[self.metrics])
        callback_list=self.get_callback_list(self.save_path)

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

            self.model.fit(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size), \
                steps_per_epoch=len(self.x_train) / self.batch_size,epochs=self.epochs, \
                validation_data=(self.x_test,self.y_test), callbacks=callback_list)

        else:
            self.model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,\
                validation_data=(self.x_test,self.y_test), callbacks=callback_list)
       

    def test(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.batch_size)
        scores= self.model.evaluate(self.val_dataset,batch_size=self.batch_size)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def build(self):
        print(f'loading {self.model_name}..\n')
        self.train()
        self.test()


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='Dataset [ "cifar10", "cifar100" ] ')
    parser.add_argument('--model_name', type=str, default='vgg11_bn', help='Model name')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--save_dir', type=str, default='./saved_models/', help='Saved model path')
    parser.add_argument('--data_augmentation', type=bool, default=True, help='Saved model path')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parser.parse_args()

    trainer=TrainTeacher(dataset_name=args.dataset_name,
                        model_name=args.model_name, 
                        batch_size=args.batch_size, 
                        epochs=args.epochs, 
                        save_dir=args.save_dir,
                        data_augmentation=args.data_augmentation,
                        metrics='accuracy'
                        )

    trainer.build()

if __name__ == '__main__':

    main()