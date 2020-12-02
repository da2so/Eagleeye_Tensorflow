import argparse
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from core.eagleeye import EagleEye

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10','cifar100'])
    parser.add_argument('--model_path', type=str, default='./saved_models/cifar10_resnet34.h5', help = 'model path')
    parser.add_argument('--bs', type=int, default=256, help = 'batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epoch while fine-tuning')
    parser.add_argument('--lr', type=float, default=0.001 , help='learning rate')
    parser.add_argument('--min_rate',type=float, default=0.0, help='define minimum of search space')
    parser.add_argument('--max_rate',type=float, default=0.5, help='define maximum of search space')
    parser.add_argument('--flops_target', type=float, default=0.5, help='flops constraints for pruning')
    parser.add_argument('--num_candidates', type=int, default=15, help='the number of candidates')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory for a prunned model')


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF2_BEHAVIOR'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES']= '0'

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    args = parser.parse_args()

    eagleeye_obj=EagleEye(dataset_name=args.dataset_name,
                        model_path=args.model_path,
                        bs=args.bs,
                        epochs=args.epochs,
                        lr=args.lr,
                        min_rate=args.min_rate,
                        max_rate=args.max_rate,
                        flops_target=args.flops_target,
                        num_candidates=args.num_candidates,
                        result_dir=args.result_dir
                        )

    eagleeye_obj.build()

if __name__ == '__main__':
    main()