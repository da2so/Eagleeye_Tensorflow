# Eagleeye: fast sub-net evaluation for efficient neural network pruning


![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![Tensorflow version support](https://img.shields.io/badge/tensorflow-2.3.0-red.svg)

:star: Star us on GitHub — it helps!!


Tensorflow keras implementation for *[EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491)*

## Install

You will need a machine with a GPU and CUDA installed.  
Then, you prepare runtime environment:

   ```shell
   pip install -r requirements.txt
   ```

## Use

### Train base model

If you want to train a network for yourself:

   ```shell
   python train_network.py --dataset_name=cifar10 --model_name=resnet34
   ```

Arguments:

- `dataset_name` - Select a dataset ['cifar10' or 'cifar100']
- `model_name` - Trainable network names
	- Available list
		- VGG: ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn']
		- ResNet: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
		- MobileNet: ['mobilenet']
- `batch_size` - Batch size
- `epochs` - The number of epochs


### Prune the model using EagleEye!!

Next, you can prune the trained model as following:

   ```shell
   python main.py --dataset_name=cifar10 --model_path=./saved_models/cifar10_resnet34.h5 --epochs=50 --min_rate=0.0 --max_rate=0.5 --num_candidates=15
   ```

Arguments:

- `dataset_name` - Select a dataset ['cifar10' or 'cifar100']
- `model_path` - Model path
- `bs` - Batch size
- `epochs` - The number of epochs
- `min_rate` - Minimum rate of search space
- `max_rate` - Maximum rate of search space
- `num_candidates` - The number of candidates


## Result

The progress looks like as following:
```
Adaptive-BN-based accuracy for 0-th prunned model: 0.08163265138864517
Adaptive-BN-based accuracy for 1-th prunned model: 0.20527011036872864
Adaptive-BN-based accuracy for 2-th prunned model: 0.10084033757448196
...
Adaptive-BN-based accuracy for 13-th prunned model: 0.10804321616888046
Adaptive-BN-based accuracy for 14-th prunned model: 0.11284513771533966

The best candidate is 1-th prunned model (Acc: 0.20527011036872864)
```


|Model|Acc|[min_rate, max_rate]|Flops|Param num|File size|Download|
|-----|---|--------------------|-----|---------|---------|--------|
|ResNet34|90.99%|-|145M|1.33M|15.9MB|[cifar10_resnet34.h5](https://drive.google.com/file/d/1SLqkXqImSIFBFEB_GRH7KGDxjojouqGD/view?usp=sharing)|
|ResNet34|90.94%|[0, 0.5]|107M|1.07M|12.5MB|[cifar10_resnet34_pruned.h5](https://drive.google.com/file/d/1GuJAHrrWb_aa3DA4POhum562prSUWN9K/view?usp=sharing)|
