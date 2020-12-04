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

Else if you want to use a pre-trained model, download the model through a download link in the below result section.
Please put the downloaded models in the dir of `./saved_models/`.


### Prune the model using EagleEye!!

Next, you can prune the trained model as following:

   ```shell
   python main.py --dataset_name=cifar10 --model_path=./saved_models/cifar10_resnet34.h5 --epochs=100 --min_rate=0.0 --max_rate=0.5 --num_candidates=15
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

The result of progress looks like as following:
```
Adaptive-BN-based accuracy for 0-th prunned model: 0.08163265138864517
Adaptive-BN-based accuracy for 1-th prunned model: 0.20527011036872864
Adaptive-BN-based accuracy for 2-th prunned model: 0.10084033757448196
...
Adaptive-BN-based accuracy for 13-th prunned model: 0.10804321616888046
Adaptive-BN-based accuracy for 14-th prunned model: 0.11284513771533966

The best candidate is 1-th prunned model (Acc: 0.20527011036872864)

Epoch 1/200
196/195 [==============================] - 25s 125ms/step - loss: 1.7189 - accuracy: 0.4542 - val_loss: 1.8755 - val_accuracy: 0.4265
...

Test loss (on base model): 0.3912913203239441
Test accuracy (on base model): 0.9172999858856201
The number of parameters (on base model): 33646666
The number of flops (on base model): 664633404

Test loss (on pruned model): 0.520440936088562
Test accuracy (on pruned model): 0.9136999845504761
The number of parameters (on pruned model): 27879963
The number of flops (on prund model): 401208200
```

Then, the pruned model will be saved in 'result'(default) folder. The sub-architecture is represented when before and after pruning:

<img src="./assets/fig1.png" alt="drawing" width="400"/>


### ResNet 34 on cifar10

|Model|Acc|[min_rate, max_rate]|Flops|Param num|File size|Download|
|-----|---|--------------------|-----|---------|---------|--------|
|Original|90.99%|None|145M|1.33M|15.9MB|[cifar10_resnet34.h5](https://drive.google.com/file/d/1SLqkXqImSIFBFEB_GRH7KGDxjojouqGD/view?usp=sharing)|
|Pruned|90.94%|[0, 0.5]|107M|1.07M|12.5MB|[cifar10_resnet34_pruned0.5.h5](https://drive.google.com/file/d/1GuJAHrrWb_aa3DA4POhum562prSUWN9K/view?usp=sharing)|
|Pruned|90.95%|[0, 0.7]|87M|0.90M|10.9MB|[cifar10_resnet34_pruned0.7.h5](https://drive.google.com/file/d/1aC-vINStAd1H5jwMlGS9GVwfpH18Kf9U/view?usp=sharings)|
|Pruned|89.14%|[0, 0.9]|59M|0.65M|8.17MB|[cifar10_resnet34_pruned0.9.h5](https://drive.google.com/file/d/1BUf_ml56DQG9k4AdD4Kfgm1LE2fuis-G/view?usp=sharing)|

### ResNet 18 on cifar10

|Model|Acc|[min_rate, max_rate]|Flops|Param num|File size|Download|
|-----|---|--------------------|-----|---------|---------|--------|
|Original|88.49%|None|70.2M|0.70M|8.4MB|[cifar10_resnet18.h5](https://drive.google.com/file/d/16mKwg1doK1fD6TlKWWxFZcke3Iov4hYk/view?usp=sharing)|
|Pruned|88.41%|[0, 0.5]|41.0M|0.43M|5.30MB|[cifar10_resnet18_pruned0.5.h5](https://drive.google.com/file/d/1qzEb1OtlU0-G6tJT6Dv9KF6zSdRxfg6F/view?usp=sharing)|
|Pruned|88.80%|[0, 0.7]|46.4M|0.42M|5.25MB|[cifar10_resnet18_pruned0.7.h5](https://drive.google.com/file/d/1Ly-iB_hTf8oK8U75yZa_jX7l14xHrT--/view?usp=sharing)|
|Pruned|86.11%|[0, 0.9]|24.2M|0.15M|2.12MB|[cifar10_resnet18_pruned0.9.h5](https://drive.google.com/file/d/1CWosBFFYoZrWt5-nTVk711kMzkoU0q4Q/view?usp=sharing)|


### vgg 16_bn on cifar10

|Model|Acc|[min_rate, max_rate]|Flops|Param num|File size|Download|
|-----|---|--------------------|-----|---------|---------|--------|
|Original|91.72%|None|664M|33.64M|385 MB|[cifar10_vgg16_bn.h5](https://drive.google.com/file/d/1zivoy2hB7_8bBFqN9fqDcOQQQ4LDwaMa/view?usp=sharing)|
|Pruned|91.37%|[0, 0.5]|401M|27.87M|319MB|[cifar10_vgg16_bn_pruned0.5.h5](https://drive.google.com/file/d/1rLVbvRqheHEtQLpkJSErIoYUK28_BTlC/view?usp=sharing)|
|Pruned|90.92%|[0, 0.7]|328M|24.92M|285MB|[cifar10_vgg16_bn_pruned0.7.h5](https://drive.google.com/file/d/1WA0o_JrMtSYaqIJaBVCjRKuASkXxm66s/view?usp=sharing)|
|Pruned|90.79%|[0, 0.9]|271M|22.68M|259MB|[cifar10_vgg16_bn_pruned0.9.h5](https://drive.google.com/file/d/1WQk_AwWNLBBcYgqrEknC2xyr_qQP_bYF/view?usp=sharing)|


:no_entry: If you run this code, the result would be different from mine.


## Understanding this paper

:white_check_mark: Check my blog!!
[Here](https://da2so.github.io/2020-10-25-EagleEye_Fast_Sub_net_Evaluation_for_Efficient_Neur_Network_Pruning/)