# Quantizing Deep Convolutional Networks for Efficient Inference

## Abstract

本篇文章是对卷积神经网络推断过程中使用到的量化技术的概述，主要分为以下几部分：

1. 对多种CNN架构进行**post-training**量化：对weights采用8bit的per-channel量化、对activations采用8bit的per-layer量化。采用此种量化方法所得的分类准确性与使用浮点数的网络相差在2%以内。[Section 3.1]
2. 通过将weights量化为8bit，我们可以让模型的规模缩小4倍。此种量化方法实现非常简单，只需要对训练后的模型进行线性量化即可(without re-train)。[Section 3.1]
3. 我们对量化过后的网络在CPUs和DSPs上运行的延迟(lantencies)进行了测评。结果显示，与未量化的浮点数网络相比，量化(8bit)过后的网络在CPUs可以获得2x-3x的加速比。而在具有SIMD(Single Instruction Multiple Data)功能的专用处理器上(如高通QDSPs with HVX)上，则可以获得10x的加速比。[Section 6]
4. Quantization-aware training(训练时量化)与之前的post-train量化相比能带来一些性能上的提升，比如可以将8bit量化时造成的精度损失降低为只比浮点网络低1%以内。Quantization-aware training也能将weights的精度量化为4bit，同时带来2%到10%之间的精度下降(规模越小的网络精度下降越多)。[Section 3.2]
5. 对一些使用Quantization-aware training量化方法获得高预测精度的实验进行分析。
6. 在量化过程中，我们推荐对weights进行per-channel量化，对activations进行per-layer量化。这样有利于硬件加速和对运算kernel进行优化。[Section 7]

-----

## 1 Introduction

深度神经网络已经越来越多地被应用在端系统中。端设备由于存储和功耗的限制而导致其运算能力不足，而且我们也需要减少运算和端设备之间的通信量以减少由网络通讯带来的功耗。所以，我们急需寻找到一种可以减少网络规模、加快推断速度、降低运行功耗的优化模型。对神经网络模型压缩技术的研究近几年已经取得了很多不错的进展，目前主要有以下几种主要方法：

1. 在训练模型之初就是用高效地模型架构,如MobileNet[1][2],SqueezeNet[3]等。
2. 通过剪枝、量化等压缩方法对复杂模型进行压缩。如quantization[4],Binaryconnect[5],Deepcompression[6]等。
3. 还可以通过对压缩后低精度的网络采用更高效的运算kernel来提高推断的速度。如GEMMLOWP[7],Intel MKL-DNN[8],ARM CMSIS[9],Qualcomm SNPE[10],Nvidia TensorRT[11],定制硬件推断加速器[12][13][14]。

**量化**是其中一种非常常见和重要的方法，通过降低神经网络权值和激励的表示精度，量化后的网络可以获得以下的优点：

* 首先量化方法适用于绝大多数的应用场景和基本上全部的神经网络模型。我们不需要为了使用量化压缩而设计新的模型架构，这极大地节省了开发的时间。在很多情况下，我们可以直接将训练好的浮点数模型快速量化为定点数模型，在不进行re-train的情况下就可以带来很少的精度损失。同时，多种架构和硬件设计都已经支持了定点数(多数为8bit)权值和激励模型的快速inference。
* 更小的内存占用：对一个模型使用8bit量化，我们可以将其规模减小4倍。这种操作不引入新的需要存储的参数，只是对weights进行了量化。这使得我们可以更快地下载和更新模型。
* 更少的缓存占用：对activations进行量化意味着我们可以降低模型运行时的缓存占用。深度神经网络的中间结果都保存在缓存中供下一层网络进行使用，对模型的activations进行低精度的表示可以大大减少对缓存的占用。
* 更快的运算：有许多硬件架构都支持对定点数的快速运算。
* 更低的功耗：在很多应用中，对访存所带来的功耗要远远大于预算的功耗，而移动8bit的数据要比移动32bit浮点数带来的功耗小4倍，对模型进行量化可以有效降低网络的功耗。

-----

## 2 Quantizer Design

**To be continued**

-----

## 3 Quantized Inference:Performance and Accuracy

在本章中我们主要讨论采用不同的量化方法对模型的性能和精度所带来的影响。

### 3.1 Post Training Quantization

Post Training 顾名思义，就是指直接对训练后的网络进行量化而不进行re-train的量化方法，我们既可以只对weights量化，也可以同时对weights和activations进行量化。这种量化方法无非是最简单、最方便实现的一种，那么它的实际效果如何呢？接下来的几小节我们就具体看看它在实际应用中的表现吧。

#### 3.1.1 Weight only quantization

一个最简单的方法就是将模型的weights从32bit的浮点数表示为低精度的8bit定点数。这种量化方法适用于只希望将模型压缩为便于传输和存储的大小而不是很在意预测精度损失的情况(个人不建议使用)。

#### 3.1.2 Quantizing both weights and activations

此方法同时对weights和activations进行量化，由于加入了对中间结果的量化，所以我们需要标定数据和计算activations的动态范围(每层之间的activations范围不同小则相差数倍大则相差两个数量级)。通常情况下，100个mini-batch就足够我们预测该层activations的范围了，在实际应用中也可以将训练集的activations的范围当做测试集的范围。

#### 3.1.3 Experiments

Network | Model Parameters | Top-1 Accuracy on ImageNet(fp32)
-----|-----|-----|
Mobilenet_V1_0.25_128 | 0.47M | 0.415
Mobilenet_V2_1_224 | 3.54M | 0.719
Mobilenet_V1_1_224 | 4.25M | 0.709
Nasnet_Mobile | 5.3M | 0.74
Mobilenet_V2_1.4_224 | 6.06M | 0.749
Inception_V3 | 23.9M | 0.78
Resnet_v1_50 | 25.6M | 0.752
Resnet_v2_50 | 25.6M | 0.756
Resnet_v1_152 | 60.4M | 0.768
Resnet_v2_152 | 60.4M | 0.778

Table1: Deep Convolutional networks:Model size and accuracy

上表罗列了几种深度神经网络(float)的模型大小和在ImageNet数据集上的Top-1准确率。

**Weight only quantization**：
Network | Asymmetric,per-layer | Symmetric ,per-channel | Asymmetric,per-channel | Floating Point
-----|-----|-----|-----|-----|
Mobilenetv1_1_224 | 0.001 | 0.591 | 0.704 | 0.709
Mobilenetv2_1_224 | 0.001 | 0.698 | 0.698 | 0.719
NasnetMobile | 0.722 | 0.721 | 0.74 | 0.74
Mobilenetv2_1.4_224 | 0.004 | 0.74 | 0.74 | 0.749
Inceptionv3 | 0.78 | 0.78 | 0.78 | 0.78
Resnet_v1_50 | 0.75 | 0.751 | 0.752 | 0.752
Resnet_v2_50 | 0.75 | 0.75 | 0.75 | 0.756
Resnet_v1_152  | 0.766 |  0.763 | 0.762 | 0.768
Resnet_v2_152  | 0.761  | 0.76 | 0.77 | 0.778
Table2:Weight only quantization:非对称的逐通道量化可以获得更高的准确率

**Weight and Activation Quantization**:
Network | Asymmetric,per-layer | Symmetric ,per-channel | Asymmetric,per-channel | Activation Only | Floating Point
-----|-----|-----|-----|-----|-----
Mobilenet-v1_1_224 | 0.001 | 0.591 | 0.703 | 0.708 | 0.709 
Mobilenet-v2_1_224 | 0.001 | 0.698 | 0.697 | 0.7 | 0.719
Nasnet-Mobile | 0.722 | 0.721 | 0.74 | 0.74 | 0.74
Mobilenet-v2_1.4_224 | 0.004 | 0.74 | 0.74 | 0.742 | 0.749
Inception-v3 | 0.78 | 0.78 | 0.78 | 0.78 | 0.78
Resnet-v1_50 | 0.75 | 0.751 | 0.751 | 0.751 | 0.752
Resnet-v2_50 | 0.75 | 0.75 | 0.75 | 0.75 | 0.756
Resnet-v1_152 | 0.766 | 0.762 | 0.767 | 0.761 | 0.768
Resnet-v2_152 | 0.761 | 0.76 | 0.76 | 0.76 | 0.778
Table3：Post training quantization of weights and activations:
对weights进行per-channel量化,对activations进行per-layer量化在以上几种网络中表现都十分不错，同时我们可以发现非对称量化往往可以获得比对称量化更好的效果。

> 补充：per-channel与per-layer的区别以及为什么在weights上选择per-channel量化而在activations中选择per-layer量化：
>> 量化操作是通过quantizer实现的(quantizer由scale和zero-point定义)，per-channel quantization 与per-layerquantization的区别就在于quantizer作用的范围不同。已知每层的weights是一个四维的tensor，per-layer quantization即是对该四维tensor设置统一的scale和zero-point，而per-channel quantization则是对每个通道都设置专有的scale和zero-point。可想而知细粒度的per-channel quantization应该能取得更好的效果，那么为什么不对activations也采用per-channel quantization呢？这是由于对activations采取per-channel量化会影响数据的局域性从而使卷积和全连接层中的乘法操作变得更加复杂。

**总结**：

1. per-channel 量化可以提供较高预测精度，使用非对称的per-channel quantization可以获得和原模型精度相差很小的量化模型。
2. 对activations量化到8bit时基本上不会对模型的精度带来太大影响。
3. 参数越多的网络对量化操作的鲁棒性越高。
4. 当对weights采用layer粒度的量化时会有一个比较大的精度下降(规模越小的网络下降越明显)。
5. 量化所带来的精度下降基本都集中于weights量化时带来的精度损失。


### 3.2 Quantization Aware Training

Quantization Aware Training 的量化操作在训练过程中进行，可以获得比Post-train 量化更高的预测精度。在训练过程中，我们在正向传播和反向传播中都对weights和activations进行量化，与此同时我们还要保留一份浮点数weights的副本，所有梯度的更新都应该对浮点数表示的weights进行。这样可以确保我们每次对weights的更新都以很小的梯度进行。之后再对更新后的weights进行量化，并将量化后得weights应用于随后的正向、反向传播中。

在训练过程中量化的具体流程如下：

1. 使用一个pre-traind浮点数模型或者重新训练一个模型。
2. 在模型中加入模拟的quantizer。
3. 训练模型：在训练结束后，我们会获得量化后模型的量化参数(scale,zero-point)。
4. 在上一步中训练出的模型中的weights和activations仍然是通过浮点数来表示的，只是它所代表的的精度有所下降。所以在这一步中，我们对模型进行转换，将浮点数表述的模型转换为定点数模型。
5. 将使用定点数表示的模型部署在专用加速器上。

#### 3.2.1 Experiments

Quantization aware training可以减小压缩模型和浮点数模型精度上的差距，甚至也适用于per-layer的weights量化。我们重复了与Post-training quantization 中进行过的相同的实验，将两种量化思路进行对比。

Network | Asymmetric,per-layer(Post Training Quantization) | Symmetric ,per-channel(Post Training Quantization) | Asymmetric,per-layer(Quantization Aware Training) | Symetric, per-channel(Quantization Aware Training) | Floating Point
-----|-----|-----|-----|-----|-----
Mobilenet-v1_1_224 | 0.001 | 0.591 | 0.70 | 0.707 | 0.709 
Mobilenet-v2_1_224 | 0.001 | 0.698 | 0.709 | 0.711 | 0.719
Nasnet-Mobile | 0.722 | 0.721 | 0.73 | 0.73 | 0.74
Mobilenet-v2_1.4_224 | 0.004 | 0.74 | 0.735 | 0.745 | 0.749
Inception-v3 | 0.78 | 0.78 | 0.78 | 0.78 | 0.78
Resnet-v1_50 | 0.75 | 0.751 | 0.75 | 0.75 | 0.752
Resnet-v2_50 | 0.75 | 0.75 | 0.75 | 0.75 | 0.756
Resnet-v1_152 | 0.766 | 0.762 | 0.765 | 0.762 | 0.768
Resnet-v2_152 | 0.761 | 0.76 | 0.76 | 0.76 | 0.778
Table3：Post training quantization of weights and activations: