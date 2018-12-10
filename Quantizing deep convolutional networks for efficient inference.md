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

