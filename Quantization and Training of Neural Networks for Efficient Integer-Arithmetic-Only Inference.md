# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

## 简介

本文主要介绍Google在CVPR2018上的一篇8bit量化论文《Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference》，该文章在2017年12月份上传到arXiv上。这篇论文主要介绍了Tensorflow Lite 中8bit量化的原理以及量化模型的Training和Inference过程。文章中8bit量化与普通的量化过程基本一样，主要的贡献在于：（1）在8bit量化的公式基础上，通过公式变化以及设计，使得整个inference过程中只需要做整数计算；（2）提出了一种模拟量化训练的框架，在训练模型的过程中模拟量化的过程； （3）该文章的实验部分都是针对mobileNet这种精简模型进行量化实验，并且在实际的手机CPU上做性能测试，来充分展示量化算法的实用性。

## 背景介绍

当前图像领域的一些标志性网络例如AlexNet, VGG, GoogleNet, ResNet等在设计的时候主要考虑的是提升模型准确性，缺乏对于模型复杂性以及计算有效性的考虑。目前在移动端（手机，VR/AR，摄像头等）存在着大量的算法需求，而这些设备在存储以及模型运算速度上有很大的限制。当前主要的解决思路有两个：（1）设计一些特需的网络结构来减少网络参数并且加速计算，例如SqueezeNet, Xception, ShuffleNet, MobileNet(v1,v2)等等；（2）将模型参数（权重，激励）量化到更低的位宽来表示，例如TWN, BNN, XNOR-net, INQ等等。目前很多的量化算法都是在一些标志性的网络（AlexNet, VGG等等）做实验，由于这些网络设计时本身就存在很大的参数冗余，因此往往能得到较好的模型压缩比以及较大的速度提升，在作者看来这些并没有太大意义（我个人也非常赞同，本身在大公司的实际业务场景中，根本不用这些标志性网络，反而MobileNet，Xception, InceptionV4， MobileNet SSD这些才真正用的比较多，毕竟实际业务中速度还是非常重要的）。故而作者的关注重点在于将量化算法应用于本身就已经非常精简的网络（例如MobileNet）。其次，目前很多量化算法主要考虑的是减少模型在移动设备上的存储，而较少的去考虑模型在设备上的计算有效性，因此该论文着重关注量化后的算法在真实的移动CPU（例如高通骁龙835， 821等等）上计算的有效性。

![Integer-arithmetic-only inference](https://github.com/Frankzd/Reading-List/blob/master/images/a.png?raw=true)


