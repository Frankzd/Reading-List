# Reading-List

## 关于卷积

[How GEMM works for Convolutions](
https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

-----

## 关于模型压缩

### 轻量级网络模型

* Mobilenet-v2[“Inverted residuals and linear bottlenecks: Mobile networks for classification, detection and segmentation,” 2018.](https://arxiv.org/abs/1801.04381v2)

* Mobilenet-v1[“MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,” Apr. 2017.](https://arxiv.org/abs/1704.04861)

* Squeezenet[“Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <1mb model size,” CoRR, vol. abs/1602.07360, 2016.](https://arxiv.org/abs/1602.07360v2)

-----

### 量化

一般而言，神经网络模型的参数都是用的32bit长度的浮点型数表示，实际上不需要保留那么高的精度，可以通过量化，比如用0~255表示原来32个bit所表示的精度，通过牺牲精度来降低每一个权值所需要占用的空间。此外，SGD（Stochastic Gradient Descent）所需要的精度仅为6~8bit，因此合理的量化网络也可保证精度的情况下减小模型的存储体积。

根据量化方法不同，大致可以分为二值量化、三值量化和多值量化。

对网络网络进行量化要解决三个基本问题:

1. 如何对权重进行量化
2. 如何计算二值权重的梯度
3. 如何确保准确率

#### 二值量化(Binary Quantization)

思想是将权值矩阵中的单精度浮点数用两个值来表示，一般考虑使用符号函数或者加入线性化的符号函数来近似。

* [2017-Towards Accurate Binary Convolutional Neural Network](https://arxiv.org/abs/1711.11294)

* [“Binaryconnect: Training deep neural networks with binary weights during propagations,” 2015.](https://arxiv.org/abs/1511.00363)

* [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)

#### 三值量化(Ternary Quantization)

* [Neural Networks with Few Multiplications](https://arxiv.org/abs/1510.03009)

* [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)

* [Accelerating Deep Convolutional Networks using low-precision and sparsity](https://arxiv.org/abs/1610.00324)

#### 多值量化

* [Deep Learning with Limited Numerical Precision](https://arxiv.org/abs/1502.02551)

* [Towards the Limit of Network Quantization](https://arxiv.org/abs/1612.01543)

* [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

#### 变bit/组合bit量化

* [Adaptive Quantization for Deep Neural Network](https://arxiv.org/abs/1712.01048)

* [Fixed-point optimization of deep neural networks with adaptive step size retraining](https://arxiv.org/abs/1702.08171)

#### 哈希

* [Compressing Neural Networks with the Hashing Trick](https://arxiv.org/abs/1504.04788)

#### 其他

* [“Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,” Dec. 2017.](https://arxiv.org/abs/1712.05877)

* [Deep compression: Compressing deep neural network with pruning, trained quantization and huffman coding,” CoRR, vol. abs/1510.00149, 2015.](https://arxiv.org/abs/1510.00149)

* [“Model compression via distillation and quantization,” 2018.](https://arxiv.org/abs/1802.05668)

-----

### 神经网络加速器(Inference)

* [“Efficient processing of deep neural networks: A tutorial and survey,” CoRR, vol. abs/1703.09039, 2017.](https://arxiv.org/abs/1703.09039)

* [Nvidia, “The nvidia deep learning accelerator.” http://nvdla.org/](http://nvdla.org/)

* [“EIE: efficient inference engine on compressed deep neural network,” 2016.](https://arxiv.org/abs/1602.01528)

