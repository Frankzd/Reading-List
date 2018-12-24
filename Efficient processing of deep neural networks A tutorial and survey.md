# Efficient processing of deep neural networks: A tutorial and survey

> 早期的深度神经网络只关心获得最大的预测精度，而对具体实现的复杂度并没有太多考虑，这使得越来越多的算法难以在实际应用中使用。为了解决这一问题，人们发现通过使用软硬件协同优化设计的方法，可以获得精度高、吞吐量大，同时又功耗低、内存占用小的网络模型。

软硬件协同设计可以粗略地分为以下两种类型：

1. 降低操作符和操作数的精度。[**Quantization**]
2. 减少运算次数和模型尺寸。[**compression**、**pruning** .etc]

------

## 降低精度

在这里，降低精度主要通过量化操作来实现。量化是指将32bit的浮点数用低bit的定点数表示，量化的目标是在使用较低精度表示原模型的情况下最小化原始数据和量化后数据的误差。简而言之，就是想办法获得一个准确率不明显下降的低bit模型。

根据quantizer的不同，我们可以将量化方法分成以下三类：

1. linear quantization
2. log2 quantization
3. weight sharing

### 1. 线性量化(linear quantization)

线性量化，顾名思义即是将32bit的浮点数线性量化为定点数。在线性量化中，最常使用到的是8bit量化，而且现在越来越多的硬件架构也支持8bit定点数的运算。比如Google的TPU，NVIDIA的PASCAL GPU。这些硬件加速器都为神经网络inferen专门设计了8bit运算核心。

此外，还有另一种更为极端的量化方法将神经网络的参数量化为1bit(0,1)。比如[Binarized Neural Networks (BNN)](https://arxiv.org/abs/1602.02830),在BNN中，作者使用单bit的权值和激励，这可以极大地减少所需的运算量。但是这样的极端量化会带来很大的精度损失。

为了降低二值网络的精度损失，[Binary Weight Nets (BWN)](https://arxiv.org/abs/1711.11294)和[XNOR-Nets](https://arxiv.org/abs/1603.05279)介绍了几种对量化方法的改进。这其中包括给量化的输出增加一个系数，使参数量化为±w而不再是±1。

还有人提出了三值神经网络，即在二值网络的基础上增加了0的表示(-w,0+w)。三值网络虽然比二值网路在每个参数上多了一位表示，但是这样的代价是值得的。因为神经网络中往往会存在大量的0，使用三值网络往往可以在达到很高的压缩率的情况下也保持很可观的预测精度。有关三值网络的工作主要有：[Ternary Weight Nets (TWN)](https://arxiv.org/abs/1605.04711)、[Trained Ternary Quantization](https://arxiv.org/abs/1612.01064).

目前，也已经有很多硬件实现是基于二值或三值网络的。比如有[YodaNN](https://arxiv.org/abs/1606.05487v1)使用了二值的权值，[BRein](https://keio.pure.elsevier.com/en/publications/brein-memory-a-13-layer-42-k-neuron08-m-synapse-binaryternary-rec)使用了二值的权值和激励。对权值进行二值量化的方法同样在近似计算中得到了使用[SRAM](https://www.researchgate.net/publication/314649835_In-Memory_Computation_of_a_Machine-Learning_Classifier_in_a_Standard_6T_SRAM_Array)。最后，类脑计算芯片[TrueNorth](https://arxiv.org/abs/1603.08270)也可以使用期权值量化表实现二值激励和三值权重的神经网络。不过，以上硬件实现基本都不能支持state-of-the-art的DNN模型(除了YodaNN)。

### Log domain quantization

当网络模型参数呈现指数分布的时候(如下图所示)，使用指数域的量化也会带来更高的预测精度。此外，当时用log2 quantization时，我们还可以使用移位操作来代替乘法操作，大大减少了计算资源。

![moxingfenbu]()