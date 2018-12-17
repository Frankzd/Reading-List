# Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference

## 简介

本文主要介绍Google在CVPR2018上的一篇8bit量化论文《Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference》，该文章在2017年12月份上传到arXiv上。这篇论文主要介绍了Tensorflow Lite 中8bit量化的原理以及量化模型的Training和Inference过程。文章中8bit量化与普通的量化过程基本一样，主要的贡献在于：（1）在8bit量化的公式基础上，通过公式变化以及设计，使得整个inference过程中只需要做整数计算；（2）提出了一种模拟量化训练的框架，在训练模型的过程中模拟量化的过程； （3）该文章的实验部分都是针对mobileNet这种精简模型进行量化实验，并且在实际的手机CPU上做性能测试，来充分展示量化算法的实用性。

## 背景介绍

当前图像领域的一些标志性网络例如AlexNet, VGG, GoogleNet, ResNet等在设计的时候主要考虑的是提升模型准确性，缺乏对于模型复杂性以及计算有效性的考虑。目前在移动端（手机，VR/AR，摄像头等）存在着大量的算法需求，而这些设备在存储以及模型运算速度上有很大的限制。当前主要的解决思路有两个：（1）设计一些特需的网络结构来减少网络参数并且加速计算，例如SqueezeNet, Xception, ShuffleNet, MobileNet(v1,v2)等等；（2）将模型参数（权重，激励）量化到更低的位宽来表示，例如TWN, BNN, XNOR-net, INQ等等。目前很多的量化算法都是在一些标志性的网络（AlexNet, VGG等等）做实验，由于这些网络设计时本身就存在很大的参数冗余，因此往往能得到较好的模型压缩比以及较大的速度提升，在作者看来这些并没有太大意义（我个人也非常赞同，本身在大公司的实际业务场景中，根本不用这些标志性网络，反而MobileNet，Xception, InceptionV4， MobileNet SSD这些才真正用的比较多，毕竟实际业务中速度还是非常重要的）。故而作者的关注重点在于将量化算法应用于本身就已经非常精简的网络（例如MobileNet）。其次，目前很多量化算法主要考虑的是减少模型在移动设备上的存储，而较少的去考虑模型在设备上的计算有效性，因此该论文着重关注量化后的算法在真实的移动CPU（例如高通骁龙835， 821等等）上计算的有效性。

在本篇文章中，作者做出了以下几点贡献：

* 提出了一种量化方法，可以将weights和activations都量化为8-bit整数，以及将一小部分bias参数量化为32-bit整数。
* 提出了一种针对量化模型的推断框架，该框架可以非常高效地在integer-arithmetic-only(只支持整数运算的)硬件上实现，如Qualcomm Hexagon。此外，作者还描述了如何让该框架高效准确地运行在ARM NEON上。
* 提出了一种针对量化模型的训练框架，该架构与上一条中所说的推断架构相辅相成，可以有效降低量化操作对真实模型所带来的精度损失。
* 作者将该框架应用于基于MobileNet的分类与检测应用中，并且在常见的ARM CPU上进行了测试并给出benchmark结果。

## Quantized Inference

### Quantization scheme

>Our quantization scheme is implemented using integer-only arithmetic during inference and floating-point arithmetic during training, with both implementations maintaining a high degree of correspondence with each other.

作者提出的量化方案在推断期间使用整数运算，在训练期间使用浮点运算，两种实现保持高度的一致性。

文章中作者使用如下公式对模型进行量化：
$$
r = S(q-Z) \qquad \qquad \qquad \qquad(1)
$$
上式中r是量化之前的数值，q为量化后的结果，S和Z是两个常数(量化参数)。
> Our quantization scheme uses a single set of quantization parameters for all values within each activations array and within each weights array; separate arrays use separate quantization parameters.

此量化方案对每个activations阵列和每个weights阵列中的所有元素使用同样的量化参数，不同的阵列使用不同的量化参数。

对于8-bit量化，q就被量化为一个8-bit的整数。(同理，对于B-bit的量化，q就被量化为B-bit的整数)。但是对于bias参数，统一都量化为32-bit整数。

常数S(scale)是一个任意的正实数，在软件中通常表示为浮点数(在后面的文章中会介绍如何在推断过程中避免使用浮点数来表示S)。

常数Z(zero-point)的数据类型和量化后的结果q相同，用来表示原数据中的0(这使得我们可以用一个准确的量化后的数据来表示原数据中的0)。

以上所说的数据格式可以用如下的c++数据结构来表达：

```c++
template<typename Qtype>        //e.g. Qtype=uint8
struct QuantizedBuffer{
    vector<Qtype> q;        //the quantized values
    float S;                //the scale
    Qtype Z;                //the zero-point
}
```

下面我将详细分析下这种量化方式与传统的量化的差别：

```c++
//定义量化区间以及scale
float min,max
float S = (max - min) / (n-1)   //for 8-bit n=256
Uint_8 q = round((x - min)/S)

//Google Quantization
r = S * (q - Z)
  = S * (q + round(min/S))

//Normal Quantization
r = S * q + min
  = S * (q + min/S)
```

如上所示，主要的差别在于最小值的位移偏差，google的量化方式将位移的最小偏差也做了量化，可以理解为是以原始0点为基准做的量化，这样还原回去的r偏差会更大一些，当然这种量化的方式主要是为了在inference的过程中直接使用量化后的值做乘加运算而不必还原回去。综上分析可以知道，这种量化方式相比于传统的量化方式在精度上会损失一些。

### Integer-arithmetic-only matrix multiplication

这里主要解决三个问题：

1. 如何将两个不同量化的数直接做乘加操作而不用还原到实数
2. 上面的量化公式中浮点系数S如何变成整数操作
3. 如何将bias和激励函数加到整数操作中。

![multiplication](https://github.com/Frankzd/Reading-List/blob/master/images/multiplication.png?raw=true)

在实际的量化过程中S1、S2、S3都是可以预先计算的(先统计权重以及输入输出的量化空间，计算量化缩放系数S)，因此M值是可以离线计算出来的。论文作者通过经验发现M值总是在(0,1)区间内，于是$M$可以写成如下表示：
$$
M = 2^{-n}M_0
$$

其中$M_0$属于[0.5,1)区间内，n是一个非负整数。$M_0$可以采用int32的值来表示(类似于定点表示，即该值的所有值都是小数部分)。那么乘以浮点数$M$就变成了先乘一个定点的int32的整数$M_0$,再进行n次的移位操作。

> 对于这一步的操作，我的理解是先确定$M$的值($M$的取值范围在[0,1)之间)，之后再通过移位操作，对$M$进行移位操作得到一个范围在[0.5,1)之间的数，即$M_0$

第二步： 上面的公式（4）中每个数都需要先减对应的零点再做相乘操作，这样会进行较多的减法操作($2N^3$次)，在公式（4）的基础上可以进一步的推理变换如下形式： 

![less subtractions](https://github.com/Frankzd/Reading-List/blob/master/images/less_subtractions.png?raw=true)

每个$a_2^{(k)}$和$a_1^{(i)}$的计算需要$N$次加法操作，这样一共只需要进行$2N^2$次加减法操作。这时主要的计算集中在$\sum q_1^{(i,j)}q_2^{(j,k)}$处(大概需要$2N^3$次算数运算)
我们将$q_1$矩阵定义为weights矩阵$q_2$矩阵定义为activations矩阵，并且使用uint8数据类型表示,uint8类型数据的乘累加的结果使用int32类型存放(具体原因后面给出)，这样式(9)中的求和过程就能表示为：
$$
int32 += uint8 * uint8.
$$
即公式(7)中括号内的值都使用int32来表示
至此我们也得到了为什么要将bias量化为32bit定点数，就是为了方便与int32类型的累加和相加。
第三步：将bias进行32bit量化：

> 对于bias的量化还有以下几点需要注意
> * 使用int32数据类型进行量化
> * 就将0作为量化过程中的zero-point，即给予0点做量化，所以量化后的值存在正负之分。
> * 另一个量化参数$S_{bias}$与累加过程中的scale相同($S_{weights}*S_{activations}$)
> $$S_{bias} = S_1S_2,  Z_{bias} = 0.$$

所以$r_{bias}=S_{bias}(q_{bias}-Z_{bias})=S_1S_2q_{bias}$。将$r_{bias}$代入卷积求和中：
$$
S_3(q_3^{(i,k)}-Z_3) = \sum_{j=1}^N S_1(q_1^{(i,j)}-Z_1)S_2(q_2^{(j,k)}-Z_2)+S_1S_2q_{bias}
$$
做如上相同的变换可以得到：
$$
q_3^{(i,k)}=Z_3+M(NZ_1Z_2-Z_1a_2^{(k)}-Z_2a_1^{(i)}+\sum_{j=1}^N q_1^{(i,j)}q_2^{(j,k)}+q_{bias})
$$
至此我们还需要进行步骤：

1. 将输出结果的取值范围缩小到最终8bit输出的量级(scale down)
2. 转换为uint8类型数据(saturating cast)
3. 施加激活函数

```c++
//saturating cast
if(data<0)  
    data=0;  
else if(data>255)  
    data=255;  
```

上式括号中的值通过整数的乘累加运算得到int32的整数，乘以浮点数M相当于将int32的整数进行缩放，乘以浮点数M的过程采用前面所述的，先乘以32bit定点数$M_0$，之后再进行移位操作，最后将结果截取到[0,255]之间，即将小于0的值变成0，大于255的值变成255.
所以对于激活函数，例如ReLU和ReLU6，其效果已经体现在了clamping到[0,255]这个过程中，所以不再需要激活函数。

### Training with simulated quantization
传统的量化方法往往在大模型上可以取得较好的效果，但是应用在小模型上时精度下降就会很大。作者认为这是两个原因所导致的：

1. 不同的输出channel对应的weights存在较大的差异(more than 100x)
2. 一些明显的outlier权重值会使其他权重值量化时的精度偏低(outlier权重值影响scale，进而影响量化精度)

本文提出了一种模拟量化的训练方式即添加一些伪量化操作，这些伪量化的位置与前向过程中量化的位置保持一致。训练过程中的一些关键点如下： 

1. 对比前向过程，在相同位置插入伪量化操作 
2. 求导过程依然不变，即伪量化操作不参与求导，参数更新依然在原始的权值上面 
3. 所有的weight和bias是采用原始的float方式存储 
4. 对于存在BN的网络，在做伪量化之前，将BN的参数“folded into”权重weight中 
5. 对于存在连接的网络（ResNet），在连接操作后做伪量化操作

> 伪量化操作的理解：简单来说就是将量化过程对模型预测精度的影响引入到训练过程中，让模型在训练时学习量化将会带来的精度损失，进而得到一个更加符合实际情况的模型。

第一步：伪量化过程如下：
$$
clamp(r;a,b) := min(max(x,a),b)
$$
$$
s(a,b,n) := \frac {b - a}{n - 1}
$$
$$
q(r;a,b,n) := \lfloor \frac {clamp(r;a,b)-a}{s(a,b,n)} \rceil s(a,b,n) + a
$$
即在训练过程中，模拟量化的精度损失，将weight先做量化再还原回去，这里需要注意的是该还原过程采用的是传统的量化还原方式，没有采用本文提出的量化公式进行还原。

第二步：一个不带BN层的模拟量化训练流程：
![Algorithm1](https://github.com/Frankzd/Reading-List/blob/master/images/Algorithm1.png?raw=true)
其结构图如下：

![FigureC4](https://github.com/Frankzd/Reading-List/blob/master/images/FigureC4.png?raw=true)


训练的过程中需要学习两个东西： 
1. 学习权重等参数
2. 学习权重和激励值的量化区间

对于权重而言，只需要统计其最大最小值作为量化区间；对于激励值，激励值在inference的过程中会随着输入的不同而不断变化，因此需要做类似BN层中的（均值和方差）一样的操作，在训练的过程中通过EMA算法统计出激励值的量化区间，EMA算法统计的过程如下：
$$
a_0 = min(O_0)
a_t = alpha * a_{t-1} + (1 - alpha)*min(O_t)
$$
上面展示了采用EMA统计激励的量化区间下限值的过程，时间t代表训练过程中不同batch，$O_0$代表0时刻的输出激励值，$a_0$表示0时刻统计的下限值，$a_t$表示t时刻统计的下限值，alpha表示平滑系数(一般取接近于1的值，例如0.99)。

第三步：一个带BN层的模拟量化训练过程:
首先复习一下BN的计算过程：
![batchnorm](https://github.com/Frankzd/Reading-List/blob/master/images/batchnorm.png?raw=true)
上面展示了BN的计算过程，在Inference的过程中，一般会将BN的参数融于卷积运算过程中：
$$
y_i = r\overline x_i + \beta = \frac {r(\sum {wx + bias -u})}{\sigma} + \beta = \sum {\frac {rw} {\sigma}x} + \frac {r * bias} {\sigma} + \beta - \frac {ru}{\sigma}
$$
可以看到，在做卷积前先将w和bias结合BN参数做一个变换$\frac {rw}{\sigma}$,$\frac {rbias}{\sigma}$,再做正常的卷积，卷积结果加上$(\beta - \frac {ru}{\sigma})$。

![BN_Inference](https://github.com/Frankzd/Reading-List/blob/master/images/BN_inference.png?raw=true)
