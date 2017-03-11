## 题目：
## 摘要
## 引言
## 开发语言和工具：
开发环境：Mac
开发语言： Python
开发框架： Tensorflow

### Tensorflow
Tensorlflow 是一个实现机器学习算法的接口，同是也是执行机器学习算法的框架，它的前端支持Python， C++， Go 和 Java等多种开发语言，后端使用C++， CUDA等写成。这样用户可以在一个硬件配置较好的机器中用Python开发，并在资源比较紧张的嵌入式环境和低延迟环境中用C++部署。
Tensorflow 建立的大型深度学习模型的应用场景非常广泛，包括语音识别，自然语言处理，计算机视觉，机器人控制，信息提取，药物研发，分子活动预测等。
Tensorflow于2015年11月在Github上开源。Tensorflow最早由Google Brain 的研究员和工程师共同开发，设计的初衷是加速机器学习的研究，并希望快速将研究原型转化为产品。次年4月补充了分布式版本，并于2017年1月发布1.0版本的预览，API接口日渐趋于稳定。
Tensorflow 使用数据流式图来规划计算流程，它可以将计算映射到不同的硬件和操作系统平台。凭借统一的架构，Tensorflow 可以方便地部署到各种平台，大大简化了真实场景中应用机器学习的难度。另外，采用数据流式图，可以让用户简单的在大规模的神经网络训练中实现并行计算。
除了支持常见的网络结构：卷积神经网络(Convolutional Neural Network, CNN)和循环递归神经网络(Recurrent Neural Network, RNN)外，Tensorflow 还支持深度强化学习(Deep Reinforcement Learning)以及其他的计算密集的科学计算。新加入的XLA已经开始支持JIT和AOT，另外它使用bucketing trick可以高效的实现循环神经网络。
2016年2月， Google 开源了 Tensorflow Serving ，这个组件可以将Tensorflow 训练好的模型导出，并部署可以对外提供预测的RESTful 接口。同时Tensorflow 也提供了 Tensorflow这一 Web 应用，用来监控Tensorflow 的运行过程。 Tensorflow 也有内置的TF.Learn和TF.Slim等上层组件可以帮助用户快速地设计网络，并且兼容Scikit Learn estimator 接口，可以方便的实现 evaluate， gird search， cross validation等功能。
Tensorlfow 拥有产品级的高质量代码，同时 Google具有强大的开发和维护能力，Tensorflow的整体架构设计也非常优秀。因此本次毕业设计采用Tensorflow 作为深度学习框架。

## 可行性分析与需求分析
## 六. 深度学习知识储备
#### 1.深度学习简介
神经网络的介绍？引入？后向传播算法？

误差逆传播（error BackPropagation,简称BP）算法是迄今为止最为成功的网络学习算法。现实任务中使用神经网络时，大多在使用BP算法进行训练。BP算法给予梯度下降(gradient descent)策略，以目标的负梯度方向对参数进行调整。BP算法执行如下操作：先将输入提供给输入层神经元，然后逐层将信号前传播，直到产生输出层的结果；然后计算输出层的误差，再将误差逆向传播至隐层神经元，最后根据隐层神经元的误差来对权重和偏置进行调整。该迭代过程循环进行，直到达到某些停止条件为止。BP算法的思想比较容易理解，现将反向传播过程推导如下所示：
1. 前向传播

如上图所示的一个三层人工神经网络，layer1至layer3分别是输入层、隐藏层和输出层。先定义变量如下：
![](http://latex.codecogs.com/gif.latex?w^{l})表示第**l-1**层神经元连接到第**l**层神经元的权重;
![](http://latex.codecogs.com/gif.latex?b^{l})表示第**l**层神经元的偏置;![](http://latex.codecogs.com/gif.latex?z^{l})表示第层神经元的输出,![](http://latex.codecogs.com/gif.latex?\\sigma)表示激活函数。即：
![](http://latex.codecogs.com/gif.latex?z^{l} = w^{l}a^{l-1} + b^{l};a^{l} = \\sigma(z^{l}))

2. 计算输出层产生的误差
误差函数是什么，这里使用常见的二次代价函数（quadratic cost function）：

![](http://latex.codecogs.com/gif.latex?C = \\frac{1}{2n}\\sum_{x}\|\|y(x)-a^{L}(x)\|\|^{2})
其中，**x**表示输入的样本， **y**表示实际的分类，![](http://latex.codecogs.com/gif.latex?a^{L})表示预测的输出， **L**表示神经网络的最大层数。则输出层的误差可以表示为：

![](http://latex.codecogs.com/gif.latex?\\delta^{L} = \\bigtriangledownC\\odot\\sigma^{\\prime}(z^{L}))
其中，![](http://latex.codecogs.com/gif.latex?\\odot})被称为Hadamard乘积，用于矩阵或向量之间点对点的乘法运算。

3. 从后向前，计算每一层神经网络产生的错误：

![](http://latex.codecogs.com/gif.latex?\\delta^{l} = ((w^{l+1})^{T}\\delta^{l+1}\\odot\\sigma^{\\prime}(z^{l}))
（4）计算权重和偏置的梯度

![](http://latex.codecogs.com/gif.latex?\\frac{\\partialC}{\\partialw} = a^{l-1}\\delta^{l})
![](http://latex.codecogs.com/gif.latex?\\frac{\\partialC}{\\partialb} = \\delta^{l})

        上图是一个三层人工神经网络，layer1至layer3分别是输入层、隐藏层和输出层。如图，先定义一些变量：
  表示第 l - 1层神经元连接到第l层神经元的权重； 表示第 层神经元的偏置； 
 ； 
   表示第 层神经元的输出， 其中 表示激活函数。即：
 

2. 计算输出层产生的误差
误差函数是什么，这里使用什么误差函数，常用的代价函数是二次代价函数（quadratic cost function）：
 
其中， 表示输入的样本，  表示实际的分类，  表示预测的输出，  表示神经网络的最大层数。

 

        其中，  表示Hadamard乘积，用于矩阵或向量之间点对点的乘法运算。
3. 从后向前，计算每一层神经网络产生的错误：
 

（4）计算权重和偏置的梯度


#### 2.最优化技术

#### 3.正则化技术

#### 4.卷积神经网络

为什么图像可以使用卷积神经网络进行处理?
什么是卷积神经网络？ 卷积神经网络的主要组成成分？
卷积？ 卷积层？ 重要思想？ 卷积层作用？本质：特征提取?
在卷积神经网络的中，输入与核函数做运算，产生输出，输出有时被称作特征映射。在机器学习中，输入和核函数通常都是高维张量。因此需要同时对多个维度进行卷积运算。例如，如果把二维的图像I作为输入，我们也相应的需要使用二维的核K：

卷积通过三个重要的思想来帮助改进机器学习系统： 稀疏交互，参数共享，等变表示。卷积也提供了一种处理可变大小的输入的方法：
稀疏交互：
传统的全连接层网络的每一个输入单元都和输出单元相连。传统的参数矩阵中的每一个参数代表一个输入和一个输出之间的连接值，也就是说每一个输入单元和每一个输入单元都相连。然而，卷积神经网络具有稀疏交互的特征。这通过使得核的规模远小于输入的规模来实现。
参数共享
等变表示：
卷积层的作用：

卷积层的实质是进行特征提取：


池化层池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置处的输出。常见的池化函数有：最大池化，平均池化，L^2范数，以及依据距离中像素距离的加权平均函数。
池化层的最为显著的作用是不变性(invariant)。
##### 5.RNN





2.6	对抗生成网络
3.	生成对抗网络（GAN：Generative Adversarial Networks） 
    GAN启发自博弈论中的二人零和博弈，由[Goodfellow et al, NIPS 2014]开创性地提出，包含一个生成模型（generative model G）和一个判别模型（discriminative model D）。生成模型捕捉样本数据的分布，判别模型是一个二分类器，判别输入是真实数据还是生成的样本。这个模型的优化过程是一个“二元极小极大博弈（minimax two-player game）”问题，训练时固定一方，更新另一个模型的参数，交替迭代，使得对方的错误最大化，最终，G 能估测出样本数据的分布。
4.	变分自编码器（VAE: Variational Autoencoders） 
    在概率图形模型（probabilistic graphical models ）的框架中对这一问题进行形式化——在概率图形模型中，我们在数据的对数似然上最大化下限（lower bound）。
5.	自回归模型（Autoregressive models） 
    PixelRNN 这样的自回归模型则通过给定的之前的像素（左侧或上部）对每个单个像素的条件分布建模来训练网络。这类似于将图像的像素插入 char-rnn 中，但该 RNN 在图像的水平和垂直方向上同时运行，而不只是字符的 1D 序列

1.	6.1 对抗生成网络介绍

1．6 .2 使用的网络
VGG  U-net Pix2Pix


## 相关实现
4.1 原理

4.2 流程

4.3 创新点
4.4 结果对比



## 结论






### 参考文献
[1]黄文坚, 唐源著.  Tensorflow 实战[M]- 北京， 电子工业出版社， 2017.2  
[2]Abadi M, Agarwal A, Barham P, et al. TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems[J]. 2016.
[3]Lecun Y, Bengio Y, Hinton G. Deep learning[J]. Nature, 2015, 521(7553):436-444.
[4]org.cambridge.ebooks.online.book.Author@ea. Deep Learning[M].
[5] Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[M]// Computer Vision – ECCV 2014. Springer International Publishing, 2014:818-833.
[6]Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. Journal of Machine Learning Research, 2014, 15(1):1929-1958.
[7] Goodfellow I J, Pougetabadie J, Mirza M, et al. Generative Adversarial Networks[J]. Advances in Neural Information Processing Systems, 2014, 3:2672-2680.
[8] Isola P, Zhu J Y, Zhou T, et al. Image-to-Image Translation with Conditional Adversarial Networks[J]. 2016.
[9] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
[10]Gregor, Karol, et al. "DRAW: A recurrent neural network for image generation." arXiv preprint arXiv:1502.04623
[11]Mirza M, Osindero S. Conditional Generative Adversarial Nets[J]. Computer Science, 2014:2672-2680.
[12] Chen X, Duan Y, Houthooft R, et al. InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets[J]. 2016.
[13]Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[J]. Computer Science, 2015.
[14] Arjovsky M, Chintala S, Bottou L. Wasserstein GAN[J]. 2017.
[15] https://github.com/wiseodd/generative-models
[16] CS231n
[17] http://deeplearning.stanford.edu/tutorial/

常见的课程相关主页
常见的网络
训练方法：最优化与过拟合理论
网络架构
GAN与VAE
Google 白皮书

