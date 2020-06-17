[**CLD的博客**](https://cenleiding.github.io/)

# 神经网络ANN

什么是神经网络？对于神经网络整体的简单理解。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_1.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_1.jpg)

## 神经网络是什么？

首先来看一下神经网络的流程：数据从输入层输入，然后经过中间的隐藏层的激励函数，最后在输出层输出相应大小的结果。

### 方向一：从过程看

　　该部分内容参考[colah的博客。](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

　　每一层的隐藏层都可以看成一个变换函数，输入数据经过一层隐藏层就相当于进行了一次变换。于是我们可以通过变换将不同平面的输入数据拉伸变换到一个平面并进行分离，这样对于输出结果我们只要进行简单的线性分割就能分类数据了。

　　比如我们现在要对以下的两个曲线进行分类：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_2.png)](https://cenleiding.github.io/神经网络ANN/ANN_2.png)

　　我们发现无法使用线性分割的方式分类曲线，而获得一个“扭曲”的边界函数并不容易。于是通过神经网络我们将曲线变换为如下样子：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_3.png)](https://cenleiding.github.io/神经网络ANN/ANN_3.png)

　　这样我们就可以对神经网络的输出结果进行线性的分割，问题就变的很简单。当然这只是最简单情况，实际上对于较复杂的情况也一样：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_4.gif)](https://cenleiding.github.io/神经网络ANN/ANN_4.gif)

　　我们只要增加网络的隐藏层数量和每层的激励单元数量，就可以将一些更为复杂的函数分离开来。

　　但是这只是理想状态，往往我们并不能做到随便增加网络节点和层数因为计算量实在太复杂，而实际问题的数据又往往有着复杂的结构，如下图：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_5.png)](https://cenleiding.github.io/神经网络ANN/ANN_5.png)

　　现实数据往往是缠绕在一起的，要完全分离这样的数据需要十分复杂的神经网络。好在我们在实际应用时并不要求一定要得到完美的结果，于是我们可以通过神经网络将数据变换为以下样式：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_6.png)](https://cenleiding.github.io/神经网络ANN/ANN_6.png)

　　我们尽可能的让数据重叠的部分变少，然后分离数据，这样会产生误差，但是能较快的得到可用结果。

### 方向二：从结果看

　　上面讲的从过程看虽然十分形象，但是并不能帮助我们实现得到神经网络~毕竟我们并不知道应该怎么变化~~至少现在不能~~。但是从结果出发我们可以通过数学训练得到想要的神经网络！

　　对数据进行变形的过程，可以理解为对输入数据矩阵进行变换，每一个网络节点都是一次变换，我们的目标是强行将输入矩阵变的和目标结果矩阵相似。而每一次变换都是一次矩阵的相乘，于是可以看成【输入矩阵*网络矩阵】=>【输出矩阵】，关键是确定网络矩阵中茫茫多的参数。而输出矩阵和目标矩阵的相似度则可以用**代价函数（cost -function）**来表示，具体怎么设计代价函数看情况来定。这时我们可以将整体看成【输入矩阵 * 输出矩阵 * 代价函数】=>【代价值】。

　　毫无疑问这个代价值越小越好。而这个带价值是一个多元多次方程，如下图所示：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_7.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_7.jpg)

　　我们的目标就是从中找到最低的那个点。要求一个方程的最低点，就需要对每个参数求偏导，然后沿着梯度下降的方向变换参数，这就是**梯度下降算法 （Gradient Descent algorithm）GD。**而具体如何确定每个节点参数的变化，这就需要用到**反向传播算法 （Back Propagation algorithm）BP。**

> 所以从结果来看：**神经网络的搭建就是假设一个高次的变化方程；训练就是凑变换函数参数。** 这样看来神经网络原理也就没什么，就是靠计算机的算力强行凑函数。。。

## 梯度下降和反向传播是什么？

这部分内容还是主要参考[colah的博客。](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) 和 [AGH](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)

> **梯度下降**:

　　上面也讲到了，梯度下降算法的目的就是获得目标方程的最低点。而最低点的“斜率”是为0的，那么只要对方程求偏导，就能知道当前点的“斜率”，然后只要沿着这个斜率的反方向移动就能理论上到达最低点了。

　　过程如下图所示：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_8.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_8.jpg)

　　**局部最低点问题：** 从上图我们可以看出，沿着梯度下降的方向移动并不一定能获得全局的最低点。因为神经网络创建的方程往往是很复杂的，有着大量的“谷底”，不同的起点最后掉落的“谷底”不一定相同。而这个起点取决于我们对参数的初始值设定。因此在实际训练网络时，有时对初始值设定进行一些“小操作”会有助于提高最后结果的准确度。。。嗯，所以神经网络很玄学！

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_9.png)](https://cenleiding.github.io/神经网络ANN/ANN_9.png)

　　**学习速率问题：** 前面讲了要沿着梯度下降的方向移动，但是具体移动多少呢？首先很明显跟预测值与目标值的差距有关，两者差距越大那就要移动的越多，这就是**梯度误差（error gradient）**。但是如果直接根据这个误差来调整参数，发现参数变化幅度太大了，会导致结果一直在震荡（如右图所示），甚至直接就飞出这个“山谷”。。。因此需要一个值来缩小这个变化幅度，这就是**学习速率（learning rate）** ，学习速率往往是一个比较小的值如0.01，这样就能使得方程慢慢的向谷底移动。学习速率的选取是个需要注意的点，太小会导致下降的十分缓慢，训练网络需要很长时间，而太大又会引起“震荡”，学不到好的结果。

> **反向传播：**

　　知道了要进行梯度下降，但怎么具体操作是个问题，毕竟参数这么多，一个个求过来这个计算量不敢想~

　　于是就有了**反向传播算法**。

　　这个部分一开始看公式一直很疑惑为什么要这么做？不就是个链式法则？这和直接求每个参数的偏导（正向传播）到底快在哪里？直到看了colah的讲解才恍然大悟，什么链式法则都不是重点，**避免重复计算才是反向传播算法的精髓！**

　　接下来膜拜colah大佬==>

　　首先看一下正常的求偏导是怎么求的：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_10.png)](https://cenleiding.github.io/神经网络ANN/ANN_10.png)

　　比如要求b对e的影响能力，我们用链式法则就能很简单的算出e对b的偏导值。

　　然后抽象一点，比如现在有如下网络，分别正向和反向的求一次偏导：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_11.png)](https://cenleiding.github.io/神经网络ANN/ANN_11.png)

　　正向（Forward-Mode）:以x为主，及表示了x**参数对每个节点的影响能力** 。

　　反向（Reverse-Mode）:以Z为主，及表示了Z**结果对每个节点的影响能力**。

　　嗯。。。。。看起来很有道理的样子，但是这有什么用呢？

　　那么就需要用例子体验一下两者的不同了！

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_12.png)](https://cenleiding.github.io/神经网络ANN/ANN_12.png)

　　首先，两者都是要将各个路径的偏微分值计算出来的 ，毕竟这是基础，在网络中就是**路径权重值** 。。。

　　**向前传播**：是一个**路径遍历求和**的过程，要求b的影响要遍历一次，要求a的影响又要遍历一次，**同一路径会被多次计算**。而神经网络的参数和路径如此之多，要是每进行一次梯度下降就要对每个参数进行一次路径遍历，这个计算了就爆炸了！！！

　　**反向传播**：是一个**一次性扩散**的过程，可以看到向后传播直接确定了结果对每个节点的影响能力，**一条路径只会被计算一次**，这个几乎没有什么计算量。

　　看完不经感叹，反向传播原来这么简单，又这么有用。。。

　　最后跟着[AGH](http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html)上的反向传播图理一遍整个过程。

　　首先我们有一个简单的神经网络，如下图：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_13.png)](https://cenleiding.github.io/神经网络ANN/ANN_13.png)

　　然后输入x1,x2获得输出y，并计算得出误差δ：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_14.png)](https://cenleiding.github.io/神经网络ANN/ANN_14.png)

　　有了整体误差，那么就要开始反向传播了，进而获得每个节点的输出误差：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_15.png)](https://cenleiding.github.io/神经网络ANN/ANN_15.png)

　　最后就是对权重的调整，η就是学习速率：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_16.png)](https://cenleiding.github.io/神经网络ANN/ANN_16.png)

　　这里要注意一下，传递误差δ只是代表节点输出值的偏导，而我们的目标是路径权重w的偏导，中间还需要过度一下。

## 优化器

所谓的优化器，实际上就是具体的实现参数更新的策略。像上面讲的梯度下降就是一种最直接、古老的优化器。接下来简单讲一讲几个常见优化器。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_42.gif)](https://cenleiding.github.io/神经网络ANN/ANN_42.gif)[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_43.gif)](https://cenleiding.github.io/神经网络ANN/ANN_43.gif)

　　常见优化算法的发展历程：GD => SGD => SGDM => AdaGrad => RRMSProp => Adam。

　　**▲ DG(Gradient Descent)：** 就是上面讲的原始梯度下降算法。其最大的问题在于：**一口气对整个数据集计算误差然后进行反向传播，而任务中的数据集往往很大，每次都对整个数据集计算误差会使得训练速度非常非常慢。**

　　**★ SGD(Stochastic Gradient Descent)：**随机梯度下降。就原始梯度下降算法的问题，随机梯度下降算法**每个批次都对参数进行更行**(实际上应该是Batch Gradient Descent,BGD更准确，SGD一般指每一个样本更新一次，也不知道为什么不太讲BGD而用SGD代替)。由于每次都是对一个批次的数据进行梯度下降，而非真实的数据分布，随机打乱抽取的数据会引入噪声，这形成了其**震荡的现象。** 为缓解这种现象，**实践中有必要随着时间逐渐降低学习率**。SGD最大的缺点就是**下降速度慢，存在震荡，停在局部最优解。**

　　**▲ SGDM(SGD with Momentum) ：** 动量SGD。**引入了一阶动量。** 所谓的动量其实就是之前几个时刻的梯度向量和的平均值。SGDM需要随时间调整β值，**使其一开始以当前梯度方向为准，慢慢的以积累动量为准，这样能够避免震荡现象，而且能够加快训练速度。**

　　**▲ AdaGrad：** 自适应梯度。**引入二阶动量。** 上面的这些优化器都需要手动调整学习率，如果优化器能够自动调整那就太棒了。所以引入这么一个策略：**对于经常更新的参数，认为已经积累了大量关于它的知识所以学习速率可以慢一些，而对于不太更新的参数，则希望学习速率可以大一些。** 而对于一个参数的历史更新频率则用二阶动量来度量，二阶动量是**至今为止所有梯度值的平方和。** 学习速率则与二阶动量成反比。**AdaGrad的问题在于二阶动量是递增的，学习速率会单调递减至0，可能使训练提前结束。**

　　**▲ RMSProp：**均方差传播。AdaGrad直接对所有梯度求平方和，导致学习率单调递减。对此RMSProp引入衰减因子，**简单来说就是距离越远的梯度影响越小，只关注一段时间内的梯度。** 这样就能避免二阶动量的持续积累，防止训练提前结束。

　　**★ Adam：** 上面那些策略的集成者，**使用一阶动量抑制震荡现象，使用二阶动量实现学习率的自适应调整。** 可以说是目前最流行的优化器了。但Adam仍然存在两个重要的缺陷：**1.可能不收敛。** Adam和RMSProp都采用时间窗口的二阶动量来调整学习速率，而如果遇到的数据变化较大会导致学习率的震荡，导致无法收敛。**2.可能错过全局最优解。**自适应学习率优化器的结果往往会稍差，因为自适应学习率算法可能会对前期出现的特征过拟合，后期出现的特征很难纠正前期的拟合效果。

**优化器选择：** 同一个问题，不同的优化器可能找到不同的答案。Adam优化器使用方便，收敛速度快。SGD则往往最终结果好，收敛速度慢。实际使用中，一般先用Adam看看大致结果如何，如果有需要再用SGD调优，当然如果很闲可以把几个优化器都试一试~~

## 激活函数的作用

> 首先，感谢一下[Daniel Godoy](https://towardsdatascience.com/@dvgodoy?source=post_header_lockup)大佬的独特的可视化解释，以及[机器之家](https://www.jqr.com/article/000161)的翻译（╮(╯▽╰)╭还是中文看的舒服）

　　激活函数是神经网络中必不可少的部分，每个神经节点都会调用激活函数，那么激活函数的作用是什么呢？

　　**引入非线性！** 当然，这是大家都知道的事，但它具体是怎么影响神经网络的呢？效果又是怎么样的呢？借助Godoy的可视化训练，可以让我们更为直接的感受激活函数的作用。

### sigmoid

　　最传统的激活函数。尽管今时今日，它的使用场景主要限制在**分类任务的输出层**。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_17.png)](https://cenleiding.github.io/神经网络ANN/ANN_17.png)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_18.png)](https://cenleiding.github.io/神经网络ANN/ANN_18.png)

　　如上图所示，sigmoid激活函数将输入值“压入”**区间(0, 1)**（和概率值的区间一致，这正是它在输出层中用于分类任务的原因）。由于sigmoid的区间在(0, 1)，**激活值以0.5为中心**，而不是以零为中心（正态分布输入通常以零为中心）。其**梯度**（gradient）峰值为0.25（当z = 0时），而当|z|达到5时，它的值已经很接近零了。

<iframe width="700" height="393" src="https://www.youtube.com/embed/4RoTHKKRXgE" allow="autoplay; encrypted-media" allowfullscreen="" style="width: 828px; height: 464.859px; position: absolute; top: 0px; left: 0px;"></iframe>

　　可以从视频中看出sigmoid激活函数成功分开了两条曲线，不过损失下降得比较缓慢，训练时间有显著部分停留在高原（plateaus）上。

### tanh

　　**双曲正切函数**tanh激活函数从sigmoid演进而来，和其前辈不同，其输出值的均值为零。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_19.png)](https://cenleiding.github.io/神经网络ANN/ANN_19.png)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_20.png)](https://cenleiding.github.io/神经网络ANN/ANN_20.png)

　　如上图所示，tanh激活函数“挤压”输入至**区间(-1, 1)** 。因此，**中心为零**，（某种程度上）激活值已经是下一层的正态分布输入了。至于梯度，它有一个大得多的峰值1.0（同样位于z = 0处），但它下降得更快，当|z|的值到达3时就已经接近零了。这是所谓**梯度消失（vanishing gradients）**问题背后的原因，这导致网络的训练进展变慢。

<iframe width="700" height="393" src="https://www.youtube.com/embed/PFNp8_V_Apg" allow="autoplay; encrypted-media" allowfullscreen="" style="width: 828px; height: 464.859px; position: absolute; top: 0px; left: 0px;"></iframe>

　　tanh激活函数以更快的速度达到了所有情形正确分类的点，而损失函数同样下降得更快（损失函数下降的时候），但它同样在高原上花了很多时间。

### ReLu

　　修正线性单元（Rectified Linear Units），简称**ReLU**，是寻常使用的激活函数**。**ReLU处理了两个前辈常见的**梯度消失**问题，同时也是计算梯度**最快**的激活函数。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_21.png)](https://cenleiding.github.io/神经网络ANN/ANN_21.png)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_22.png)](https://cenleiding.github.io/神经网络ANN/ANN_22.png)

　　如上图所示，**ReLU**是一头完全不同的野兽：它并不“挤压”值至某一区间——它只是**保留正值**，并将所有**负值转化为零**。

　　使用**ReLU**的积极方面是它的**梯度**要么是1（正值），要么是0（负值）——**再也没有梯度消失了！**这一模式使网络**更快收敛**。

　　另一方面，这一表现导致所谓的**“死亡神经元”**问题，也就是输入持续为负的神经元激活值总是为零。

<iframe width="700" height="393" src="https://www.youtube.com/embed/Ji_05nOFLE0" allow="autoplay; encrypted-media" allowfullscreen="" style="width: 828px; height: 464.859px; position: absolute; top: 0px; left: 0px;"></iframe>

**损失**从开始就保持**稳定下降**，直到**接近零**后才**进入平原**，花了**大约75%**的**tanh**训练时间就达成所有情形**正确分类!!**

### ↑上三种激活函数比较

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_23.png)](https://cenleiding.github.io/神经网络ANN/ANN_23.png)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_24.png)](https://cenleiding.github.io/神经网络ANN/ANN_24.png)

　　嗯。难怪ReLU激活函数现在这么常用。。。

### ELU

　　**指数线性单元ELU**融合了sigmoid和ReLU，具有左侧软饱性。其正式定义为：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_28.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_28.jpg)

　　右侧线性部分使得ELU能够缓解梯度消失，而左侧软饱和能够让ELU对输入变化或噪声更鲁棒。ELU的输出均值接近于零，所以**收敛速度更快**。实验中，ELU的收敛性质的确优于ReLU和PReLU。

### LReLu

　　**Leaky ReLU** 是给所有负值**赋予一个非零斜率(固定)**。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以表示为：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_26.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_26.jpg)

### PReLU

　　**参数化修正线性单元PReLU**是ReLU和LReLU的**改进版本** ，具有非饱和性：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_25.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_25.jpg)

　　与LReLU相比，PReLU中的负半轴**斜率a可学习而非固定**。原文献建议初始化a为0.25。与ReLU相比，PReLU收敛速度更快。因为PReLU的输出更接近0均值，使得SGD更接近natural gradient。证明过程参见[原文](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf?spm=5176.100239.blogcont55892.28.pm8zm1&file=He_Delving_Deep_into_ICCV_2015_paper.pdf)

### RReLU

　　**随机纠正线性单元RReLu**数学形式与PReLU类似，但RReLU是一种非确定性激活函数，其**参数是随机的** 。这种随机性类似于一种噪声，能够在一定程度上起到正则效果。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_27.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_27.jpg)

### Maxout

　　Maxout是ReLU的推广，其发生饱和是一个零测集事件（measure zero event）。正式定义为：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_29.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_29.jpg)

　　Maxout网络能够近似任意连续函数，且当w2,b2,…,wn,bn为0时，退化为ReLU。 其实，Maxout的思想在视觉领域存在已久。例如，在HOG特征里有这么一个过程：计算三个通道的梯度强度，然后在每一个像素位置上，仅取三个通道中梯度强度最大的数值，最终形成一个通道。这其实就是Maxout的一种特例。

　　Maxout能够缓解梯度消失，同时又规避了ReLU神经元死亡的缺点，但增加了参数和计算量。

### ↑以上Relu亲戚的比较

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_30.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_30.jpg)

　　其实ReLu的亲戚还有很多，左边什么形状都看得到~但**最常用的还是ReLU和ELU**。

## 权重初始化

　　提前感谢[Daniel Godoy](https://towardsdatascience.com/@dvgodoy?source=post_header_lockup)和[夏飞](https://www.leiphone.com/news/201703/3qMp45aQtbxTdzmK.html)文章的帮助。

　　首先为什么要进行权重初始化？直接赋0让机器自己去学习不行吗？
　　这当然不行，毕竟如果权重全为0，那反向传播时weight updata 也都为0了，也就是说根本无法学习！

　　那么我们应该怎么初始化权重？
　　常用的初始化方法有：

　　╋ **Random initialization**

　　╋ **Xavier / Glorot initialization**

　　╋ **He initialization**

　　╋ **Batch Normalization Layer**

### Random initialization

```
W = tf.Variable(np.random.randn(d1,d2))
```

> 最为**常用**的初始化方法。
>
> 参数呈**正太分布**。
>
> 适用于比较小的神经网络。

　　但是这种初始化在面对**层次比较多的网络**时尝尝会出现问题，主要是因为**梯度消失**和**梯度爆炸**。

　　首先拿已经被抛弃的**sigmoid激活函数为例**，权重初始化为σ=0.01运行结果如下图所示。可以看到出现了梯度消失的现象，这是因为在反向传播的过程中，是需要不断的乘以权重和f’(e)的，而权重初始化小于1且sigmoid的导数最大也只有0.25，那么误差在传递过程中越来越小，从而导致了梯度消失。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_31.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_31.jpg)

　　那么我们加大初始化时的标准差呢？可以看到情况变好了一点，但仍然不够。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_32.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_32.jpg)

　　那么我们再次加大正态分布的标准差。结果发现梯度消失倒是没了，但是激活函数的输出值几乎呈二值状态（因为权值太大导致Z值太大，而当|z|达到5时sigmoid函数基本上就是0或1了），这样一来激活函数就没什么用了。。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_33.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_33.jpg)

　　嗯，难道是因为sigmoid函数求导实在太小了(难怪被抛弃~)？
　　那么我们换成比较常见的**tanh激活函数**呢？

　　从下面3幅图可以看到，标准差为0.01时还是会出现梯度消失，标准差为1时不但激活函数输出值呈二值状态甚至出现了**梯度爆炸**的情况！但是当标准差为0.1时，终于找到了一个平衡点！看起来是那么的完美！

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_34.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_34.jpg)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_35.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_35.jpg)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_36.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_36.jpg)

　　可是0.1只是这里例子的特殊情况，平时初始化时我们又不可能自己去凑这个值，那么就有了**Xavier / Glorot 初始化**和 **He 初始化** 。

### Xavier / Glorot initialization

　　Glorot和Bengio设计的一种初始化方案，该方案试图保持前面列出的**获胜特性**，即**梯度**、**Z值**、**激活**在**所有层上相似**。换句话说，保持**所有层的方差相似**。具体证明见[Daniel Godoy](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)

> 适用于tanh激活函数
>
> 方差为sqrt(2/(fan_in+fan_out))的正太分布
>
> 或上下限为sqrt(6/(fan_in+fan_out))的均匀分布

fan_in，fan_out指的是当前权重输入层和输出层的节点数。**一般每一层的节点数都是一样的，及fan_in=fan_out**

```python
W = tf.Variable(np.random.randn(node_in, node_out)) * np.sqrt(1/node_in)
## tensorflow API
if type == 'xavier_uniform':
trueinitial = tf.contrib.layers.xavier_initializer(uniform=True, dtype=tf.float32)
elif type == 'xavier_normal':
trueinitial = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
W = tf.Variable(initializer=initial,shape=shape)
```

### He initialization

　　那么对于Relu激活函数也类似，**保证所有层上相似即可。**具体证明见[Daniel Godoy](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)

> 适用于ReLU激活函数
>
> 方差为sqrt(2/fan_in)的正太分布
>
> 或上下限为sqrt(6/fan_in)的均匀分布

```python
W = tf.Variable(np.random.randn(node_in,node_out)) * np.sqrt(2/node_in)
## tensorflow API
if type == 'he_normal':
trueinitial = tf.contrib.layers.variance_scaling_initializer(uniform=False, 
                                                             factor=2.0, 
                                                             mode='FAN_IN', 
                                                             dtype=tf.float32)
elif type == 'he_uniform':
trueinitial = tf.contrib.layers.variance_scaling_initializer(uniform=True, 
                                                             factor=2.0, 
                                                             mode='FAN_IN', 
                                                             dtype=tf.float32)
W = tf.Variable(initializer=initial,shape=shape)
```

### Batch Normalization Layer

　　前面两种方法都是通过计算巧妙的来保持每一层的相似。
　　而Batch Normalization是一种直接暴力的方法，既然目标是保持每一层的相似，也就是说我们想要的是在非线性激活函数之前，输出值应该有比较好的分布（例如高斯分布）。于是Batch Normalization将输出值强行做一次Gaussian Normalization和线性变换：

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_38.png)](https://cenleiding.github.io/神经网络ANN/ANN_38.png)

　　Batch Normalization中所有的操作都是平滑可导，这使得back propagation可以有效运行并学到相应的参数*γ*,*β*γ,β。需要注意的一点是Batch Normalization在training和testing时行为有所差别。**Training时μBμB和σBσB由当前batch计算得出；在Testing时μBμB和σBσB应使用Training时保存的均值或类似的经过处理的值，而不是由当前batch计算。**

需要注意的地方：

- 使用BN时不能是One Example SGD，因为需要通过多个样本计算均值和方差。
- BN是在激活函数之前进行的，**将输入转化成均值为0方差为1的比较标准的正态分布**。
- 如果只进行正太转化，那么会导致绝大多数点落入非线性函数的线性区内，虽然训练收敛会加快，但网络的表达能力会下降，所以BN在最后给每个神经元都添加了γ和β，将正态分布稍稍移动变换了一下，使得输入往非线性区移动了一下。而scale和shift就是BN需要学习的参数。让人感觉整个BN就是在寻找一个线性和非线性的一个平衡点，既能快速学习又避免导数太小。
- 推断时，直接用全局统计量进行正太化，即训练时保存的均值和方差。
- Relu也能使用BN。不过有些研究将BN放在Relu之前，而有些研究将BN放在Relu之后，而且都有效果。具体原因不清楚，根据实际场景可以都尝试一下。

```python
for i in range(0,num_layers -1)
trueW = tf.Variable(np.random.randn(node_in,node_out)) * np.sqrt(2/node_in)
truefc = tf.matmul(X,W)
truefc = tf.contrib.layers.batch_norm(fc, center=True, scale=True,is_training=True)
truefc = tf.nn.relu(fc)
```

## 梯度消失与爆炸

### 产生原因

> 根本因素–梯度下降
>
> 影响因素–激活函数,权重初始化

　　梯度消失和爆炸产生的**根本因素是因为神经网络使用了梯度下降算法**。在梯度下降算法中，误差会进行向前传播，在传播过程中误差会不断**乘以权重和激活函数的导数** 。那么如果这个乘积一直远小于1，则误差越传越小，最后趋于0，导致无法学习，这就是**梯度消失**。而反之，如果这个乘积一直远大于1，则误差越传越大，最后导致权重变化幅度太大，这就是**梯度爆炸**。而这个乘积的大小由**当前权重**和**激活函数导数**决定，而权重在初始化时往往是小于1的（如何初始化权重对网络也有很大影响！），那么激活函数的导数就变的极为的重要！

　　先看一下**sigmoid激活函数**： 导数最大值为1/4，且当输入达到5时，它的导数已经很接近零了。这就意味着乘积基本上是小于1/4的，那么**梯度消失**就很容易发生！(这也是为什么不在用sigmoid激活函数的主要原因)。

　　再看一下**tanh激活函数**：导数最大值为1，且当输入达到3时，导数就接近零了。那么也就意味着仍然会发生**梯度消失**，这使得用tanh激活函数的网络**训练会比较缓慢**。

　　最后看一下**ReLu激活函数**： 导数不是0就是1，那么梯度消失的问题基本上就没有了（虽然会出现死细胞的情况-当导数一直为0时），而且导数为1意味着梯度下降会比较快，所以现在神经网络多用Relu激活函数。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_31.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_31.jpg)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_36.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_36.jpg)

### 解决方案

> 1. 使用relu、leakrelu、elu等激活函数
> 2. 使用batchnorm
> 3. 梯度剪切、正则
> 4. 残差结构

#### 使用relu、leakrelu、elu等激活函数

　　上面也提到了这些激活函数可以有效的防止梯度爆炸和梯度消失问题。

#### 使用 batch_norm

　　关于batch_norm在**权重初始化**中就讲过，其主旨就是在每次Z值输出时，对其进行强行Gaussian变换，这样每一层的输出就会相似且不会出现相差过大的情况，这样就能够影响w不会浮动过大，从而使得误差在传导过程中不会变化太快。

```
for i in range(0,num_layers -1)
    W = tf.Variable(np.random.randn(node_in,node_out)) * np.sqrt(2/node_in)
    fc = tf.matmul(X,W)
    fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True,is_training=True)
    fc = tf.nn.relu(fc)
```

#### 梯度剪切、正则

　　**梯度剪切**这个方案主要是针**对梯度爆炸**提出的，其思想是设置一个梯度剪切阈值，然后更新梯度的时候，如果梯度超过这个阈值，那么就将其强制限制在这个范围之内。这可以防止梯度爆炸。

　　**权重正则化（weithts regularization）**可用于解决**梯度爆炸问题** 。比较常见的是L1正则，和L2正则，其就是在计算损失函数时加上了一个权重的‘惩罚’，这样能够有效的减缓权重的变化速度。当然这更多的用于防止**过拟合** 。

　　这两种方法都用于解决梯度爆炸问题，但实际上，在神经网络中更多的是梯度消失问题。

#### 残差结构

论文：[Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)

知乎链接：<https://zhuanlan.zhihu.com/p/31852747这里只简单介绍残差如何解决梯度的问题。>

　　事实上，就是残差网络的出现导致了image net比赛的终结，自从残差提出后，几乎所有的深度网络都离不开残差的身影，相比较之前的几层，几十层的深度网络，在残差网络面前都不值一提，残差可以很轻松的构建几百层，一千多层的网络而不用担心梯度消失过快的问题，原因就在于残差的捷径（shortcut）部分，其中残差单元如下图所示：**可以看出求导时一定是大于1的~**

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_39.png)](https://cenleiding.github.io/神经网络ANN/ANN_39.png)

## 过拟合与欠拟合

首先感谢[漫漫成长知乎文章](https://zhuanlan.zhihu.com/p/29707029)和[AI柠檬文章](https://blog.ailemon.me/2018/04/09/deep-learning-the-ways-to-solve-underfitting/)

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_40.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_40.jpg)

### **什么是过拟合,什么是欠拟合**？

　　**过拟合**： 学得的模型过于契合训练集，将训练集中的噪声也学了进去，这会导致在训练集中模型表现很好，但到测试集中就表现的不怎么样了。过拟合表现为**高方差(high Variance)**。

　　**欠拟合**： 设计的模型未能拟合训练数据。欠拟合表现为**高偏差(high Bias)** 。

### 怎么发现发生了过拟合、欠拟合

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_41.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_41.jpg)

　　**欠拟合**： 很简单的，训练集和测试集都表现很差。

　　**过拟合**： 训练集的loss已经比较小了，但测试集的loss还比较大。

### 如何解决欠拟合？

> 1. 提高神经网络复杂度
> 2. 更改参数初始化
> 3. 调整优化器和学习速率
> 4. 检查激活函数
> 5. 减小正则项其参数

#### 提高神经网络复杂度

　　如果怀疑神经网络的拟合能力不够，可以试试**用小数据量的数据进行训练，如果小数据量可行，但大数据量不可行，那么就说明神经网络的拟合能力不够！** 因为这意味着这个网络只能拟合大量样本的整体特征，或者少数样本的具体特征。这个时候就需要增加神经网络的复杂度了而且一般**增加网络的深度优于增加宽度！**

#### 更改参数初始化

　　在前面关于权重初始化的部分也讲了，初始化对网络的训练有很大的影响！因为**TensorFlow中默认用的glorot_uniform**初始化，有可能会导致学习很慢，梯度消失。所以可以换成**he_normal或xavier_normal**试试。

#### 调整优化器和学习速率

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_42.gif)](https://cenleiding.github.io/神经网络ANN/ANN_42.gif)[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_43.gif)](https://cenleiding.github.io/神经网络ANN/ANN_43.gif)

　　在神经网络中最常用的就是梯度下降(SGD)了，但是他有一个问题就是训练相对来说比较缓慢！所以可以尝试**先用Adam优化器快速收敛，最后再换成SGD进行最后的收敛**。 如果不想替换优化器，也可以**在SGD开始的时候使用较大的学习速率，然后逐渐减小学习速率。**

#### 检查激活函数

　　在前面关于激活函数也讲到了几个常用激活函数的优缺点。而**TensorFlow中默认使用的是ReLu激活函数，因此梯度消失的问题比较小**。 另外**尽量不要使用sigmoid激活函数，就算在最后一层也用softmax函数来代替！**

#### 减小正则项其参数 *λ*λ

　　前面在关于梯度爆炸的章节中提到了用正则化可以防止梯度爆炸和过拟合(使用正则操作是十分常见的)，但也产生了一个问题：**如果正则参数太大会抑制网络的学习，所以可以检查一下自己的正则参数是否太大了！**

### 如何解决过拟合？

> 1. 提前停止
> 2. 增大数据集
> 3. 正则化
> 4. dropout

#### 提前停止

　　提前停止(Early stopping)的具体做法是:**在每一个Epoch结束时,计算一下准确度，当准确度不再提高时，就停止训练。**而判断准确度是否不再提高可以用“No-improvement-in-n”策略：**在训练的过程中，记录到目前为止最好的准确度，当连续10次Epoch（或者更多次）没达到最佳准确度时，则可以认为准确度不再提高了，可以提前停止训练了。**

#### 增大数据集

　　往往发生过拟合时第一个想法就是，是不是我的数据集不够？因为有这么一句话**“有时候往往拥有更多的数据胜过一个好的模型”**。 因为只有数据集足够大才能覆盖尽可能全的情况，这样训练出来的模型才能更加全面，避免过拟合。

　　而增大数据集也不是说就是单纯的增加数据量，因为单纯的增加有时候不现实，代价大，数据可用性降低，一般可以用以下策略来增大数据集：

◇ 从数据源头采集更多数据

◇ 复制原有数据并加上随机噪声

◇ 重采样

◇ 根据当前数据集估计数据分布参数，使用该分布产生更多数据等

#### 正则化

　　这部分内容参考自[莫烦](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-09-l1l2regularization/#核心思想)大佬的讲解。

　　**正则化方法是指在进行目标函数或代价函数优化时，在目标函数或代价函数后面加上一个正则项，一般有L1正则与L2正则等。**

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_45.png)](https://cenleiding.github.io/神经网络ANN/ANN_45.png)

　　**正则项的目的是对网络的参数进行约束。** 因为当某一个参数项特别大时，相应的特征就会特别明显，这往往会导致过拟合，而我们并不希望完全的拟合，而只需要近似拟合就够了。那么可以认为我们不希望出现参数特别大的情况，希望**参数都趋于0**。这就是一个“消去网络棱角，变的中庸的过程”。。。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_46.png)](https://cenleiding.github.io/神经网络ANN/ANN_46.png)

　　于是就有了L1，L2正则。实际上很简单，就是在原有的**代价函数上加了一个惩罚项**。
[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_47.png)](https://cenleiding.github.io/神经网络ANN/ANN_47.png)

　　怎么理解这个惩罚项呢？
　　比如现在我们只有两个参数，**代价函数是上图中蓝色的线，惩罚项是上图中黄色的线。**则变化后的式子的值变为了<原点=>白点=>蓝色中心>的距离，这样一来白点就不会一个劲的往蓝色中点钻，这样就防止了过拟合！

**统一表达式：**[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_48.png)](https://cenleiding.github.io/神经网络ANN/ANN_48.png)参数λ控制正则化强弱。

**注意：实际中更多的使用L2正则！因为L1正则会产生稀疏性。**

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_51.png)](https://cenleiding.github.io/神经网络ANN/ANN_51.png)

　　第一幅图是正常的损失函数，第二幅图是加了L2的损失函数，第三幅图是加了L1的损失函数。可以很明显的看到L1造成稀疏性的原因：L1在0点有一个巨大的转折，如果原先的损失函数在0点的导数比较平缓，那么加了L1之后很容易称为一个极小值点（简单来说就是L1在0点太强势了）。而L2在0点的转折就很平缓，它只会将极小值点向0点靠。[参考[王赟 Maigo](https://www.zhihu.com/people/maigo)]

**TensorFlow 实现**

```
tf.contrib.layers.l1_regularizer(
    scale,       # 0.0则禁用
    scope=None   # 目标参数
)
tf.contrib.layers.l2_regularizer(
    scale,       # 0.0则禁用
    scope=None   # 目标参数
)
```

#### Dropout

　　Dropout 也是一种十分常用的防止过拟合的方法。

[![img](https://cenleiding.github.io/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9CANN/ANN_44.jpg)](https://cenleiding.github.io/神经网络ANN/ANN_44.jpg)

　　**该方法是在对网络进行训练时用的一种技巧（trick）**：在训练开始时，随机得删除一些（可以设定为一半，也可以为1/3，1/4等）隐藏层神经元，即认为这些神经元不存在，同时保持输入层与输出层神经元的个数不变，然后按照BP学习算法对ANN中的参数进行学习更新（虚线连接的单元不更新，因为认为这些神经元被临时删除了）。这样一次迭代更新便完成了。下一次迭代中，同样随机删除一些神经元，与上次不一样，做随机选择。这样一直进行瑕疵，直至训练结束。
**Dropout方法是通过修改ANN中隐藏层的神经元个数来防止ANN的过拟合。**

其思想和随机森林差不多，强行放弃部分特征，降低过拟合。需要注意的是在预测时要使用所有的神经节点。

```
tf.nn.dropout(
    x,   
    keep_prob,
    noise_shape=None,
    seed=None,
    name=None
)
```



<iframe name="easyXDM_default8569_provider" id="easyXDM_default8569_provider" src="https://embed.widgetpack.com/widget/xdm/index.html?xdm_e=https%3A%2F%2Fcenleiding.github.io&amp;xdm_c=default8569&amp;xdm_p=1" frameborder="0" style="color: rgb(85, 85, 85); font-family: Lato, &quot;PingFang SC&quot;, &quot;Microsoft YaHei&quot;, sans-serif; font-size: 16px; font-style: normal; font-variant-ligatures: normal; font-variant-caps: normal; font-weight: 400; letter-spacing: normal; orphans: 2; text-align: start; text-indent: 0px; text-transform: none; white-space: normal; widows: 2; word-spacing: 0px; -webkit-text-stroke-width: 0px; text-decoration-style: initial; text-decoration-color: initial; position: absolute !important; top: -2000px !important; left: 0px !important;"></iframe>