# 1. 基础概念

## 1.1 卷积

译自Tim Dettmers的[Understanding Convolution in Deep Learning](http://timdettmers.com/2015/03/26/convolution-deep-learning/)，以及知乎相关回答

**什么是卷积：**

- 你可以把卷积想象成一种混合信息的手段。想象一下装满信息的两个桶，我们把它们倒入一个桶中并且通过某种规则搅拌搅拌。也就是说卷积是一种混合两种信息的流程。
- 卷积也可以形式化地描述，事实上，它就是一种数学运算，跟减加乘除没有本质的区别。虽然这种运算本身很复杂，但它非常有助于简化更复杂的表达式。在物理和工程上，卷积被广泛地用于化简等式

**我们如何对图像应用卷积：**

当我们在图像上应用卷积时，我们在两个维度上执行卷积——水平和竖直方向。我们混合两桶信息：第一桶是输入的图像，由三个矩阵构成——RGB三通道，其中每个元素都是0到255之间的一个整数。第二个桶是卷积核（kernel），单个浮点数矩阵。可以将卷积核的大小和模式想象成一个搅拌图像的方法。卷积核的输出是一幅修改后的图像，在深度学习中经常被称作feature map。

边缘检测卷积核的效果

![convolution.png](http://wx2.sinaimg.cn/large/006Fmjmcly1fdwjpji6qtj30dw05d0t8.jpg)



这是怎么做到的呢，我们现在演示一下如何通过卷积来混合这两种信息。一种方法是从输入图片中取出一个与卷积核大小相同的区块——这里假设图片为100×100，卷积核大小为3×3，那么我们取出的区块大小就是3×3——然后对每对相同位置的元素执行乘法后求和（不同于矩阵乘法，却类似向量内积，这里是两个相同大小的矩阵的“点乘”）。乘积的和就生成了feature map中的一个像素。当一个像素计算完毕后，移动一个像素取下一个区块执行相同的运算。当无法再移动取得新区块的时候对feature map的计算就结束了。这个流程可以用如下的动画演示：



**为什么机器学习中图像卷积有用**

- 图像中可能含有很多我们不关心的噪音
  - 如果你想要区分衣服的式样，那么衣服的颜色就不那么重要了；另外像商标之类的细节也不那么重要。最重要的可能是衣服的外形。一般来讲，女装衬衫的形状与衬衣、夹克和裤子的外观非常不同。如果我们过滤掉这些多余的噪音，那我们的算法就不会因颜色、商标之类的细节分心了。我们可以通过卷积轻松地实现这项处理。
  - 通过索贝尔边缘检测滤波器（与上上一幅图类似）去掉了图像中除了边缘之外的所有信息——这也是为什么卷积应用经常被称作滤波而卷积核经常被称作滤波器（更准确的定义在下面）的原因。由边缘检测滤波器生成的feature map对区分衣服类型非常有用，因为只有外形信息被保留下来。

![autoencoder_fashion_features_and_results.png](http://wx4.sinaimg.cn/large/006Fmjmcly1fdwkhjv1xhj30ks0q60z9.jpg)



- 再进一步：有许多不同的核可以产生多种feature map，比如锐化图像（强调细节），或者模糊图像（减少细节），并且每个feature map都可能帮助算法做出决策（一些细节，比如衣服上有3个纽扣而不是两个，可能可以区分一些服饰）。

使用这种手段——读入输入、变换输入、然后把feature map喂给某个算法——被称为特征工程。特征工程非常难，很少有资料帮你上手。特征工程这么难的原因是，对每种数据每种问题，有用的特征都是不同的：图像类任务的特征可能对时序类任务不起作用；即使两个任务都是图像类的，也很难找出相同的有效特征，因为视待识别的物体的不同，有用的特征也不同。这非常依赖经验。所以特征工程对新手来讲特别困难。不过对图像而言，是否可以利用卷积核自动找出某个任务中最适合的特征？

**进入卷积神经网络**

卷积神经网络就是干这个的。不同于刚才使用固定数字的卷积核，我们赋予参数给这些核，参数将在数据上得到训练。随着卷积神经网络的训练，这些卷积核为了得到有用信息，在图像或feature map上的过滤工作会变得越来越好。这个过程是自动的，称作**特征学习**。特征学习自动适配新的任务：我们只需在新数据上训练一下自动找出新的过滤器就行了。这是卷积神经网络如此强大的原因——不需要繁重的特征工程了！

通常卷积神经网络并不学习单一的核，而是同时学习多层级的多个核。比如一个32x16x16的核用到256×256的图像上去会产生32个241×241的feature map。所以自动地得到了32个有用的新特征。这些特征可以作为下个核的输入。一旦学习到了多级特征，我们简单地将它们传给一个全连接的简单的神经网络，由它完成分类。这就是在概念上理解卷积神经网络所需的全部知识了（池化也是个重要的主题，但还是在另一篇博客中讲吧）。



**卷积定理**

要理解卷积，不得不提convolution theorem，它将时域和空域上的复杂卷积对应到了频域中的元素间简单的乘积。这个定理非常强大，在许多科学领域中得到了广泛应用。卷积定理也是快速傅里叶变换算法被称为20世纪最重要的算法之一的一个原因。

![convolution-theorem1.png](http://wx3.sinaimg.cn/large/006Fmjmcly1fdwlpw0mh0j311s06g3yw.jpg)

第一个等式是一维连续域上两个连续函数的卷积；第二个等式是二维离散域（图像）上的卷积。这里![latex (1).png](http://wx2.sinaimg.cn/large/006Fmjmcly1fdwlrkpb39j300n00n09p.jpg)指的是卷积，![latex (2).png](http://wx2.sinaimg.cn/large/006Fmjmcly1fdwlrynlggj300s00p084.jpg)指的是傅里叶变换，![latex (3).png](http://wx2.sinaimg.cn/large/006Fmjmcly1fdwlsb9rlvj301j00s09v.jpg)表示傅里叶逆变换，![latex (4).png](http://wx2.sinaimg.cn/large/006Fmjmcly1fdwlssoxu7j301q00y0g3.jpg)是一个正规化常量。这里的“离散”指的是数据由有限个变量构成（像素）；一维指的是数据是一维的（时间），图像则是二维的，视频则是三维的。

为了更好地理解卷积定理，我们还需要理解数字图像处理中的傅里叶变换。

**快速傅里叶变换**

快速傅里叶变换是一种将时域和空域中的数据转换到频域上去的算法。**傅里叶变换用一些正弦和余弦波的和来表示原函数**。必须注意的是，傅里叶变换一般涉及到复数，也就是说一个实数被变换为一个具有实部和虚部的复数。通常虚部只在一部分领域有用，比如将频域变换回到时域和空域上；而在这篇博客里会被忽略掉。你可以在下面看到一个信号（一个以时间为参数的有周期的函数通常称为信号）是如何被傅里叶变换的：

![fourier_transform_time_and_frequency_domains.gif](http://wx4.sinaimg.cn/large/006Fmjmcly1fdwm3am0yag30dw0b4tji.gif)

红色是时域，蓝色为频域

**傅里叶域上的图像**

![fourier-transforms.png](http://wx4.sinaimg.cn/large/006Fmjmcly1fdwmdi9fo2j30f90fja9w.jpg)

我们如何想象图片的频率呢？想象一张只有两种模式的纸片，现在把纸片竖起来顺着线条的方向看过去，就会看到一个一个的亮点。这些以一定间隔分割黑白部分的波就代表着**频率**。

- 在频域中，低频率更接近中央而高频率更接近边缘。
- 频域中高强度（亮度、白色）的位置代表着原始图像亮度改变的方向。这一点在接下来这张图与其对数傅里叶变换（对傅里叶变换的实部取对数，这样可以减小像素亮度的差别，便于观察更广的亮度区域）中特别明显：

![fourier_direction_detection.png](http://wx4.sinaimg.cn/large/006Fmjmcly1fdwmn64lk2j30w40spwr4.jpg)

我们马上就可以发现傅里叶变换包含了关于物体朝向的信息。如果物体被旋转了一个角度，从图像像素上可能很难判断，但从频域上可以很明显地看出来。

这是个很重要的启发，基于傅里叶定理，我们知道卷积神经网络在频域上检测图像并且捕捉到了物体的方向信息。于是卷积神经网络就比传统算法更擅长处理旋转后的图像（虽然还是比不上人类）。



**频率过滤与卷积**

为什么卷积经常被描述为过滤，为什么卷积核经常被称为过滤器呢？通过下一个例子可以解释：

![filtered-image1.png](http://wx3.sinaimg.cn/large/006Fmjmcly1fdwmsfhyphj30sc07oq4h.jpg)

如果我们对图像执行傅里叶变换，并且乘以一个圆形（背景填充黑色，也就是0），我们可以过滤掉所有的高频值（它们会成为0，因为填充是0）。注意过滤后的图像依然有条纹模式，但图像质量下降了很多——这就是jpeg压缩算法的工作原理（虽然有些不同但用了类似的变换），我们变换图形，然后只保留部分频率，最后将其逆变换为二维图片；压缩率就是黑色背景与圆圈的比率。

我们现在将圆圈想象为一个卷积核，然后就有了完整的卷积过程——就像在卷积神经网络中看到的那样。要稳定快速地执行傅里叶变换还需要许多技巧，但这就是基本理念了。

现在我们已经理解了卷积定理和傅里叶变换，我们可以将这些理念应用到其他科学领域，以加强我们对深度学习中的卷积的理解。





从数学上讲，卷积就是一种运算。

某种运算，能被定义出来，至少有以下特征：

- 首先是抽象的、符号化的
- 其次，在生活、科研中，有着广泛的作用

比如加法：

- ![a+b](https://www.zhihu.com/equation?tex=a%2Bb) ，是抽象的，本身只是一个数学符号
- 在现实中，有非常多的意义，比如增加、合成、旋转等等

卷积，是我们学习高等数学之后，新接触的一种运算，因为涉及到积分、级数，所以看起来觉得很复杂。

**1 卷积的定义**

我们称 ![(f*g)(n)](https://www.zhihu.com/equation?tex=%28f%2Ag%29%28n%29) 为 ![f,g](https://www.zhihu.com/equation?tex=f%2Cg) 的卷积

其连续的定义为：

![\displaystyle (f*g)(n)=\int _{-\infty }^{\infty }f(\tau )g(n-\tau )d\tau \\](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+%28f%2Ag%29%28n%29%3D%5Cint+_%7B-%5Cinfty+%7D%5E%7B%5Cinfty+%7Df%28%5Ctau+%29g%28n-%5Ctau+%29d%5Ctau+%5C%5C)

其离散的定义为：

![\displaystyle (f*g)(n)=\sum _{\tau =-\infty }^{\infty }{f(\tau )g(n-\tau )}\\](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+%28f%2Ag%29%28n%29%3D%5Csum+_%7B%5Ctau+%3D-%5Cinfty+%7D%5E%7B%5Cinfty+%7D%7Bf%28%5Ctau+%29g%28n-%5Ctau+%29%7D%5C%5C)

这两个式子有一个共同的特征：

![img](https://pic1.zhimg.com/80/v2-d3df01f12b869d431c65f97ad307508f_hd.jpg)

这个特征有什么意义？

我们令 ![x=\tau ,y=n-\tau ](https://www.zhihu.com/equation?tex=x%3D%5Ctau+%2Cy%3Dn-%5Ctau+) ，那么 ![x+y=n](https://www.zhihu.com/equation?tex=x%2By%3Dn) 就是下面这些直线：

![img](https://pic3.zhimg.com/50/v2-8be52f6bada3f7a21cebfc210d2e7ea0_hd.gif)

如果遍历这些直线，就好比，把毛巾沿着角卷起来：

![img](https://pic1.zhimg.com/50/v2-1d0c819fc7ca6f8da25435da070a2715_hd.jpg)

此处受到 [荆哲：卷积为什么叫「卷」积？](https://zhihu.com/question/54677157/answer/141245297) 答案的启发。

只看数学符号，卷积是抽象的，不好理解的，但是，我们可以通过现实中的意义，来**习惯**卷积这种运算，正如我们小学的时候，学习加减乘除需要各种苹果、糖果来帮助我们习惯一样。

我们来看看现实中，这样的定义有什么意义。

**2 离散卷积的例子：丢骰子**

我有两枚骰子, 把这两枚骰子都抛出去, 求两枚骰子加起来为4的概率是多少？

这里问题的关键是，两个骰子加起来要等于4，这正是卷积的应用场景。

我们把骰子各个点数出现的概率表示出来：

![img](https://pic2.zhimg.com/80/v2-4763fd548536b21640d01d3f8a59c546_hd.jpg)

那么，两枚骰子点数加起来为4的情况有：

![img](https://pic1.zhimg.com/80/v2-a67a711702ce48cd7632e783ae0a1f42_hd.jpg)



![img](https://pic2.zhimg.com/80/v2-d6ff10bf39c46397ab2bebb971d4b58c_hd.jpg)



![img](https://pic3.zhimg.com/80/v2-0cdabcc04398ea723aa6e47e05072e5c_hd.jpg)

因此，两枚骰子点数加起来为4的概率为：

![f(1)g(3)+f(2)g(2)+f(3)g(1)\\](https://www.zhihu.com/equation?tex=f%281%29g%283%29%2Bf%282%29g%282%29%2Bf%283%29g%281%29%5C%5C)

符合卷积的定义，把它写成标准的形式就是：（把卷积运算变成了简单的乘积）

![\displaystyle (f*g)(4)=\sum _{m=1}^{3}f(4-m)g(m)\\](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+%28f%2Ag%29%284%29%3D%5Csum+_%7Bm%3D1%7D%5E%7B3%7Df%284-m%29g%28m%29%5C%5C)

**3 连续卷积的例子：做馒头**

楼下早点铺子生意太好了，供不应求，就买了一台机器，不断的生产馒头。

假设馒头的生产速度是 ![f(t)](https://www.zhihu.com/equation?tex=f%28t%29) ，那么一天后生产出来的馒头总量为：

![\int _{0}^{24}f(t)dt\\](https://www.zhihu.com/equation?tex=%5Cint+_%7B0%7D%5E%7B24%7Df%28t%29dt%5C%5C)

馒头生产出来之后，就会慢慢腐败，假设腐败函数为 ![g(t)](https://www.zhihu.com/equation?tex=g%28t%29) ，比如，10个馒头，24小时会腐败：

![10*g(t)\\](https://www.zhihu.com/equation?tex=10%2Ag%28t%29%5C%5C)

想想就知道，第一个小时生产出来的馒头，一天后会经历24小时的腐败，第二个小时生产出来的馒头，一天后会经历23小时的腐败。

如此，我们可以知道，一天后，馒头总共腐败了：

![\int _{0}^{24}f(t)g(24-t)dt\\](https://www.zhihu.com/equation?tex=%5Cint+_%7B0%7D%5E%7B24%7Df%28t%29g%2824-t%29dt%5C%5C)

这就是连续的卷积。

**4 图像处理**

**4.1 原理**

有这么一副图像，可以看到，图像上有很多噪点：

![img](https://pic3.zhimg.com/80/v2-8d161328acd72d035e461c0b89b753e5_hd.jpg)

高频信号，就好像平地耸立的山峰：

![img](https://pic1.zhimg.com/80/v2-294698966c5a833cd750df70c0a00c21_hd.jpg)

看起来很显眼。

平滑这座山峰的办法之一就是，把山峰刨掉一些土，填到山峰周围去。用数学的话来说，就是把山峰周围的高度平均一下。

平滑后得到：

![img](https://pic1.zhimg.com/80/v2-83b24e8ed70f17df6bc3b921ebe6276c_hd.jpg)

**4.2 计算**

卷积可以帮助实现这个平滑算法。

有噪点的原图，可以把它转为一个矩阵：

![img](https://pic3.zhimg.com/80/v2-8dd14775ab8c91a09507f52e44f347f3_hd.jpg)

然后用下面这个平均矩阵（说明下，原图的处理实际上用的是正态分布矩阵，这里为了简单，就用了算术平均矩阵）来平滑图像：

![g=\begin{bmatrix} \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\ \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\ \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \end{bmatrix}\\](https://www.zhihu.com/equation?tex=g%3D%5Cbegin%7Bbmatrix%7D+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%5C%5C+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%5C%5C+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%26+%5Cfrac%7B1%7D%7B9%7D+%5Cend%7Bbmatrix%7D%5C%5C)

记得刚才说过的算法，把高频信号与周围的数值平均一下就可以平滑山峰。

比如我要平滑 ![a_{1,1}](https://www.zhihu.com/equation?tex=a_%7B1%2C1%7D) 点，就在矩阵中，取出 ![a_{1,1}](https://www.zhihu.com/equation?tex=a_%7B1%2C1%7D) 点附近的点组成矩阵 ![f](https://www.zhihu.com/equation?tex=f) ，和 ![g](https://www.zhihu.com/equation?tex=g) 进行卷积计算后，再填回去：

![img](https://pic2.zhimg.com/80/v2-5ee9a99988137a42d1067deab36c4e51_hd.jpg)

要注意一点，为了运用卷积， ![g](https://www.zhihu.com/equation?tex=g) 虽然和 ![f](https://www.zhihu.com/equation?tex=f) 同维度，但下标有点不一样：

![img](https://pic1.zhimg.com/80/v2-779d4e972dc557be55e6131edbb8db9f_hd.jpg)

我用一个动图来说明下计算过程：

![img](https://pic3.zhimg.com/50/v2-c658110eafe027eded16864fb6a28f46_hd.gif)

写成卷积公式就是：

![\displaystyle (f*g)(1,1)=\sum _{k=0}^{2}\sum _{h=0}^{2}f(h,k)g(1-h,1-k)\\](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+%28f%2Ag%29%281%2C1%29%3D%5Csum+_%7Bk%3D0%7D%5E%7B2%7D%5Csum+_%7Bh%3D0%7D%5E%7B2%7Df%28h%2Ck%29g%281-h%2C1-k%29%5C%5C)

要求 ![c_{4,5}](https://www.zhihu.com/equation?tex=c_%7B4%2C5%7D) ，一样可以套用上面的卷积公式。

这样相当于实现了 ![g](https://www.zhihu.com/equation?tex=g) 这个矩阵在原来图像上的划动（准确来说，下面这幅图把 ![g](https://www.zhihu.com/equation?tex=g) 矩阵旋转了180度，目的是将卷积运算原本不是对应位置乘积之和，变成对应位置乘积之和，方便观察：

![img](https://pic1.zhimg.com/50/v2-15fea61b768f7561648dbea164fcb75f_hd.gif)

对卷积的困惑

卷积这个概念，很早以前就学过，但是一直没有搞懂。教科书上通常会给出定义，给出很多性质，也会用实例和图形进行解释，但究竟为什么要这么设计，这么计算，背后的意义是什么，往往语焉不详。作为一个学物理出身的人，一个公式倘若倘若给不出结合实际的直观的通俗的解释（也就是背后的“物理”意义），就觉得少了点什么，觉得不是真的懂了。

教科书上一般定义函数 ![f,g](https://www.zhihu.com/equation?tex=f%2Cg) 的卷积 ![f*g(n)](https://www.zhihu.com/equation?tex=f%2Ag%28n%29) 如下：

连续形式：

![(f*g)(n)=\int_{-\infty }^{\infty}f(\tau )g(n-\tau)d\tau](https://www.zhihu.com/equation?tex=%28f%2Ag%29%28n%29%3D%5Cint_%7B-%5Cinfty+%7D%5E%7B%5Cinfty%7Df%28%5Ctau+%29g%28n-%5Ctau%29d%5Ctau)

离散形式：

![(f*g)(n)=\sum_{\tau=-\infty }^{\infty}f(\tau)g(n-\tau)](https://www.zhihu.com/equation?tex=%28f%2Ag%29%28n%29%3D%5Csum_%7B%5Ctau%3D-%5Cinfty+%7D%5E%7B%5Cinfty%7Df%28%5Ctau%29g%28n-%5Ctau%29)

并且也解释了，先对g函数进行翻转，相当于在数轴上把g函数从右边褶到左边去，也就是卷积的“卷”的由来。

然后再把g函数平移到n，在这个位置对两个函数的对应点相乘，然后相加，这个过程是卷积的“积”的过程。

这个只是从计算的方式上对公式进行了解释，从数学上讲无可挑剔，但进一步追问，为什么要先翻转再平移，这么设计有何用意？还是有点费解。

在知乎，已经很多的热心网友对卷积举了很多形象的例子进行了解释，如卷地毯、丢骰子、打耳光、存钱等等。读完觉得非常生动有趣，但过细想想，还是感觉有些地方还是没解释清楚，甚至可能还有瑕疵，或者还可以改进（这些后面我会做一些分析）。

带着问题想了两个晚上，终于觉得有些问题想通了，所以就写出来跟网友分享，共同学习提高。不对的地方欢迎评论拍砖。。。

明确一下，这篇文章主要想解释两个问题：

\1. 卷积这个名词是怎么解释？“卷”是什么意思？“积”又是什么意思？

\2. 卷积背后的意义是什么，该如何解释？

## 考虑的应用场景

为了更好地理解这些问题，我们先给出两个典型的应用场景：

\1. 信号分析

一个输入信号*f(t)*，经过一个线性系统（其特征可以用单位冲击响应函数*g(t)*描述）以后，输出信号应该是什么？实际上通过卷积运算就可以得到输出信号。

\2. 图像处理

输入一幅图像*f(x,y)*，经过特定设计的卷积核*g(x,y)*进行卷积处理以后，输出图像将会得到模糊，边缘强化等各种效果。

### 对卷积的理解

对卷积这个名词的理解：**所谓两个函数的卷积，本质上就是先将一个函数翻转，然后进行滑动叠加。**

在连续情况下，叠加指的是对两个函数的乘积求积分，在离散情况下就是加权求和，为简单起见就统一称为叠加。

整体看来是这么个过程：

翻转——>滑动——>叠加——>滑动——>叠加——>滑动——>叠加.....

多次滑动得到的一系列叠加值，构成了卷积函数。

卷积的“卷”，指的的函数的翻转，从 *g(t)* 变成 *g(-t)* 的这个过程；同时，“卷”还有滑动的意味在里面（吸取了网友[李文清](https://www.zhihu.com/people/li-wen-qing-25-49)的建议）。如果把卷积翻译为“褶积”，那么这个“褶”字就只有翻转的含义了。

卷积的“积”，指的是积分/加权求和。

有些文章只强调滑动叠加求和，而没有说函数的翻转，我觉得是不全面的；有的文章对“卷”的理解其实是“积”，我觉得是张冠李戴。

对卷积的意义的理解：

\1. 从“积”的过程可以看到，我们得到的叠加值，是个全局的概念。以信号分析为例，卷积的结果是不仅跟当前时刻输入信号的响应值有关，也跟过去所有时刻输入信号的响应都有关系，考虑了对过去的所有输入的效果的累积。在图像处理的中，卷积处理的结果，其实就是把每个像素周边的，甚至是整个图像的像素都考虑进来，对当前像素进行某种加权处理。所以说，“积”是全局概念，或者说是一种“混合”，把两个函数在时间或者空间上进行混合。

\2. 那为什么要进行“卷”？直接相乘不好吗？我的理解，进行“卷”（翻转）的目的其实是施加一种约束，它指定了在“积”的时候以什么为参照。在信号分析的场景，它指定了在哪个特定时间点的前后进行“积”，在空间分析的场景，它指定了在哪个位置的周边进行累积处理。

## 举例说明

下面举几个例子说明为什么要翻转，以及叠加求和的意义。

### 例1：信号分析

如下图所示，输入信号是 *f(t)* ，是随时间变化的。系统响应函数是 *g(t)* ，图中的响应函数是随时间指数下降的，它的物理意义是说：如果在 *t*=0 的时刻有一个输入，那么随着时间的流逝，这个输入将不断衰减。换言之，到了 *t*=T时刻，原来在 *t*=0 时刻的输入*f*(0)的值将衰减为*f*(0)*g*(T)。



![img](https://pic1.zhimg.com/80/v2-59c8bcf17c24119810ad3071b960f1ba_hd.jpg)

考虑到信号是连续输入的，也就是说，每个时刻都有新的信号进来，所以，最终输出的是所有之前输入信号的累积效果。如下图所示，在T=10时刻，输出结果跟图中带标记的区域整体有关。其中，f(10)因为是刚输入的，所以其输出结果应该是f(10)g(0)，而时刻t=9的输入f(9)，只经过了1个时间单位的衰减，所以产生的输出应该是 f(9)g(1)，如此类推，即图中虚线所描述的关系。这些对应点相乘然后累加，就是T=10时刻的输出信号值，这个结果也是f和g两个函数在T=10时刻的卷积值。



![img](https://pic1.zhimg.com/80/v2-de38ad49f9a1c99dafcc5d0a7fcac2ef_hd.jpg)

显然，上面的对应关系看上去比较难看，是拧着的，所以，我们把g函数对折一下，变成了g(-t)，这样就好看一些了。看到了吗？这就是为什么卷积要“卷”，要翻转的原因，这是从它的物理意义中给出的。



![img](https://pic4.zhimg.com/80/v2-5d5ca564c8f0eaba9cd9865a9c944fbb_hd.jpg)

上图虽然没有拧着，已经顺过来了，但看上去还有点错位，所以再进一步平移T个单位，就是下图。它就是本文开始给出的卷积定义的一种图形的表述：



![img](https://pic4.zhimg.com/80/v2-847a8d7c444508862868fa27f2b4c129_hd.jpg)

所以，在以上计算T时刻的卷积时，要维持的约束就是： *t+ (T-t) = T* 。这种约束的意义，大家可以自己体会。

### 例2：丢骰子

在本问题 [如何通俗易懂地解释卷积](https://www.zhihu.com/question/22298352/answer/228543288)？中排名第一的 [马同学](https://www.zhihu.com/people/matongxue)在中举了一个很好的例子（下面的一些图摘自马同学的文章，在此表示感谢），用丢骰子说明了卷积的应用。

要解决的问题是：有两枚骰子，把它们都抛出去，两枚骰子点数加起来为4的概率是多少?



![img](https://pic1.zhimg.com/80/v2-a5238ad7dfccc0f2645e1c34c10a19c9_hd.jpg)

分析一下，两枚骰子点数加起来为4的情况有三种情况：1+3=4， 2+2=4, 3+1=4

因此，两枚骰子点数加起来为4的概率为：

![img](https://pic4.zhimg.com/80/v2-e5a05d465fc32d4e880f7d77e811fb7e_hd.jpg)

写成卷积的方式就是：

![\displaystyle (f*g)(4)=\sum _{m=1}^{3}f(4-m)g(m)\\](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+%28f%2Ag%29%284%29%3D%5Csum+_%7Bm%3D1%7D%5E%7B3%7Df%284-m%29g%28m%29%5C%5C)

在这里我想进一步用上面的翻转滑动叠加的逻辑进行解释。

首先，因为两个骰子的点数和是4，为了满足这个约束条件，我们还是把函数 g 翻转一下，然后阴影区域上下对应的数相乘，然后累加，相当于求自变量为4的卷积值，如下图所示：



![img](https://pic4.zhimg.com/80/v2-c6d14a16dee215b2d6b9e020aefd2542_hd.jpg)

进一步，如此翻转以后，可以方便地进行推广去求两个骰子点数和为 *n* 时的概率，为*f* 和 *g*的卷积 *f\*g(n)*，如下图所示：

![img](https://pic4.zhimg.com/80/v2-860cdc53a489be168e9a12845c7eadc4_hd.jpg)

由上图可以看到，函数 *g* 的滑动，带来的是点数和的增大。这个例子中对f和g的约束条件就是点数和，它也是卷积函数的自变量。有兴趣还可以算算，如果骰子的每个点数出现的概率是均等的，那么两个骰子的点数和n=7的时候，概率最大。

### 例3：图像处理

还是引用知乎问题 [如何通俗易懂地解释卷积](https://www.zhihu.com/question/22298352/answer/228543288)？中 [马同学](https://www.zhihu.com/people/matongxue)的例子。图像可以表示为矩阵形式（下图摘自马同学的文章）：



![img](https://pic3.zhimg.com/80/v2-8dd14775ab8c91a09507f52e44f347f3_hd.jpg)

对图像的处理函数（如平滑，或者边缘提取），也可以用一个g矩阵来表示，如：

![img](https://pic4.zhimg.com/80/v2-c9844a1d908a5792ebfe3d54e89aa52f_hd.jpg)

注意，我们在处理平面空间的问题，已经是二维函数了，相当于：

![f(x,y)=a_{x,y}](https://www.zhihu.com/equation?tex=f%28x%2Cy%29%3Da_%7Bx%2Cy%7D)![g(x,y)=b_{x,y}](https://www.zhihu.com/equation?tex=g%28x%2Cy%29%3Db_%7Bx%2Cy%7D)

那么函数f和g的在（u，v）处的卷积 ![f*g(u,v)](https://www.zhihu.com/equation?tex=f%2Ag%28u%2Cv%29) 该如何计算呢？



![img](https://pic2.zhimg.com/80/v2-29b46dc4d83fb10239888227fceac1a3_hd.jpg)

首先我们在原始图像矩阵中取出（u,v）处的矩阵：

![f=\begin{bmatrix} &a_{u-1,v-1} &a_{u-1,v} &a_{u-1,v+1}\\ &a_{u,v-1} &a_{u,v} &a_{u,v+1} \\ &a_{u+1,v-1} &a_{u+1,v} &a_{u+1,v+1} \end{bmatrix}](https://www.zhihu.com/equation?tex=f%3D%5Cbegin%7Bbmatrix%7D+%26a_%7Bu-1%2Cv-1%7D+%26a_%7Bu-1%2Cv%7D+%26a_%7Bu-1%2Cv%2B1%7D%5C%5C+%26a_%7Bu%2Cv-1%7D+%26a_%7Bu%2Cv%7D+%26a_%7Bu%2Cv%2B1%7D+%5C%5C+%26a_%7Bu%2B1%2Cv-1%7D+%26a_%7Bu%2B1%2Cv%7D+%26a_%7Bu%2B1%2Cv%2B1%7D+%5Cend%7Bbmatrix%7D)

然后将图像处理矩阵翻转（延x轴和y轴两个方向翻转），如下：

![g^{'}=\begin{bmatrix} &b_{1,1} &b_{1,0} &b_{1,-1}\\ &b_{0,1} &b_{0,0} &b_{0,-1} \\ &b_{-1,1} &b_{-1,0} &b_{-1,-1} \end{bmatrix}](https://www.zhihu.com/equation?tex=g%5E%7B%27%7D%3D%5Cbegin%7Bbmatrix%7D+%26b_%7B1%2C1%7D+%26b_%7B1%2C0%7D+%26b_%7B1%2C-1%7D%5C%5C+%26b_%7B0%2C1%7D+%26b_%7B0%2C0%7D+%26b_%7B0%2C-1%7D+%5C%5C+%26b_%7B-1%2C1%7D+%26b_%7B-1%2C0%7D+%26b_%7B-1%2C-1%7D+%5Cend%7Bbmatrix%7D)

计算卷积时，就可以用 ![f](https://www.zhihu.com/equation?tex=f) 和 ![g^{'}](https://www.zhihu.com/equation?tex=g%5E%7B%27%7D) 的内积：

![f*g(u,v)=a_{u-1,v-1} \times b_{1,1} + a_{u-1,v} \times b_{1,0} +a_{u-1,v} \times b_{1,0} ](https://www.zhihu.com/equation?tex=f%2Ag%28u%2Cv%29%3Da_%7Bu-1%2Cv-1%7D+%5Ctimes+b_%7B1%2C1%7D+%2B+a_%7Bu-1%2Cv%7D+%5Ctimes+b_%7B1%2C0%7D+%2Ba_%7Bu-1%2Cv%7D+%5Ctimes+b_%7B1%2C0%7D+)

![ + a_{u,v-1} \times b_{0,1} + a_{u,v} \times b_{0,0} + a_{u,v+1} \times b_{0,-1}](https://www.zhihu.com/equation?tex=+%2B+a_%7Bu%2Cv-1%7D+%5Ctimes+b_%7B0%2C1%7D+%2B+a_%7Bu%2Cv%7D+%5Ctimes+b_%7B0%2C0%7D+%2B+a_%7Bu%2Cv%2B1%7D+%5Ctimes+b_%7B0%2C-1%7D)

![ + a_{u+1,v-1} \times b_{-1,1} + a_{u+1,v} \times b_{-1,0} + a_{u+1,v+1} \times b_{-1,-1}](https://www.zhihu.com/equation?tex=+%2B+a_%7Bu%2B1%2Cv-1%7D+%5Ctimes+b_%7B-1%2C1%7D+%2B+a_%7Bu%2B1%2Cv%7D+%5Ctimes+b_%7B-1%2C0%7D+%2B+a_%7Bu%2B1%2Cv%2B1%7D+%5Ctimes+b_%7B-1%2C-1%7D)

请注意，以上公式有一个特点，做乘法的两个对应变量a,b的下标之和都是（u,v），其目的是对这种加权求和进行一种约束。这也是为什么要将矩阵g进行翻转的原因。

以上计算的是（u,v）处的卷积，延x轴或者y轴滑动，就可以求出图像中各个位置的卷积，其输出结果是处理以后的图像（即经过平滑、边缘提取等各种处理的图像）。

再深入思考一下，在算图像卷积的时候，我们是直接在原始图像矩阵中取了（u,v）处的矩阵，为什么要取这个位置的矩阵，本质上其实是为了满足以上的约束。因为我们要算（u，v）处的卷积，而g矩阵是3x3的矩阵，要满足下标跟这个3x3矩阵的和是（u,v），只能是取原始图像中以（u，v）为中心的这个3x3矩阵，即图中的阴影区域的矩阵。

推而广之，如果如果g矩阵不是3x3，而是6x6，那我们就要在原始图像中取以（u，v）为中心的6x6矩阵进行计算。由此可见，这种卷积就是把原始图像中的相邻像素都考虑进来，进行混合。相邻的区域范围取决于g矩阵的维度，维度越大，涉及的周边像素越多。而矩阵的设计，则决定了这种混合输出的图像跟原始图像比，究竟是模糊了，还是更锐利了。

比如说，如下图像处理矩阵将使得图像变得更为平滑，显得更模糊，因为它联合周边像素进行了平均处理：

![g=\begin{bmatrix} &\frac{1}{9} &\frac{1}{9} &\frac{1}{9}\\ &\frac{1}{9} &\frac{1}{9} &\frac{1}{9} \\ &\frac{1}{9} &\frac{1}{9} &\frac{1}{9} \end{bmatrix}](https://www.zhihu.com/equation?tex=g%3D%5Cbegin%7Bbmatrix%7D+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D%5C%5C+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D+%5C%5C+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D+%26%5Cfrac%7B1%7D%7B9%7D+%5Cend%7Bbmatrix%7D)

而如下图像处理矩阵将使得像素值变化明显的地方更为明显，强化边缘，而变化平缓的地方没有影响，达到提取边缘的目的：

![g=\begin{bmatrix} &-1 &-1 &-1\\ &-1 &9 &-1 \\ &-1 &-1 &-1 \end{bmatrix}](https://www.zhihu.com/equation?tex=g%3D%5Cbegin%7Bbmatrix%7D+%26-1+%26-1+%26-1%5C%5C+%26-1+%269+%26-1+%5C%5C+%26-1+%26-1+%26-1+%5Cend%7Bbmatrix%7D)

## 对一些解释的不同意见

上面一些对卷积的形象解释，如知乎问题[卷积为什么叫「卷」积？](https://www.zhihu.com/question/54677157)中 [荆哲](https://www.zhihu.com/people/jing-zhe-511) ，以及问题 [如何通俗易懂地解释卷积](https://www.zhihu.com/question/22298352/answer/228543288)？中 [马同学](https://www.zhihu.com/people/matongxue) 等人提出的如下比喻：

![img](https://pic1.zhimg.com/80/v2-fd98e1ae48d582b127853ea3832f329c_hd.jpg)



![img](https://pic1.zhimg.com/80/v2-bc4ae0b5fbf526796f5c204ed71018cc_hd.jpg)

其实图中“卷”的方向，是沿该方向进行积分求和的方向，并无翻转之意。因此，这种解释，并没有完整描述卷积的含义，对“卷”的理解值得商榷。

## 一些参考资料

《数字信号处理（第二版）》程乾生，北京大学出版社

《信号与系统引论》 郑君里，应启珩，杨为理，高等教育出版社

[编辑于 2019-04-19](https://www.zhihu.com/question/22298352/answer/637156871)

赞同 799

分享



[![张俊博](https://pic1.zhimg.com/1be9d5c0ccc6b73c724ed50812dc2350_xs.jpg)](https://www.zhihu.com/people/jimbozhang)

[张俊博](https://www.zhihu.com/people/jimbozhang)



[夏飞](https://www.zhihu.com/people/feixia586)

、

[王赟 Maigo](https://www.zhihu.com/people/maigo)

 

等

 

有那么麻烦吗？
**不推荐用“反转/翻转/反褶/对称”等解释卷积。好好的信号为什么要翻转？**导致学生难以理解卷积的物理意义。
这个其实非常简单的概念，国内的大多数教材却没有讲透。

直接看图，不信看不懂。以离散信号为例，连续信号同理。

已知![x[0] = a, x[1] = b, x[2]=c](https://www.zhihu.com/equation?tex=x%5B0%5D+%3D+a%2C+x%5B1%5D+%3D+b%2C+x%5B2%5D%3Dc)

![img](https://pic4.zhimg.com/80/153fd3e7911d486edaf0475afb1e54b3_hd.jpg)



已知![y[0] = i, y[1] = j, y[2]=k](https://www.zhihu.com/equation?tex=y%5B0%5D+%3D+i%2C+y%5B1%5D+%3D+j%2C+y%5B2%5D%3Dk)

![img](https://pic1.zhimg.com/80/c47d9d7f7a29c491782bf7b1baea3f8e_hd.jpg)



下面通过演示求![x[n] * y[n]](https://www.zhihu.com/equation?tex=x%5Bn%5D+%2A+y%5Bn%5D)的过程，揭示卷积的物理意义。

第一步，![x[n]](https://www.zhihu.com/equation?tex=x%5Bn%5D)乘以![y[0]](https://www.zhihu.com/equation?tex=y%5B0%5D)并平移到位置0：

![img](https://pic1.zhimg.com/80/91f5eff235013ac729c44e98b3a537d0_hd.jpg)

第二步，



乘以



并平移到位置1：

![img](https://pic4.zhimg.com/80/67c05239b05f671766b9df9393026f2c_hd.jpg)

第三步，



乘以



并平移到位置2：

![img](https://pic4.zhimg.com/80/c34e839a49c6b616c57bde3c3dbbd67d_hd.jpg)

最后，把上面三个图叠加，就得到了



：

![img](https://pic3.zhimg.com/80/4ce6cdcc28b10aca73db3f877d86ca02_hd.jpg)

简单吧？无非是

平移（没有反褶！）、叠加。



====================================================

从这里，可以看到卷积的重要的物理意义是：一个函数（如：单位响应）在另一个函数（如：输入信号）上的**加权叠加。**

重复一遍，这就是卷积的意义：**加权叠加**。

对于线性时不变系统，如果知道该系统的单位响应，那么将单位响应和输入信号求卷积，就相当于把输入信号的各个时间点的单位响应 加权叠加，就直接得到了输出信号。

通俗的说：
**在输入信号的每个位置，叠加一个单位响应，就得到了输出信号。**
这正是单位响应是如此重要的原因。

**在输入信号的每个位置，****叠****加一个单位响应，就得到了输出信号。**
这正是单位响应是如此重要的原因。

在输入信号的每个位置，

叠

加一个单位响应，就得到了输出信号。

这正是单位响应是如此重要的原因。