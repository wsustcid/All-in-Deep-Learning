## 概率

### 期望

在概率论和统计学中，数学期望(mean) （简称期望，或称均值）是实验中每次可能结果发生的概率乘以其结果的总和，反映的是随机变量平均取值的大小。

*Note: 算术平均值可以看做是期望的一种特殊形式，即所有数的取值都已经确定，概率均为1/n。*

大数定律规定：随着重复次数接近无穷大，数值的算术平均值几乎肯定的收敛于期望值

#### 离散型：

**定义**：

如果随机变量只取得有限个值或无穷能按一定次序一一列出，其值域为一个或若干个有限或无限区间，这样的随机变量称为离散型随机变量。

**算法：**

离散型随机变量的一切可能的取值 $x_i$ 与对应的概率 $p(x_i)$ 乘积之和称为该离散型随机变量的数学期望 (若该求和绝对收敛)，记为 $E(x)$。它是简单算术平均的一种推广，类似加权平均。具体来说，

离散型随机变量X的取值为$x_1, x_2, x_3, ...$ ，$p(x_1), p(x_2), p(x_3), ...$ 为$x$对应取值的概率，可理解为数据$x_i$出现的频率$f(x_i)$，则：
$$
E(X) = \sum_{k=1}^{\infin} x_k p_k
$$

#### 连续型

**定义：**

若随机变量$X$的分布函数$F(X)$ 可表示成一个非负可积函数$f(x)$的积分，则称$X$为连续型随机变量，$f(x)$称为$X$的概率密度函数（分布密度函数）。

**算法：**

设连续型随机变量$X$的概率密度函数为$f(x)$, 若积分绝对收敛，则其期望为
$$
E(x) = \int_{-\infin}^{+\infin}xf(x)dx
$$
数学期望$E(x)$完全由随机变量X的概率分布确定，若X服从某一分布，也称$E(X)$为这一分布的数学期望。

**定理：**

若随机变量$Y$符合函数$Y=g(x)$，且$\int_{-\infin}^{+\infin}g(x)f(x)dx$ 绝对收敛，则有：
$$
E(Y) = E(g(X)) =\int_{-\infin}^{+\infin}g(x)f(x)dx
$$
该定理的意义在于：我们在求$E(Y)$时不需要算出$Y$的分布律或者概率分布，只要利用$X$的分布律或概率分布即可。

上述定理还可以推广到两个或以上随机变量的函数情况：

设$Z$是随机变量$X, Y$的函数$Z=g(X, Y)$， $Z$是一个一维随机变量，二维随机变量$(X,Y)$的概率密度为$f(x,y)$，则有：
$$
E(Z) = E[g(X,Y)] =\int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}g(x,y)f(x,y)dxdy
$$

#### 性质

设c是一个常数，$X$ 和 $Y$ 是两个随机变量，

1. $E(c) = c$
2. $E(cX) = cE(X)$
3. $E(X+Y)= E(X) + E(Y)$
4. 当$X$和$Y$相互独立时，$E(XY)=E(X)E(Y)$ 

**证明：**

Note:这里仅对连续的情况加以证明，离散的情况仅需要将积分改为求和即可。

1. $c$的值是确定的，常数的平均数还是其本身

2. $$
   E(cX) = \int_{-\infin}^{+\infin} cxf(x)dx = c \int_{-\infin}^{+\infin} xf(x)dx = cE(X)
   $$

3. 设二维随机变量$(X,Y)$的概率密度函数为$f(x,y)$
   $$
   \begin{align}
   E(X+Y) &=\int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}(x+y)f(x,y)dxdy \notag\\
          &= \int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin} xf(x,y)dxdy +
   \int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}yf(x,y)dxdy \notag\\
          &= \int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}xf(x)dx + \int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}y)f(y)dy \notag \\
          &= E(X) + E (Y)
   \end{align}
   $$
   
   
   
4. 若$X$ 和 $Y$相互独立，其边缘概率密度函数为$f_X(x)$, $f_Y(y)$, 有 $f(x,y)=f_X(x)f_Y(y)$
   $$
   \begin{align}
   E(XY) &=\int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin}xyf(x,y)dxdy \notag\\
         &= \int_{-\infin}^{+\infin}\int_{-\infin}^{+\infin} xyf_X(x)f_Y(y)dxdy \notag\\
         &= \int_{-\infin}^{+\infin}xf_X(x)dx\int_{-\infin}^{+\infin}yf_Y(y)dy \notag\\
         &= E(X)E (Y)
   \end{align}
   $$
   

### 方差



方差_百度百科
https://baike.baidu.com/item/%E6%96%B9%E5%B7%AE

矩阵求导术（上） - 知乎
https://zhuanlan.zhihu.com/p/24709748

矩阵求导术（下） - 知乎
https://zhuanlan.zhihu.com/p/24863977

矩阵的TR迹的相关性质 - qq_33166535的博客 - CSDN博客
https://blog.csdn.net/qq_33166535/article/details/53318272

aisha 的笔记本 - Microsoft OneNote Online
https://onedrive.live.com/redir?resid=E3BE1385BE069B7F%21406&page=Edit&wd=target%28%E6%95%B0%E7%90%86%E5%9F%BA%E7%A1%80.one%7C34db0c5d-47b7-47f7-8fa8-1f669c97aba1%2F%E5%AF%BC%E6%95%B0%E5%81%8F%E5%AF%BC%E6%A2%AF%E5%BA%A6%7Cb75219c6-2c93-4950-a5b5-b47f7c5dd794%2F%29



## 优化

**符号定义：**

标量：小写字母，如 $x$

向量：小写加粗，如 $\mathbf{x}$

矩阵：大写加粗，如 $\mathbf{X}$



**引入:**

首先看优化中一个很常见的例子
$$
\text{min} f
$$
如果这个函数 $f$ 是只含有一个变量 $x$ 的一元函数，如$f(x) = x^2$，它的图像我们很容易画出来，

<src 

当然，导数为0的点就是极值点；

但如果是二元函数呢，如 $f(x)=x_1^2 + x_2^2$，图像我们也可以画出来，



你可能想，多元函数求极值



拉格朗日乘子法



首先，这里需要用到求导



其次，我们为了计算或书写的方便，经常要把多变量写成矩阵的形式，因此更难计算



但偏导为0，是极值点吗



求极值还有数值解法，拉格朗日展开，一阶二阶，hession矩阵

多元函数中的导数变成了梯度，梯度为0的点就是极值点。但到底什么叫做梯度，为什么梯度为0的点就是极值点呢？

比如，这个函数$f$往往非常复杂，<https://www.cnblogs.com/liuxuanhe/p/9245344.html>





[https://zh.wikipedia.org/wiki/%E9%93%BE%E5%BC%8F%E6%B3%95%E5%88%99#%E5%A4%9A%E5%85%83%E5%A4%8D%E5%90%88%E5%87%BD%E6%95%B0%E6%B1%82%E5%AF%BC%E6%B3%95%E5%88%99](https://zh.wikipedia.org/wiki/链式法则#多元复合函数求导法则)