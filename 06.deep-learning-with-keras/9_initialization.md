class: middle, center

#Weight Initialization

####Shuai Wang

####USTC, August 12, 2019

Copyright (c) 2019 Shuai Wang. All rights reserved.

---

## 核心文献

### Understanding the difficulty of training deep feedforward neural networks

*Xavier Glorot, Yoshua Bengio, In AISTATS, 2010. DIRO, Universite de Montreal, Montreal, Quebec, Canada* 

**Abstract**

- Whereas before 2006 it appears that deep multi-layer neural networks were not successfully trained.
- Our objective here is to understand better why standard gradient descent from **random initialization** is doing so poorly with deep neural networks
- We first observe the influence of the non-linear activations functions. We find that the logistic sigmoid activation is unsuited for deep networks with random initialization because of its mean value, which can drive especially the top hidden layer into saturation. （饱和区造成梯度消失，not zero-centered造成梯度方向受限，更新缓慢）
- Based on these considerations, we **propose a new initialization scheme** that brings substantially faster convergence.

#### 1. Deep Neural Networks

- In order to learn the kind of complicated functions that can represent high-level abstractions (e.g. in vision, language, and other AI-level tasks), one may **need deep architectures**. 

- Our analysis is driven by investigative experiments to **monitor activations** (watching for saturation of hidden units) **and gradients, across layers and across training iterations**. 
- We also evaluate the effects on these of **choices of activation function** (with the idea that it might affect saturation) and **initialization procedure** (since unsupervised pre-training is a particular form of initialization and it has a drastic impact).

#### 2. Experimental Setting and Datasets

**2.1 Online Learning on an Infinite Dataset: Shapeset-3 × 2**

- Contains images of 1 or 2 two-dimensional objects (triangle, parallelogram, ellipse)
- The second object does not overlap with the first by more than fifty percent of its area
- The task is to predict the objects present. This therefore defines nine configuration classes.
- Need to discover in- variances over rotation, translation, scaling, object color, occlusion and relative position of the shapes. In parallel we need to extract the factors of variability that predict which object shapes are present.

.center[<img src=imgs/9_1_1.png width=300 >]

---

**2.2 Finite Datasets**

- MNIST digits dataset
- CIFAR-10
- Small-ImageNet

---

**2.3 Experimental Setting**

- We optimized feedforward neural networks with one to five hidden layers, with one thousand hidden units per layer, and with a softmax logistic regression for the output layer. 
- The cost function is the negative log-likelihood $−log P(y|x)$, where $(x, y)$ is the (input image, target class) pair. 
- The neural networks were optimized with stochastic back-propagation on mini-batches of size ten, i.e., the **average** $g$ of $\frac{\partial −log P(y|x)}{\partial \theta}$ was computed over 10 consecutive training pairs $(x, y)$ and used to update parameters $\theta$ in that direction, with $θ ← θ − \epsilon g$ . ($L =\frac{1}{N} \sum_i Li$, $\frac{\partial L}{\partial \theta} = \frac{1}{N}\sum_i \frac{\partial L_i}{\partial \theta}$, 也即多GPU数据并行时loss求和梯度取平均的原理)
- The learning rate $\epsilon$ is a hyper- parameter that is optimized based on validation set error after a large number of updates (5 million).

---

We varied the type of non-linear activation function in the hidden layers: 

- the sigmoid $1/(1 + e^{-x})$

- the hyperbolic tangent $tanh(x)$, 

- and a newly proposed activation function called the softsign, $x/(1 + |x|)$. The soft-sign is similar to the hyperbolic tangent (its range is -1 to 1) but its tails are quadratic polynomials rather than exponentials, i.e., it approaches its asymptotes much slower.

- We initialized the **biases to be 0** and the weights $W_{ij}$ at each layer with the following commonly used heuristic:
  $$
  W_{i,j} \sim U[-\frac{1}{\sqrt{n}},\frac{1}{\sqrt{n}}]
  $$
  where U is the uniform distribution and n is the size of the previous layer (the number of columns of W).
  $$
  E(w)=0\\
  Var(w) = \frac{1}{3n}
  $$
  

#### 3 Effect of Activation Functions and Saturation During Training

**3.1 Experiments with the Sigmoid**

> The sigmoid non-linearity has been already shown to slow down learning because of its none-zero mean that induces **important singular values in the Hessian** (LeCun et al., 1998b). (hessian矩阵条件数过大，会造成Zig path, Hession与神经网络参数更新关系待研究)

.center[<img src=imgs/9_1_2.png width=450 >]

- Mean and standard deviation (vertical bars) of the activation values (output of the sigmoid) during supervised learning, for the different hidden layers of a deep architecture. 
- The top hidden layer quickly saturates at 0 (slowing down all learning), but then slowly desaturates around epoch 100.

---

**3.2 Experiments with the Hyperbolic tangent**

> As discussed above, the hyperbolic tangent networks do not suffer from the kind of saturation behavior of the top hidden layer observed with sigmoid networks, because of its symmetry around 0. ???

.center[<img src=imgs/9_1_3.png width=450 >]

- **Top:** 98 percentiles (markers alone) and standard deviation (solid lines with markers) of the distribution of the activation values for the hyperbolic tangent networks in the course of learning. We see the first hidden layer saturating first, then the second, etc. (对称？)

---



**3.3 Experiments with the Softsign**

.center[<img src=imgs/9_1_3_2.png width=450 >]

- **Bottom:** 98 percentiles (markers alone) and standard deviation (solid lines with markers) of the distribution of activation values for the soft-sign during learning. Here the different layers saturate less and **do so together**.

---

Activation values normalized histogram at the end of learning, averaged across units of the same layer and across 300 test examples. 

.center[<img src=imgs/9_1_4.png height=300 >]

- **Top:** activation function is hyperbolic tangent, we see important saturation of the lower layers. 
- **Bottom:** activation function is soft-sign, we see many activation values around (-0.6,-0.8) and (0.6,0.8) where the units do not saturate but are non-linear.

---

#### 4 Studying Gradients and their Propagation 

**4.1 Effect of the Cost Function**

.center[<img src=imgs/9_1_5.png width=400 >]

- **Cross entropy** (black, surface on top) and **quadratic** (red, bottom surface) cost as a function of two weights (one at each layer) of a network with two layers, W1 respectively on the first layer and W2 on the second, output layer.
- The plateaus in the training criterion (as a function of the parameters) are less present with the log-likelihood cost function.

---

**4.2 Gradients at initialization**
**4.2.1 Theoretical Considerations and a New Normalized Initialization**

思路：计算每一层状态梯度与参数梯度的方差

目标：

.center[<img src=imgs/9_1_0.png width=500 >]

- $s^i$ 为第i层的状态值，$z^{i-1}$为第i-1层的激活值，用作第i层的输入；

---

首先：求第i层第k个神经元状态值的梯度与第i层参数矩阵第$l$行k列元素的梯度
$$
\begin{align}
\frac{\partial Cost}{\partial s^i_k} &= \frac{\partial Cost}{\partial \textbf{s}^{i+1}} .\frac{\partial \textbf{s}^{i+1}}{z^{i}_k}\frac{\partial z^{i}_k}{s^{i}_k} = f'(s^i_k)W^{i+1}_{(k,)}\frac{\partial Cost}{\partial \textbf{s}^{i+1}} \\
\frac{\partial Cost}{\partial w^i_{(l,k)}} &=\frac{\partial Cost}{\partial s^i_k} .\frac{\partial s^i_k}{\partial w^i_{(l,k)}}=z^{i-1}_l\frac{\partial Cost}{\partial s^i_k}
\end{align}
$$

- $s^i_k$ 与 $z^{i}_k$相关，$z^i_k$ 与W的第k行相乘进而与整个$\textbf{s}^{i+1}$相关
- $w^i_{(l,k)}$与$z^i$相乘与$s^i_k$ 相关
- . 为数量积，链式求导的向量形式



假设：

- For a dense artificial neural network using symmetric activation function $f$ with unit derivative at 0 (i.e. $f'(0) = 1$) (激活值z均值为0，进一步假设每一层所有z独立同分布)
- Consider the hypothesis that we are in a linear regime at the initialization $f'(s^i)=1$ 
- the weights are initialized independently （每一层所有参数w均值为0且独立同分布，且与z独立）and that the inputs features variances are the same ($= Var[x]$).

(注：以上三条根据使用激活函数的不同每一条都可能不满足)

---

**前向：计算每一层激活值的方差，目标是初始化之后，各层激活值的方差相同**
$$
\begin{align}
Var(z^i_l) &= Var(f(s^{i}_l)) = Var(s^{i}_l) = Var(z^{i-1}W^{i}_{(,l)}) \notag\\ 
           &= Var(\sum_{j=1}^{n^{i-1}}z^{i-1}_jW^{i}_{(j,l)}) =n^{i-1}Var(z^{i-1}w^{i}) \notag\\
           &= n^{i-1}Var(z^{i-1})Var(w^{i}) \notag\\
           &= n^{i-1}n^{i-2}Var(z^{i-2})Var(w^{i-1})Var(w^{i}) \notag\\
           &= n^{i-1}n^{i-2}...n^0Var(z^0)Var(w^1)....Var(z^{i-2})Var(w^{i-})Var(w^{i})... \notag\\
           &= Var(x) \prod_{i'=1}^{i} n^{i'-1}Var(w^{i'})
\end{align}
$$
**反向：对于一个d层的神经网络，计算各层状态梯度的方差及参数梯度的方差，目标是初始化后各层方差相同**
$$
\begin{align}
Var[\frac{\partial Cost}{\partial s^i_k}] 
&= Var(f'(s^i_k)W^{i+1}_{(k,)}\frac{\partial Cost}{\partial \textbf{s}^{i+1}})= Var(W^{i+1}_{(k,)}\frac{\partial Cost}{\partial s^{i+1}}) \notag \\
&=Var(\sum_{j=1}^{n^{i+1}}w^{i+1}\frac{\partial Cost}{\partial s_j^{i+1}}) = n^{i+1}Var(w^{i+1})Var(\frac{\partial Cost}{\partial s_j^{i+1}}) \notag \\
&= n^{i+1}Var(w^{i+1})n^{i+2}Var(w^{i+2})Var(\frac{\partial Cost}{\partial s_j^{i+2}})... \notag \\
&= Var(\frac{\partial Cost}{\partial s_j^{d}}) \prod_{i'=i+1}^{d}n^{i'}Var(w^{i'})
\end{align}
$$

$$
\begin{align}
Var[\frac{\partial Cost}{\partial w^i_{(l,k)}}] 
&=Var(z^{i-1}_l)Var(\frac{\partial Cost}{\partial s^i_k}) \notag \\
&= Var(x) \prod_{i'=1}^{i-1} n^{i'-1}Var(w^{i'})Var(\frac{\partial Cost}{\partial s_j^{d}}) \prod_{i'=i+1}^{d}n^{i'}Var(w^{i'}) \notag \\
&=Var(x) Var(\frac{\partial Cost}{\partial s_j^{d}})  \prod_{i'=1}^{i-1} n^{i'-1}Var(w^{i'}) \prod_{i'=i+1}^{d}n^{i'}Var(w^{i'}) 
\end{align}
$$

*Remark:*

- 由于定义方式不同，造成与原文对$w^i$标号不同，原文定义$i$从$0\sim d-1$,此处定义$1-d$,二者相差1，公式代表含义相同

---

From a forward-propagation point of view, to keep information flowing we would like that
$$
\forall (i,i'), Var(z^i) = Var(z^{i'})
$$
From a back-propagation point of view we would similarly like to have
$$
\forall(i,i'), Var[\frac{\partial Cost}{\partial s^i_k}] = Var[\frac{\partial Cost}{\partial s^{i'}_k}]
$$
These two conditions transform to:
$$
\forall i, n^iVar(w^{i+1}) = 1 \\
\forall i, n^{i+1}Var(w^{i+1})=1
$$
As a compromise between these two constraints, we might want to have
$$
\forall i, Var(w^{i+1})= \frac{2}{n^i+n^{i+1}}
$$
---

Note how both constraints are satisfied when all layers have the same width. If we also have the same initialization for the weights we could get the following interesting properties:
$$
\forall i, Var[\frac{\partial Cost}{\partial s^i_k}] =Var(\frac{\partial Cost}{\partial s_j^{d}}) [nVar(w)]^{d-i} \\


Var[\frac{\partial Cost}{\partial w^i_{(l,k)}}] 
=Var(x) Var(\frac{\partial Cost}{\partial s_j^{d}})[nVar(w)]^{d-1} 
$$

- We can see that the variance of the gradient on the weights is the same for all layers, but the variance of the back-propagated gradient might still **vanish or explode** as we consider deeper networks. 

The standard initialization that we have used (eq.1) gives rise to variance with the following property:
$$
nVar(w)=\frac{1}{3}
$$
where n is the layer size (assuming all layers of the same size). 

- This will cause the variance of the back-propagated gradient to be dependent on the layer (and decreasing). 

The normalization factor may therefore be important when initializing deep networks because of the multiplicative effect through layers, and we suggest the following initialization procedure to approximately satisfy our objectives of maintaining activation variances and back-propagated gradients variance as one moves up or down the network. We call it the **normalized initialization**:
$$
W \sim U[-\frac{\sqrt{6}}{n^j+n^{j+1}},\frac{\sqrt{6}}{n^j+n^{j+1}}]
$$

$$
V(w)= \frac{1}{12}（\frac{2\sqrt{6}}{n^j+n^{j+1}}）^2=\frac{2}{n^j+n^{j+1}}
$$

---

To empirically validate the above theoretical ideas, we have plotted some normalized histograms of activation values, weight gradients and of the back-propagated gradients at initialization with the two different initialization methods.

<img src=imgs/9_1_6.png width=200 > <img src=imgs/9_1_7.png width=200 ><img src=imgs/9_1_8.png width=200 >

**Remark:**

 We cannot use simple variance calculations in our theoretical analysis **because the weights values are not anymore independent of the activation values and the linearity hypothesis is also violated**



### 2.2 Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. ICCV 2015 Microsoft Research*

**Abstract**

- We propose a Parametric Rectified Linear Unit (PReLU) that generalizes the traditional rectified unit.
- We derive a robust initialization method that particularly considers the rectifier nonlinearities. 
- Based on our PReLU networks (PReLU-nets), we achieve 4.94% top-5 test error on the ImageNet 2012 classification dataset. To our knowledge, our result is the first to surpass human-level performance (5.1%, [22]) on this visual recognition challenge.

#### 1. Introduction

In the last few years, we have witnessed tremendous improvements in recognition performance, mainly due to advances in two technical directions: 

- building more powerful models,
  - increased complexity (e.g., increased depth [25, 29], enlarged width [33, 24], and the use of smaller strides [33, 24, 2, 25]), new nonlinear activations [21, 20, 34, 19, 27, 9], and sophisticated layer designs [29, 11].
- and designing effective strategies against overfitting. 
  - better generalization is achieved by effective regularization techniques [12, 26, 9, 31], aggressive data augmentation [16, 13, 25, 29], and large-scale data [4, 22].



#### 2. Approach 

In this section, we first present the PReLU activation function (Sec. 2.1). Then we derive our initialization method for deep rectifier networks (Sec. 2.2). Lastly we discuss our architecture designs (Sec. 2.3).
**2.1. Parametric Rectifiers**

**Definition:**
$$
f(y_i)=\left\{ 
             \begin{array}  \\  
             y_i, & y_i > 0 \\  
             a_iy_i & y_i \leq 0   
             \end{array}  
\right.
$$

- Here $y_i$ is the input of the nonlinear activation $f$ on the $i$**th channel**, and $a_i$ is a coefficient controlling the slope of the negative part. 
- The subscript $i$ in $a_i$ indicates that we allow the nonlinear activation to vary on different channels.

---



<img src=imgs/9_2_1.png width=300 >

- PReLU introduces a very small number of extra parameters. The number of extra parameters is equal to the total number of channels, which is negligible when considering the total number of weights.

- We also consider a channel-shared variant:
  $$
  f(y_i)=max(0,y_i) + \text{a}min(0,y_i)
  $$
  where the coefficient is shared by all channels of one layer. This variant **only introduces a single extra parameter into each layer**

---

**Optimization:**

The gradient of $a_i$ for one layer is:
$$
\frac{\partial \varepsilon}{\partial a_i} =\sum_{y_i}\frac{\partial \varepsilon}{\partial f(y_i)}\frac{{\partial f(y_i)}}{\partial a_i}
$$

- where $\varepsilon$ represents the objective function. The term $\frac{\partial \varepsilon}{\partial f(y_i)}$ is the gradient propagated from the deeper layer.

- The summation $\sum_{y_i}$ runs over all positions of the feature map. （我们也可以写成向量形式进行求导,将每个卷积核的所有输出写成向量）
  $$
  \frac{\partial \varepsilon}{\partial a_i} =\frac{\partial \varepsilon}{\partial \textbf{f}(y_i)}.\frac{{\partial \textbf{f}(y_i)}}{\partial a_i}
  $$

- For the channel-shared variant, the gradient of $a$ is
  $$
  \frac{\partial \varepsilon}{\partial a} =\sum_i \sum_{y_i}\frac{\partial \varepsilon}{\partial f(y_i)}\frac{{\partial f(y_i)}}{\partial a_i}
  $$
  where $\sum_i$ sums over all channels of the layer. 

- The gradient of the activation is given by:
  $$
  \frac{{\partial f(y_i)}}{\partial a_i} =\left\{ 
               \begin{array}  \\  
               0, & y_i > 0 \\  
               y_i & y_i \leq 0   
               \end{array}  
  \right.
  $$
  

---

We adopt the momentum method when updating $a_i$:
$$
\Delta a_i = \mu \Delta a_i + \epsilon \frac{\partial \varepsilon}{\partial a_i}
$$

- It is worth noticing that we do not use weight decay (l2 regularization) when updating $a_i$. A weight decay tends to push $a_i$ to zero, and thus biases PReLU toward ReLU. Even without regularization, the learned coefficients rarely have a magnitude larger than 1 in our experiments. 
- We use $a_i$ = 0.25 as the initialization throughout this paper

---

**Comparison Experiments：**

Table 1. A small but deep 14-layer model [10]. The learned coefficients of PReLU are also shown. For the channel-wise case, the average of {ai} over the channels is shown for each layer.

.center[<img src=imgs/9_2_t1.png height=300 >]

---

Table 2. Comparisons between ReLU and PReLU on the small model.

.center[<img src=imgs/9_2_t2.png width=300 >]

- First, the first conv layer (conv1) has coefficients (0.681 and 0.596) significantly greater than 0. 
  - As the filters of conv1 are mostly Gabor-like filters such as edge or texture detectors, the learned results show that **both positive and negative responses of the filters are respected**. 
  - We believe that this is a more economical way of exploiting low-level information, given the limited number of filters (e.g., 64).  (得到更多样的激活值)
- Second, for the channel-wise version, **the deeper conv layers in general have smaller coefficients**. 
  - This implies that the activations gradually become “**more nonlinear**” at increasing depths. 
  - In other words, the learned model tends to **keep more information in earlier stages** and becomes **more discriminative** in deeper stages.

---

**2.2. Initialization of Filter Weights for Rectifier**

- Rectifier networks are easier to train [8, 16, 34] compared with traditional sigmoid-like activation networks. But a bad initialization can still hamper the learning of a highly non-linear system.
- Glorot and Bengio [7] proposed to adopt a properly scaled uniform distribution for initialization. This is called **“Xavier” initialization** in [14]. Its derivation is **based on the assumption that the activations are linear.** This assumption is invalid for ReLU and PReLU.

**Basis：**
$$
E(x) = \int_{-\infin}^{+\infin}xf(x)dx \\
V(x) = E[(x-E(x))^2] = Ex^2 - E^2x \\
E(X+Y) = EX + EY
$$
当X， Y相互独立时：
$$
E(XY) = EXEY \\
V(X+Y) = V(X) + V(Y) \\
V(XY) = V(X)V(Y)+V(X)E^2Y+V(Y)E^2X
$$
---

**Forward Propagation Case:**

Our derivation mainly follows [7]. The central idea is to **investigate the variance of the responses in each layer**.

For a conv layer, a response is:
$$
\textbf{y}_l=\textbf{W}_l\textbf{x}_l+\textbf{b}_l
$$

- Here, $\textbf{x}$ is a $k^2c$-by-1 **vector** that represents co-located $k×k$ pixels in $c$ input channels. 

- $k$ is the spatial filter size of the layer. 

- With $n = k^2c$ denoting the number of **connections of a response**, $\textbf{W}$ is a $d$-by-$n$ matrix, where $d$ is the number of filters and **each row of W represents the weights of a filter**. 

- $\textbf{b}$ is a vector of biases, and $\textbf{y}$ is the response at **a pixel** of the output map(dx1). （All the pixels of the output map share a same $\textbf{W}_l$ for each layer）

- We use $l$ to index a layer. We have 
  $$
  \textbf{x}_l = f(\textbf{y}_{l−1})
  $$
   where $f$ is the activation. We also have $c_l = d_{l−1}$.

.center[<img src=imgs/9_2_0.png width=500 >]

---

**Assumption:**

- We let the initialized elements in $\textbf{W}_l$ be mutually independent and share the same distribution. 

- As in [7], we assume that the elements in $\textbf{x}_l$ are also mutually independent and share the same distribution, 

- and $\textbf{x}_l$ and $\textbf{W}_l$ are independent of each other. Then we have:
  $$
  \begin{align}
  Var(y_l) 
  &= Var(W_{l(j,)}\textbf{x}_l) \notag \\
  &=Var(w_{l_1}x_{l_1}+w_{l_2}x_{l_2}+\ldots+w_{l_{n_l}}x_{l_{n_l}}) \notag \\
  &=n_l Var(w_lx_l) \notag \\
  &=n_l [Var(w_l)Var(x_l)+Var(w_l)E^2(x_l)+Var(x_l)E^2(w_l)] \notag \\
  &=n_l [Var(w_l)(E(x_l^2)-E^(x_l))+Var(w_l)E^2(x_l)] \notag \\
  &=n_l Var(w_l)E(x_l^2)
  \end{align}
  $$

- where now $y_l$, $x_l$, and $w_l$ represent the random variables of each element in $\textbf{y}_l$, $\textbf{W}_l$, and $\textbf{xl respectively. 

- We let $w_l$ have zero mean.

- It is worth noticing that $E[x^2_l ] \neq Var[x_l]$ unless $x_l$ has zero mean. For the ReLU activation, $x_l = max(0, y_{l−1})$ and thus it does not have zero mean. **This will lead to a conclusion different from [7].**

---

If we let $w_{l−1}$ have a symmetric distribution around zero and $b_{l−1} = 0$, then $y_{l−1}=\sum w_{l-1}x_{l-1}$ has zero mean and has a symmetric distribution around zero. when f is ReLU
$$
\begin{align}
E(x^2_l)
&=E(f^2(y_{l-1}))=\int_{-\infin}^{+\infin}f^2(y_{l-1}) p(y_{l-1})dy_{l-1} \notag \\
&=\int_{-\infin}^{0}f^2(y_{l-1}) p(y_{l-1})dy_{l-1} + \int_{0}^{+\infin}f^2(y_{l-1}) p(y_{l-1})dy_{l-1}\notag \\
&= 0+ \int_{0}^{+\infin}y^2_{l-1}p(y_{l-1})dy_{l-1}\notag \\
&=\frac{1}{2}\int_{-\infin}^{+\infin}y^2_{l-1}p(y_{l-1})dy_{l-1} \notag \\
&=\frac{1}{2}E(y^2_{l-1}) \notag \\
&=\frac{1}{2}Var(y_{l-1})
\end{align}
$$
we obtain:
$$
\begin{align}
Var(y_l) =\frac{1}{2}n_l Var(w_l)Var(y_{l-1})
\end{align}
$$
---

With L layers put together, we have:
$$
\begin{align}
Var(y_L)&=\frac{1}{2}n_L Var(w_L)Var(y_{L-1}) \notag \\
&=\frac{1}{2}n_L Var(w_L)\frac{1}{2}n_{L-1} Var(w_{L-1})...\frac{1}{2}n_1 Var(w_2)Var(y_{1}) \notag \\
&=Var(y_1) \prod_{l=2}^L\frac{1}{2}n_l Var(w_l)
\end{align}
$$
***A proper initialization method should avoid reducing or magnifying the magnitudes of input signals exponentially.*** So we expect the above product to take a proper scalar (e.g., 1). A sufficient condition is:
$$
\forall l, \frac{1}{2}n_l Var(w_l)=1 
$$
This leads to a zero-mean Gaussian distribution whose standard deviation (std) is $\sqrt{2/n_l}$. **This is our way of initialization**. We also initialize b = 0.

Remark:

- For the first layer (l = 1), we should have $n_1Var[w_1] = 1$, because there is no ReLU applied on the input signal($Var(y_1)=n_1Var(w_1)E(x_1^2)$, $x_1$代表输入层，不再是激活值，因此不再展开). But the factor 1/2 does not matter if it just exists on one layer.

---

**Backward Propagation Case：**

For back-propagation, the gradient of a conv layer is computed by:
$$
\Delta \textbf{x}_l = \hat{\textbf{W}}_l \Delta \textbf{y}_l
$$

- Here we use $∆x$ and $∆y$ to denote gradients ( $\frac{∂\varepsilon}{∂x}$ and $\frac{∂\varepsilon}{∂y}$ for simplicity. 

- $∆y$ represents **k-by-k pixels in d channels**, and is reshaped into a $k^2d$-by-1 vector. We denote $\hat{n} = k^2d$. Note that $\hat{n} \neq n = k^2c$.
- $\hat{W}$ is a c-by $\hat{n}$ matrix where the **filters are rearranged in the way of back-propagation**. Note that $W$ and $\hat{W}$ can be **reshaped from each other**. 
- $∆x$ is a c-by-1 vector representing the gradient **at a pixel of this layer**. 
- 通俗解释：前向是$k^2$面积的像素，深度为c, 经过卷积操作映射到feature map上的1个像素，深度为d; 因为前向中的某一个像素，卷积后至多与feature map中 $k^2 d$个像素相关，因此反向时梯度的传播也是$k^2$面积的像素，深度为d, 传播到一个像素上，深度为c; 综上，得出$∆y$是$k^2d $x1, $∆x$是cx1; 因为卷积的权重共享，每一层只有一个W,共dxk^2c 个值，通过reshape,变为cxk^2d.具体元素对应位置可展开查看。

---

As above, we assume that $w_l$ and $∆y_l$ are independent of each other, then $∆x_l$ has zero mean for all l, when $wl$ is initialized by a symmetric distribution around zero.

In back-propagation,
$$
\begin{align}
\Delta y_l 
&= \frac{\partial \varepsilon}{\partial x_{l+1}}\frac{{\partial x_{l+1}}}{\partial y_l} \notag \\
&=\Delta x_{l+1} f'(y_l) 
\end{align}
$$
For the ReLU case, $f'(y_l)$ is zero or one, and their probabilities are equal. ($E(f'(y_l))=1/2, Var(f'(y_l))=1/4$). We assume that f'(yl) and ∆xl+1 are independent of each other.
$$
\begin{align}
E(\Delta y_l) 
&=E(\Delta x_{l+1}) E(f'(y_l))=0  \\ 

E((\Delta y_l)^2)
&=Var(\Delta y_l) \notag \\
&=Var(\Delta x_{l+1} f'(y_l)) \notag \\
&=Var(\Delta x_{l+1})Var( f'(y_l)) + Var(\Delta x_{l+1})E^2(f'(y_l))+Var(f'(y_l))E^2(\Delta x_{l+1})\notag \\
&=\frac{1}{2}Var(\Delta x_{l+1}) \notag \\
\end{align}
$$

$$
\begin{align}
Var(\Delta x_l)
&=\hat{n_l}Var(w_l)Var(\Delta y_l) \notag \\
&=\frac{1}{2}\hat{n_l}Var(w_l)Var(\Delta x_{l+1}) \notag \\
\end{align}
$$

---

The scalar 1/2 in both Eqn.(12) and Eqn.(8) is the result of ReLU, though the derivations are different. With L layers put together, we have:
$$
\begin{align}
Var(\Delta x_2)
&=\hat{n}Var(w_l)Var(\Delta y_l) \notag \\
&=Var(\Delta x_{L+1})\prod_{l=2}^L \frac{1}{2}\hat{n_l}Var(w_l) \notag \\
\end{align}
$$
We consider a sufficient condition that the gradient is not exponentially large/small:
$$
 \frac{1}{2}\hat{n_l}Var(w_l) =1
$$
The only difference between this equation and Eqn.(10) is that $\hat{n_l} = k_l^2 d_l$ while $n_l = k_l^2 c_l = k_l^2d_{l−1}$. Eqn.(14) results in a zero-mean Gaussian distribution whose std is $\sqrt{2/\hat{n_l}}$.





## 1.3 总结

<https://blog.csdn.net/mzpmzk/article/details/79839047>

<https://www.twblogs.net/a/5c947a10bd9eee35fc15e7d3>

<https://blog.csdn.net/JNingWei/article/details/78835390>

<https://blog.csdn.net/VictoriaW/article/details/73000632>

<https://blog.csdn.net/VictoriaW/article/details/73166752>

实践：

<https://towardsdatascience.com/why-default-cnn-are-broken-in-keras-and-how-to-fix-them-ce295e5e5f2>


