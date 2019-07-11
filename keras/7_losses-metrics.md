# 1. Losses

## 1.1 内置损失函数

损失函数（或称目标函数、优化评分函数）**是编译模型时所需的两个参数之一**。

**用法：**

- 你可以传递一个现有的损失函数名
- 或者一个 TensorFlow/Theano 符号函数。**该符号函数为每个数据点返回一个标量，实际的优化目标是所有数据点的输出数组的平均值。**
  - *也就是说，符号函数计算的并不是$L$, 而是每一个$L_i$, 返回的是由所有$L_i$ 组成的一维张量（如果y是多维的，各维度误差计算公式不变，最后沿其维度轴取均值）。评价函数也是如此*。

传入函数名：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

传入符号函数：

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

***Remark:***

- 注意将其与评价指标 `metrics` 区分，训练模型仅仅需要损失函数和优化器参数就够了，优化算法根据损失函数值来反向传播更新参数；
- 但损失函数值有时并不能直观的反应网络模型的训练效果，因此我们需要使用恰当的评价指标（如准确率等）来直观显示训练过程的效果变化，或用来控制训练何时停止。

### 1.1.1 回归问题损失函数

#### 1. MSE: Mean Squared Error (均方误差/L2损失)

$$
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_{true}-y_{pred})^2
$$

适用：

- 如果目标变量的分布是高斯的，从数学上来讲，在最大似然估计的参考框架下，这是最优的损失函数 

优缺点：

- 导数连续使得优化过程更加稳定准确
- 这是当目标变量满足高斯分布的情况（大部分数据均分布在均值附近，因此大部分误差也产生在均值附近，预测准确的部分很大），如果不是，则需根据实际情况考虑
- 因为平方的存在，误差函数对于局外点、异常点更敏感。如果回归器对某个点的回归值很不合理，那么它的误差则比较大，从而会对MSE/RMSE的值有较大影响，即平均值是非鲁棒的

改进：

- 使用误差的分位数来代替，如中位数来代替均方误差。假设100个数，最大的数再怎么改变，中位数也不会变，因此其对异常点具有鲁棒性。

#### 2. MAE: Mean Absolute Error (平均绝对误差/L1损失)

$$
MAE = \frac{1}{n} \sum_{i=1}^{n}|y_{true}-y_{pred}|
$$

适用：

- 目标变量的分布总体上来讲是高斯的，但也存在一些outliers：一些远离均值的较大或较小的值。

优缺点：

- 对outliers更鲁棒
- 但因导数不连续可能导致寻找最优解的过程中会稍微有一些低效

#### 3. MAPE

$$
MAPE = \frac{100}{n}\sum_{i=1}^{n}|\frac{y_{true}-y_{pred}}{y_{true}}|
$$

适用：

- 数据分布比MAE面临的情况更加不均衡，最大值与最小值差距较大
- 范围[0, inf), 0为完美模型

优点：

- 将不同范围的目标值产生的误差调整到统一范围，用相对误差的百分比来衡量
- 优化过程及优化曲线将更为直观

#### 4. MSLE: Mean Squared Logarithmic Error (均方对数误差)

$$
MSLE =\frac{1}{n} \sum_{i=0}^{n-1}(log_e(1+y_{true})-log_e(1+y_{pred}))^2
$$

适用：

- 当目标实现指数增长时，例如人口数量、商品几年内的平均销量这种样本值较大且差距巨大，或样本波动较大的情况，适合使用此指标

优点：

- 缩小Loss function的值，将较大值的误差映射到一个较小的波动范围，放松对此类误差的惩罚

#### 5. logcosh

$$
L = \sum_{i=1}^n log(cosh(y_{pred}-y_{true})) \\
cosh(x)=\frac{e^x + e^{-x}}{2.0}
$$

解释：

- 预测误差的 双曲余弦 的对数
- 对于小的x, log(cosh(x)) 近似等于 `(x ^2) / 2`
- 对于大的 `x`，近似于 `abs(x) - log(2)`。这表示 'logcosh' 与均方误差大致相同，但是不会受到偶尔疯狂的错误预测的强烈影响

优点

- logcosh损失函数可以在拥有MSE优点的同时也不会受到局外点的太多影响。
- 它拥有Huber的所有优点，并且在每一个点都是二次可导的。



**例子**

```python
# mlp for regression with msle loss function
"""
we will use a standard regression problem generator provided by the 
scikit-learn library. This function will generate examples from a simple
regression problem with a given number of input variables, statistical noise, 
and other properties.
"""
from sklearn.datasets import make_regression
"""
Neural networks generally perform better when the real-valued input and output variables are to be **scaled to a sensible range**.（将属性缩放到一个指定的范围，通常是[0,1] ） 
"""
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot

# generate regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=1)

# standardize dataset
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]

# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]

# define model
model = Sequential()
model.add(Dense(25, input_dim=20, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))
opt = SGD(lr=0.01, momentum=0.9)

# 注意评测标准还是用mse, 为了对比各loss函数之间的训练结果差别
# model.compile(loss='mean_squared_error', optimizer=opt)
model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
# model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mse'])

# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)

# evaluate the model
# The list of metrics has been defined, so the model.evaluate() will return loss and mse
_, train_mse = model.evaluate(trainX, trainy, verbose=0)
_, test_mse = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()

# plot mse during training
pyplot.subplot(212)
pyplot.title('Mean Squared Error')
pyplot.plot(history.history['mean_squared_error'], label='train')
pyplot.plot(history.history['val_mean_squared_error'], label='test')
pyplot.legend()
pyplot.show()
```

<img src=imgs/loss_1.png />

### 1.1.3 分类问题损失函数 

待整理：<https://zhuanlan.zhihu.com/p/39239829

#### 1. hinge (合页损失)

$$
\text{hinge_loss} = \frac{1}{n} \sum_{i=1}^n \text{max}(0, 1-y_{true}*y_{pred})
$$

参考cs231n 笔记修正

SVM中使用的损失函数

由于合页损失函数优化到满足小于一定gap距离就会停止优化，而交叉熵损失会一直优化，后者优化效果一般会好于前者

#### 2. squared_hinge

#### 3. categorical_hinge

#### 4. binary_crossentropy

#### 5. categorical_crossentropy

#### 6. sparse_categorical_crossentropy

#### 7. kullback_leibler_divergence 

![](https://img-blog.csdn.net/20180623230535932?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE0ODQ1MTE5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### poisson

$$
L = \frac{1}{n}\sum_{i=1}^{n}(y_{pred}-y_{true}*log(y_{pred}))
$$



即`(predictions - targets * log(predictions))`的均值



详见：<https://www.cnblogs.com/tbcaaa8/p/4486297.html>



#### cosine proximity

$$
L = -\frac{\sum_{i=1}^n y_{true}*y_{pred}}{\sqrt{\sum_{i=1}^n (y_{true})^2} * \sqrt{\sum_{i=1}^n (y_{true})^2}}
$$

即预测值与真实标签的余弦距离平均值的相反数



数据集：

```python
# scatter plot of the circles dataset with points colored by class
from sklearn.datasets import make_circles
from numpy import where
from matplotlib import pyplot
# generate circles
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# select indices of points with each class label
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()
```

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/11/Scatter-Plot-of-Dataset-for-the-Circles-Binary-Classification-Problem.png width=350 />

#### binary_crossentropy


```python
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
```

[Cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) is the default loss function to use for binary classification problems. It is intended for use with binary classification where the target values are in the set {0, 1}.

- Mathematically, it is the preferred loss function under the inference framework of maximum likelihood. It is the loss function to be evaluated first and only changed if you have a good reason.
- Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for predicting class 1. The score is minimized and a perfect cross-entropy value is 0.

Cross-entropy can be specified as the loss function in Keras by specifying ‘*binary_crossentropy*‘ when compiling the model.

**Example：**

```python
# mlp for the circles problem with cross entropy loss
from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
```







#### hinge

```python
def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)
```

An alternative to cross-entropy for binary classification problems is the [hinge loss function](https://en.wikipedia.org/wiki/Hinge_loss), primarily developed for use with Support Vector Machine (SVM) models.

- It is intended for use with binary classification where the target values are in the set {-1, 1}.

- The hinge loss function encourages examples to have the correct sign, assigning more error when there is a difference in the sign between the actual and predicted class values.

- Reports of performance with the hinge loss are mixed, sometimes resulting in better performance than cross-entropy on binary classification problems.

1. Firstly, the target variable must be modified to have values in the set {-1, 1}.

```python
# change y from {0,1} to {-1,1}
y[where(y == 0)] = -1
```

2. The hinge loss function can then be specified as the ‘*hinge*‘ in the compile function.

   ```python
   model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
   ```

3. Finally, the output layer of the network must be configured to have a single node with a hyperbolic tangent activation function capable of outputting a single value in the range [-1, 1].

   ```python
   model.add(Dense(1, activation='tanh'))
   ```

   



#### squared_hinge

```python
def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)
```

The hinge loss function has many extensions, often the subject of investigation with SVM models.A popular extension is called the squared hinge loss that simply calculates the square of the score hinge loss. 

- It has the effect of smoothing the surface of the error function and making it numerically easier to work with.

- If using a hinge loss does result in better performance on a given binary classification problem, is likely that a squared hinge loss may be appropriate. （如果hinge loss效果差，squared_hinge loss 则会更差）



Multi-Class Classification Loss Functions

Multi-Class classification are those predictive modeling problems where examples are assigned one of more than two classes.

The problem is often framed as predicting an integer value, where each class is assigned a unique integer value from 0 to (*num_classes – 1*). The problem is often implemented as **predicting the probability of the example belonging to each known class.**

In this section, we will investigate loss functions that are appropriate for multi-class classification predictive modeling problems.

We will use the blobs problem as the basis for the investigation. The [make_blobs() function](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html) provided by the scikit-learn provides a way to generate examples given a specified number of classes and input features. We will use this function to generate 1,000 examples for a 3-class classification problem with 2 input variables. The pseudorandom number generator will be seeded consistently so that the same 1,000 examples are generated each time the code is run.

```python
# scatter plot of blobs dataset
from sklearn.datasets.samples_generator import make_blobs
from numpy import where
from matplotlib import pyplot
# generate dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# select indices of points with each class label
for i in range(3):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])
pyplot.show()
```

<img src=https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2018/11/Scatter-Plot-of-Examples-Generated-from-the-Blobs-Multi-Class-Classification-Problem.png width=350 />



#### categorical_crossentropy

```python
def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)
```

[Cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) is the default loss function to use for multi-class classification problems.

- Mathematically, it is the preferred loss function under the inference framework of maximum likelihood. It is the loss function to be evaluated first and only changed if you have a good reason.

- Cross-entropy will calculate a score that summarizes the average difference between the actual and predicted probability distributions for all classes in the problem. The score is minimized and a perfect cross-entropy value is 0.



1. Cross-entropy can be specified as the loss function in Keras by specifying ‘*categorical_crossentropy*‘ when compiling the model.

```python
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

2. The function requires that the output layer is configured with an *n* nodes (one for each class), in this case three nodes, and a ‘*softmax*‘ activation in order to predict the probability for each class

```python
model.add(Dense(3, activation='softmax'))
```

3. In turn, this means that the target variable must be one hot encoded. This is to ensure that each example has an expected probability of 1.0 for the actual class value and an expected probability of 0.0 for all other class values. This can be achieved using the [to_categorical() Keras function](https://keras.io/utils/#to_categorical).

```python
# one hot encode output variable
y = to_categorical(y)
```

当使用"categorical_crossentropy"作为目标函数时,标签应该为多类模式,即one-hot编码的向量,而不是单个数值. 可以使用工具中的`to_categorical`函数完成该转换.示例如下:

```python
from keras.utils.np_utils import to_categorical
 
categorical_labels = to_categorical(int_labels, num_classes=None)
```

#### sparse_categorical_crossentropy

```python
def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)
```

A possible cause of frustration when using cross-entropy with classification problems with **a large number of labels** is the one hot encoding process.

- For example, predicting words in a vocabulary may have tens or hundreds of thousands of categories, one for each label. This can mean that the target element of each training example may require a one hot encoded vector with tens or hundreds of thousands of zero values, requiring significant memory.

- Sparse cross-entropy addresses this by performing the same cross-entropy calculation of error, **without requiring that the target variable be one hot encoded prior to training.**

1. Sparse cross-entropy can be used in keras for multi-class classification by using ‘*sparse_categorical_crossentropy*‘ when calling the *compile()* function.

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

2. The function requires that the output layer is configured with an *n* nodes (one for each class), in this case three nodes, and a ‘*softmax*‘ activation in order to predict the probability for each class.

```python
model.add(Dense(3, activation='softmax'))
```

3. ***No one hot encoding of the target variable is required, a benefit of this loss function.***



**Example:**

```python
# mlp for the blobs multi-class classification problem with sparse cross-entropy loss
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='softmax'))
# compile model
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
```





#### kullback_leibler_divergence


```python
def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
```

[Kullback Leibler Divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence), or KL Divergence for short, **is a measure of how one probability distribution differs from a baseline distribution.**

- A KL divergence loss of 0 suggests the distributions are identical. In practice, the behavior of KL Divergence is very similar to cross-entropy. It calculates how much information is lost (in terms of bits) if the predicted probability distribution is used to approximate the desired target probability distribution.

- As such, the KL divergence loss function is more commonly used **when using models that learn to approximate a more complex function than simply multi-class classification**, such as in the case of an autoencoder used for learning a dense feature representation under a model that must reconstruct the original input. In this case, KL divergence loss would be preferred. Nevertheless, it can be used for multi-class classification, in which case it is functionally equivalent to multi-class cross-entropy.



1. KL divergence loss can be used in Keras by specifying ‘*kullback_leibler_divergence*‘ in the *compile()* function.

   ```python
   model.compile(loss='kullback_leibler_divergence', optimizer=opt, metrics=['accuracy'])
   ```

2. As with cross-entropy, the output layer is configured with an *n* nodes (one for each class), in this case three nodes, and a ‘*softmax*‘ activation in order to predict the probability for each class.

3. Also, as with categorical cross-entropy, we must one hot encode the target variable to have an expected probability of 1.0 for the class value and 0.0 for all other class values.

   ```python
   # one hot encode output variable
   y = to_categorical(y)
   ```



- categorical_hinge

```python
categorical_hinge(y_true, y_pred)
```

- logcosh

```python
logcosh(y_true, y_pred)
```

预测误差的双曲余弦的对数。

对于小的 `x`，`log(cosh(x))` 近似等于 `(x ** 2) / 2`。对于大的 `x`，近似于 `abs(x) - log(2)`。这表示 'logcosh' 与均方误差大致相同，但是不会受到偶尔疯狂的错误预测的强烈影响。

- poisson


```python
poisson(y_true, y_pred)
```

- cosine_proximity


```python
cosine_proximity(y_true, y_pred)
```

----

**注意**: 当使用 `categorical_crossentropy` 损失时，你的目标值应该是分类格式 (即，如果你有 10 个类，每个样本的目标值应该是一个 10 维的向量，这个向量除了表示类别的那个索引为 1，其他均为 0)。 为了将 *整数目标值* 转换为 *分类目标值*，你可以使用 Keras 实用函数 `to_categorical`：

```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```



### 1.1.4 keras.losses.py

keras官方损失函数的原始定义[losses source](https://github.com/keras-team/keras/blob/master/keras/losses.py) 如下： 

```python
"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from . import backend as K
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)


def logcosh(y_true, y_pred):
    """Logarithm of the hyperbolic cosine of the prediction error.
    `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
    to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
    like the mean squared error, but will not be so strongly affected by the
    occasional wildly incorrect prediction.
    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
    # Returns
        Tensor with one scalar loss entry per sample.
    """
    def _logcosh(x):
        return x + K.softplus(-2. * x) - K.log(2.)
    return K.mean(_logcosh(y_pred - y_true), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def serialize(loss):
    return serialize_keras_object(loss)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(name,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='loss function')


def get(identifier):
    """Get the `identifier` loss function.
    # Arguments
        identifier: None or str, name of the function.
    # Returns
        The loss function or None if `identifier` is None.
    # Raises
        ValueError if unknown identifier.
    """
    if identifier is None:
        return None
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'loss function identifier:', identifier)
```



## 1.2 其他常用损失函数

### 1.2.1 回归问题损失函数

#### RMSE : Root Mean Square Error (均方根误差)

$$
MSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_{true}-y_{pred})^2}
$$

适用：

- 与MSE相同

优点：

- 相比MSE在数量级上比较直观，比如RMSE=10，便可以认为预测值与真实值相比平均相差10

**Huber loss:**

![](https://img-blog.csdn.net/20180623232247872?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE0ODQ1MTE5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

解释：

- Huber loss具备了MAE和MSE各自的优点，当δ趋向于0时它就退化成了MAE,而当δ趋向于无穷时则退化为了MSE
- 对于Huber损失来说，δ的选择十分重要，它决定了模型处理局外点的行为。当残差大于δ时使用L1损失，很小时则使用更为合适的L2损失来进行优化。

![img](https://img-blog.csdn.net/20180623232335403?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzE0ODQ1MTE5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



优点：

- Huber损失函数克服了MAE和MSE的缺点，不仅可以保持损失函数具有连续的导数，同时可以利用MSE梯度随误差减小的特性来得到更精确的最小值，也对局外点具有更好的鲁棒性。

缺点：

- 但Huber损失函数的良好表现得益于精心训练的超参数δ。



### 1.2.2 分类问题损失函数













## 1.3 自定义损失函数

我们只要仿照这源码的定义形式，来定义自己的loss就可以了。注意：

- 自己定义的损失函数只计算Li的值，返回各个数据点的误差列表（一维），拟合函数会自动计算所有误差的均值



例如举个最简单的例子，我们定义一个loss为预测值与真实值的差，则可写为：

```python
def my_loss(y_true,y_pred):
    return K.mean((y_pred-y_true),axis = -1) # 拟合值若是多维的，那么对于每个数据点求完所有维度误差后，沿此维度求均值

```

然后，将这段代码放到你的模型中编译，例如

```python
def my_loss(y_true,y_pred):
    return K.mean((y_pred-y_true),axis = -1)

model.compile(loss=my_loss, optimizer='SGD', metrics=['accuracy'])
```

- 有一点需要注意，Keras作为一个高级封装库，它的底层可以支持theano或者tensorflow，在使用上边代码时，首先要导入这一句

  ```python
  from keras import backend as K
  ```
  

这样你自定义的loss函数就可以起作用了



Remark:

- K.mean 使用的是 tensorflow.reduce_mean, 详见<https://blog.csdn.net/dcrmg/article/details/79797826>

# 2. Model Metrics

评价函数用于评估当前训练模型的性能。

## 2.1 内置评价函数

**用法：**

- 当模型编译后（compile），评价函数应该作为 `metrics` 的参数来输入(列表形式传入)。评价函数和 [损失函数](/losses) 相似，只不过**评价函数的结果不会用于训练过程中**。
  - 我们可以传递**已有的评价函数名称**，
  - 或者传递一个自定义的 Theano/TensorFlow 函数来使用

- 每当训练数据集中有一个epoch训练完成后，此时的性能参数会被记录下来。如果提供了验证数据集，验证数据集中的性能评估参数也会一并计算出来。

- 性能评估指标可以通过输出查看，也可以通过调用模型类的`fit()`方法获得。这两种方式里，性能评估函数都被当做关键字使用。如果要查看验证数据集的指标，只要在关键字前加上`val_`前缀即可。

- 损失函数和Keras明确定义的性能评估指标都可以当做训练中的性能指标使用。

***Remark:***

- 注意将其与评价指标 损失函数 `loss` 区分，训练模型仅仅需要损失函数和优化器参数就够了，优化算法根据损失函数值来反向传播更新参数；
- 但损失函数值有时并不能直观的反应网络模型的训练效果，因此我们需要使用恰当的评价指标（如准确率等）来直观显示训练过程的效果变化，或用来控制训练何时停止。

方式1：传入字符串名称

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

方式2：传入函数

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```



**keras 模型构建后获取评价参数名称：**

```
scores = model.evaluate(X[test], Y[test], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
```



### 2.1.1 回归问题性能评估指标

- **均方误差**：mean_squared_error，MSE或mse
- **平均绝对误差**：mean_absolute_error，MAE，mae
- **平均绝对误差百分比**：mean_absolute_percentage_error，MAPE，mape
- **Cosine距离**：cosine_proximity，cosine

**实例：**

```js
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# prepare data, X.shape = (10,)
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


# create model
model = Sequential()
model.add(Dense(30, input_dim=1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', 
              metrics=['mse', 'mae', 'mape', 'cosine', 'msle'])

# train model
hist = model.fit(X, Y, epochs=100, batch_size=5, verbose=2)

# plot metrics
plt.plot(hist.history['mean_squared_error'], label="mse")
plt.plot(hist.history['mean_absolute_error'], label='mae')
plt.plot(hist.history['mean_absolute_percentage_error'], label='mape')
plt.plot(hist.history['cosine_proximity'], label='cosine')
plt.plot(hist.history['mean_squared_logarithmic_error'], label='msle')
plt.legend()
plt.show()
```

在上面的例子中，性能评估指标是通过别名'mse', 'mae', 'mape', 'cosine'指定的，通过别名对应的函数全名来作为模型对象下的键值调用对应的性能评估函数。

我们自然也可以使用函数全名来指定性能评估指标，如下所示：

```js
model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
```

如果我们已经从Keras的包中import了metrics类，那么就可以直接指定其下的函数。

```js
from keras import metrics
model.compile(loss='mse', optimizer='adam', metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity])
```

也可以使用损失函数作为度量标准，如使用均方对数误差（*mean_squared_logarithmic_error*，*MSLE*或*msle*）损失函数作为度量标准：

```js
model.compile(loss='mse', optimizer='adam', metrics=['msle'])
```

运行实例在每个epoch结束时打印性能评估指标。

```js
Epoch 96/100
 - 0s - loss: 11.4683 - mean_squared_error: 11.4683 - mean_absolute_error: 2.7736 - mean_absolute_percentage_error: 49.3052 - cosine_proximity: -1.0000e+00 - mean_squared_logarithmic_error: 0.2852
Epoch 97/100
 - 0s - loss: 11.1887 - mean_squared_error: 11.1887 - mean_absolute_error: 2.7386 - mean_absolute_percentage_error: 49.0319 - cosine_proximity: -1.0000e+00 - mean_squared_logarithmic_error: 0.2759
Epoch 98/100
 - 0s - loss: 10.9365 - mean_squared_error: 10.9365 - mean_absolute_error: 2.7083 - mean_absolute_percentage_error: 48.8079 - cosine_proximity: -1.0000e+00 - mean_squared_logarithmic_error: 0.2676
Epoch 99/100
 - 0s - loss: 10.6957 - mean_squared_error: 10.6957 - mean_absolute_error: 2.6780 - mean_absolute_percentage_error: 48.6424 - cosine_proximity: -1.0000e+00 - mean_squared_logarithmic_error: 0.2600
Epoch 100/100
 - 0s - loss: 10.4564 - mean_squared_error: 10.4564 - mean_absolute_error: 2.6476 - mean_absolute_percentage_error: 48.4749 - cosine_proximity: -1.0000e+00 - mean_squared_logarithmic_error: 0.2528
```

Keras为回归问题提供的四个内置性能评估指标及一个Loss function 指标随epoch完成个数变化的折线图：

![img](/home/ubuntu16/Deep-learning-tutorial/keras/imgs/metrics_reg.png)



### 2.1.2 分类问题性能评估指标

以下是Keras为分类问题提供的性能评估指标。

- **对二分类问题,计算在所有预测值上的平均正确率**：binary_accuracy，acc
- **对多分类问题,计算在所有预测值上的平均正确率**：categorical_accuracy，acc
- **在稀疏情况下，多分类问题预测值的平均正确率**：sparse_categorical_accuracy
- **计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确**：top_k_categorical_accuracy（需要手动指定k值）
- **在稀疏情况下的top-k正确率**：sparse_top_k_categorical_accuracy（需要手动指定k值）

**Remark：**

- 准确性是一个特别的性能指标，无论您的问题是二元还是多分类问题，都可以指定“ *acc* ”指标来评估准确性。

**实例：**

```js
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# prepare sequence
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# create model
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['acc'])
# train model
hist = model.fit(X, Y, epochs=300, batch_size=5, verbose=2)

# plot metrics
plt.plot(hist.history['acc'], label='acc')
plt.legend()
plt.show()
```

训练过程：

<img src=imgs/metrics_2.png />



也可以使用多分类：

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.utils import to_categorical

# prepare sequence
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# one hot标签
Y = to_categorical(y, num_classes=2)
# create model
model = Sequential()
model.add(Dense(10, input_dim=1))
# 最后一层输出与类别相同
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['acc'])
# train model
hist = model.fit(X, Y, epochs=100, batch_size=5, verbose=2)

# plot metrics
plt.plot(hist.history['acc'], label='acc')
plt.plot(hist.history['loss'], label='loss')
plt.legend()
plt.show()
```

 训练过程：

<img src=imgs/metrics_3.png />

### 2.1.3 keras.metrics.py

```python
"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
# 所有Loss function 亦可作为评价指标
from . import backend as K
from .losses import mean_squared_error
from .losses import mean_absolute_error
from .losses import mean_absolute_percentage_error
from .losses import mean_squared_logarithmic_error
from .losses import hinge
from .losses import logcosh
from .losses import squared_hinge
from .losses import categorical_crossentropy
from .losses import sparse_categorical_crossentropy
from .losses import binary_crossentropy
from .losses import kullback_leibler_divergence
from .losses import poisson
from .losses import cosine_proximity
from .utils.generic_utils import deserialize_keras_object
from .utils.generic_utils import serialize_keras_object


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def sparse_categorical_accuracy(y_true, y_pred):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = K.argmax(y_pred, axis=-1)
    y_pred_labels = K.cast(y_pred_labels, K.floatx())
    return K.cast(K.equal(y_true, y_pred_labels), K.floatx())


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.cast(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), K.floatx())


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return K.cast(K.in_top_k(y_pred, K.cast(K.flatten(y_true), 'int32'), k),
                  K.floatx())


# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def serialize(metric):
    return serialize_keras_object(metric)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='metric function')


def get(identifier):
    if isinstance(identifier, dict):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif isinstance(identifier, six.string_types):
        return deserialize(str(identifier))
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret '
                         'metric function identifier:', identifier)
```

## 2.2 其他常用评价指标

### 2.2.1 回归模型评价指标

**Remark:**

- 之前定义的误差参数都是预测值与真值之间的误差，即点与点之间的误差，接下来的参数大多相对于点预测值/预测误差 相对原始数据平均值定义的，即点对全。

#### 1. MedAE: Media Absolute Error (中位数绝对误差)

$$
MedAE = media(|y_{1true}-y_{1pred}|,\dots, |y_{ntrue}-y_{npred}|)
$$

优点：用于包含异常值的数据的衡量

#### 2. R-square: Coefficient of determination (决定系数)

$$
R^2 = 1-\frac{\sum (y_{true}-y_{pred})^2 }{\sum (y_{true}-y_{mean})^2}
$$

**名词解释：**

残差： $e=y_{true}-y_{pred}$

残差平方和SSE: The Sum of Squares due to Error 
$$
SSE = \sum(y_{true}-y_{pred})^2
$$
总平方和：Total sum of squares
$$
SST = \sum(y_{true}-y_{mean})^2
$$
其中 $y_{mean}=\sum_{i=1}^n y_i$

回归平方和：Sum of Squares of the Regression
$$
SSR = \sum(y_{pred}-y_{mean})^2
$$
解释：

- 从原始定义角度：$R^2$ 原始定义为回归平方和与总平方和比值，分子为预测值相对与均值的偏离程度，分母为真值相对于均值的偏离程度（作为基准），最佳情况为预测值等于真值，二者比值为1，最差情况是预测值等于均值，比值为0

$$
R^2 = \frac{SSR}{SST} = \frac{SST-SSE}{SST} = 1-\frac{SSE}{SST}
$$

- 从残差的角度：第二项的分子为使用模型预测值产生的误差，分母为采用均值作为预测值产生的误差（基准值），当模型预测值为真值是模型最优，第二项值为0，最差是模型预测值为均值，第二项值为1
- 从方差的角度：若第二项分子分母同时除以样本数n，则分子为MSE，分母为样本方差，表征原始数据的离散程度。MSE除以样本方差用来消除原始数据离散程度对结果造成的影响，这样当原始数据集离散程度过大或过小时，对应的缩小或夸大误差，使得最终都能得到类似的$R^2$值。

标准：

- 越接近1越好 （经验值>0.4拟合效果好）

缺点：

- 数据集样本数量越大，$R^2$越大，所以不能用于不同数据集的比较，无法定量说明准确程度。（可以用于交叉验证？样本数量相同但分布可能不同，防止由于数据分布不同产生不同的预测误差）

#### 3. Adjusted R-square: Degree-of-freedom adjusted coefficient of determination (校正决定系数)

$$
R^2_{adjust} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
$$

解释：n为样本数量，p为样本特征数量；消除样本数据和变量数量的影响 （样本数较少时，对（1-R^2）放大）

优点：

- 抵消样本数量对$R^2$的影响，真正做到[0,1]区间，值越大模型越好
- 当$R^2$相同，样本数量也相同时，使用样本特征少的模型更优

#### 4. EVS: Explained Variance Score (解释方差分数)

$$
EVS = 1- \frac{Var(y_{true}-y_{pred})}{Var(y_{true})}
$$

解释：衡量模型对数据集预测结果的波动程度（用预测误差的方差，MSE,MAE差不多时，即误差的均值差不多时，看谁错误的波动大， [1,1,1,1] [0,0,0,4] ,但没准最后一个是坏数据呢？）

$R^2$ 指标也包含此功能。

标准：越接近1，效果越好

**EVA: Explained Variance Ratio**
$$
EVA = \frac{Var[y_{true}-y_{pred}]}{Var[y_{true}]}
$$


标准：同一数据集下，SSE值越小，误差越小，代表拟合效果越好 

缺点：

- 不同数据集之间，SSE的值没有意义，因为其值不仅取决于预测值，还取决于样本数量



#### QQ图

<https://www.cnblogs.com/qwj-sysu/p/8484924.html>

### 2.2.2 分类模型评测指标

准确率：预测正确的样本占总样本的比例

问题：**假如有100个样本，其中1个正样本，99个负样本，如果模型的预测只输出0，那么正确率是99%，这时候用正确率来衡量模型的好坏显然是不对的。**

查准率： 预测值为1且真实值为1的样本在**所有预测值为1的样本**中所占的比例， 即挑出来的好瓜中有多大的比例是好瓜；

召回率：预测值为1且真实值为1的样本在所有真实值为1的样本中所占的比例，即真正的好瓜有多大的比例被挑出来了。

F1分数（F1-Score），又称为平衡F分数（BalancedScore），它被定义为精确率和召回率的调和平均数。

<https://blog.csdn.net/zjn295771349/article/details/84961596>

<https://blog.csdn.net/a819825294/article/details/51699211>

## 2.3 自定义评价函数

自定义评价函数应该在编译的时候（compile）传递进去。该函数需要以 `(y_true, y_pred)` 作为输入参数(顺序不可变)，并返回一个张量作为输出结果。

- __参数__：
  - __y_true__: 真实标签，Theano/Tensorflow 张量。
  - __y_pred__: 预测值。和 y_true 相同尺寸的 Theano/TensorFlow 张量。
- __返回值__：  返回一个表示**全部数据点** **平均值的一维张量**。

```python
import keras.backend as K

# 即使计算中不需要也要传入 y_true, y_pred两个参数
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

# history 字典会将函数名作为关键字存储训练历史数据
plt.plot(hist.history['mean_pred'], label='mean_pred')
```

你的自定义性能评估函数必须在Keras的内部数据结构上进行操作而不能直接在原始的数据进行操作，具体的操作方法取决于你使用的后端（如果使用TensorFlow，那么对应的就是`tensorflow.python.framework.ops.Tensor`）。

由于这个原因，我建议最好使用后端提供的数学函数来进行计算，这样可以保证一致性和运行速度。

#### 均方根误差（RMSE）。

你可以通过观察官方提供的性能评估指标函数来学习如何编写自定义指标。

下面展示的是[Keras中mean_squared_error损失函数（即均方差性能评估指标）](https://github.com/fchollet/keras/blob/master/keras/losses.py)的代码。

```js
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
```

K是Keras使用的后端(例如TensorFlow)。从这个例子以及其他损失函数和性能评估指标可以看出：需要使用后端提供的标准数学函数来计算我们感兴趣的性能评估指标。

现在我们可以尝试编写一个自定义性能评估函数来计算RMSE，如下所示：

```python
from keras import backend as K
 
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
 
```

你可以看到除了用`sqrt()`函数封装结果之外，这个函数的代码和MSE是一样的。

#### R-square

```python
def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    return SSR/SST
```

#### Adjusted R-square

#### 

$$
R^2_{adjust} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
$$



```python
def ad_r_square(y_true, y_pred, p):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    R2 = SSR/SST
    n  = len(y_true)
    return 1.0 - ((1.0-R2)*(n-1))/(n-p-1)
```





# 3. 模型验证方法

因单次模型训练效果具有偶然性，为了评测某一模型实际性能，得到准确可信的模型性能指标值就需要选择合适的验证方法。（至少也得多次训练取平均，单次训练结果不能说明任何问题）

问题：

评测指标可以多次训练，多个模型的指标值取平均，那么到底最后选择哪一个模型使用呢？

- 单次最好？也只能代表在这个验证集上效果好，其他验证集也不一定

## 3.1 数据集拆分验证

### 3.1.1 The Validation Set Approach

方案：

- 把整个数据集分成两部分，一部分用于训练，一部分用于验证，即训练集和验证集
- 最后在测试集上观察不同模型对应的MSE的大小，据此选择合适模型和参数

缺点：

- 最终模型与参数的选取依赖于对训练集和验证集的划分方法，不同的划分方法产生的MSE不同，如果划分的不够合理，无法得到最好的模型
- 由于分出一部分当做验证集，所以并没有使用所有数据进行训练



### 3.1.2 Cross-Validation

#### LOOCV (Leave-one-out cross-validation)

方案：

- 也是分为训练集与验证集，但不同的是验证集的大小只有一个数据，剩余n-1个数据全部当做训练集
- 将此步骤重复n次，直至所有数据都当过一次验证集
- 最后将n次结果求和取平均

优点：

- 不受验证集划分方法的影响，因为每一个数据都单独做过数据集
- 同时训练集大小为n-1，几乎等同于使用全部数据进行训练

缺点：

- 当数据集较大时，计算成本过大

改进：

![img](https://pic2.zhimg.com/80/v2-ec72b82d605902ddfa060c2fb5777a05_hd.png)

其中![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By_i%7D)表示第i个拟合值，而![[公式]](https://www.zhihu.com/equation?tex=h_i)则表示leverage。关于![[公式]](https://www.zhihu.com/equation?tex=h_i)的计算方法详见线性回归的部分（以后会涉及）。



#### K-fold-Cross Validation

方案：

- 将所有数据集平均分成k份
- 不重复的每次取其中一份做测试集，其他四份做训练集蓄念模型，计算此测试集上的评价指标值
- 将k此得到的结果取平均作为最后的评测结果

优点：

- 能得到和LOOCV类似的结果但计算量要远远小

**Bias-Variance Trade-Off for k-Fold Cross-Validation**: 

- K越大，每次投入的训练集的数据越多，模型的Bias越小。
- 但是K越大，又意味着每一次选取的训练集之前的相关性越大（考虑最极端的例子，当k=N，也就是在LOOCV里，每次都训练数据几乎是一样的）。而这种大相关性会导致最终的test error具有更大的Variance。
- 一般来说，根据经验我们一般选择k=5或10



## 3.2 sklearn 支持的数据集划分方法

**注意：** 

- Keras支持自动切分数据与手动切分数据，自动切分数据只需要将整个数据集指定给fit函数即可，手动切分用到本节的划分方法，划分后训练时将训练集与测试集分别指定。（详见fit()函数使用）
- 自动切分时，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,                                                                   test_size=None,
                                                    train_size=None,
                                                    random_state=None,
                                                    shuffle=True,
                                                    stratify=None)
```

参数：

- X, Y : 数据集样本与数据集对应标签；也可仅给定一个X，比如由自定义的数据格式组成的数据集（样本/样本存储位置与对应标签存在一个自定义的数据格式内），这样返回值就是 (training_set, validation_set), 然后在数据生成器内把样本与标签分别取出即可

- test_size, train_size: [0.0, 1.0]之间的浮点数，如果二者都未指定，默认test_size=0.25

- random_state: 整数，随机数种子，用于复现划分结果

- stratify: 对应的分层抽样的标签。若指定，将保证验证集与测试集内不同类样本标签的比例与原始数据集一致。

  - 注意：如果使用分层划分，标签内的样本的类别必须不少于两类，否则无法划分；并且，类别数必须小于测试集数量，否则也无法划分；

  - 举例：stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(…test_size=0.25, stratify = y_all), 那么split之后数据如下：

    training:75个数据，其中60个属于A类，15个属于B类。 
    testing: 25个数据，其中20个属于A类，5个属于B类。

    用了stratify参数，training集和testing集的类的比例是 A：B=4：1，等同于split前的比例（80：20）。通常在这种正负样本分布不平衡的情况下会用到stratify。

例子：

```python
import numpy as np
from sklearn.model_selection import train_test_split

X = np.arange(30).reshape((10,3))
Y = [0,0,0,0,0,1,1,1,1,1]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,
                                                   random_state=3,
                                                   shuffle=True,
                                                   stratify=Y)
##
[[27 28 29]
 [ 9 10 11]
 [21 22 23]
 [12 13 14]
 [24 25 26]
 [ 3  4  5]
 [ 0  1  2]
 [18 19 20]]
[[ 6  7  8]
 [15 16 17]]
[1, 0, 1, 0, 1, 0, 0, 1]
[0, 1]
```



### 3.2.1 随机划分

原理：

- 首先对样本全体进行随机打乱
- 然后划分出 train/test 对，由迭代器产生指定数量的独立的数据集划分



**ShuffleSplit:**



**GroupShuffleSplit:**



**StratifiedShuffleSplit:**

返回分层随机划分，即保证划分后的各子数据集内不同类样本比例与原始数据集一致

### 3.2.2 k折交叉验证

**KFold:**

```python
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=False, random_state=None)
```

参数：

- n_splits: 划分子数据集个数，最少为2

方法：

- get_n_splits(): 返回交叉验证过程中要拆分迭代的次数
- split(X): 返回(train_index, test_index)，即单次划分后的训练集与测试集索引列表

Remark:

- 划分后各个子数据集的数量为：有 `n_samples % n_splits` 个子数据集内样本数量为`n_samples//n_splits +1` ，其他子数据集内样本数量为`n_samples//n_splits`

例子：

```python
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
print kf.get_n_splits()
print kf  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
## ===
2
KFold(n_splits=2, random_state=None, shuffle=False)
('TRAIN:', array([2, 3]), 'TEST:', array([0, 1]))
('TRAIN:', array([0, 1]), 'TEST:', array([2, 3]))
```



**GroupKFold:**

```python
GroupKFold(n_splits=5)
```

通过对每个数据集样本指定其隶属于某个组，在划分时，同一组的数据不会划分到不同的子集内（与分层划分大致相反），同时每个子集内不同组数据的个数也会尽量保证均衡。（前提是组别数量要大于等于自数据集数量）

方法：

- get_n_splits(X, y, groups): 返回划分数据集的迭代次数
- split(X, y, groups): 返回单次划分后的训练集、验证集索引列表

例子：

```python
import numpy as np
from sklearn.model_selection import GroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
groups = np.array([0, 1, 0, 1]) # 结果必然是0,2一组

group_kfold = GroupKFold(n_splits=2)
print group_kfold.get_n_splits(X, y, groups)

for train_index, test_index in group_kfold.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, X_test, y_train, y_test)
    
## ==
2
('TRAIN:', array([0, 2]), 'TEST:', array([1, 3]))
(array([[1, 2],
       [5, 6]]), array([[3, 4],
       [7, 8]]), array([1, 3]), array([2, 4]))
('TRAIN:', array([1, 3]), 'TEST:', array([0, 2]))
(array([[3, 4],
       [7, 8]]), array([[1, 2],
       [5, 6]]), array([2, 4]), array([1, 3]))
```



**StratifiedKFold:**

```python
StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
```

根据标签数据集内不同样本数量所占的比例进行分类，保证划分后的各个自数据集内均包含所有类别数据且不同类别数据之间的比例维持与原数据集一致。（前提是保证标签样本内类别要大于等于划分子数据集的个数，否则报错）

参数：

方法：

- get_n_splits()
- split(y): 根据y内数据类别进行分层划分

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1]) # 必然是0 ，1 一组，2,3一组 (不混洗数据的话)
# 混洗后也可能 1,2一组

skf = StratifiedKFold(n_splits=2)
print skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```



**RepeatedKFold:**

```python
RepeadedKFold(n_splits=5, n_repeats=10, random_state=None)
```

重复交叉交叉验证n_repeats 次。（每次重复时使用不同的随机方式）

参数：

- n_repeats: 交叉验证的重复次数

方法：

- get_n_splits()
- split(X)

```python
import numpy as np
from sklearn.model_selection import RepeatedKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=2652124)

for train_index, test_index in rkf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
# ===
('TRAIN:', array([0, 1]), 'TEST:', array([2, 3]))
('TRAIN:', array([2, 3]), 'TEST:', array([0, 1]))
('TRAIN:', array([1, 2]), 'TEST:', array([0, 3]))
('TRAIN:', array([0, 3]), 'TEST:', array([1, 2]))
```

**RepeatedStratifiedKFold**

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html#sklearn.model_selection.RepeatedStratifiedKFold

Repeats Stratified K-Fold n times.

### 3.2.3 留一法

**LeaveOneOut**

**LeavePOut**

原理：

- 对于n个样本，每次保留p个作为测试集，其他n-p个样本作为训练集
- 这样总共会有$C_n^p$ 个训练-测试对，且样本之间会发生少量重叠
- 当p=1时，便是留1法

**LeaveOneGroupOut**

<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut>

**LeavePGroupOut**



# 附录

## 协方差与相关系数

### 维基定义

协方差是概率论与统计学中用于衡量两个变量的总体误差。方差是协方差的一种特殊情况，及两个变量是相同的情况。

期望值分别为$E(X)=u$ 与$E(Y)=v$ 的两个具有有限二阶矩的实数随机变量 $X$ 与$Y$ 之间的协方差定义为：
$$
Cov(X,Y) = E((X-u)(Y-v)) = E(XY)-uv
$$
协方差表示的是两个变量的总体误差（两个变量偏离各自期望的程度，方差是一个变量偏离自身期望的程度）。如果两个变量的变化趋势一致，即当一个变量大于其自身期望值时，对应的另一个变量也大于自身期望值，则协方差值为正，反之，两个变量朝相反的方向变化，一个大于自身期望时另一个小于自身期望，则协方差值为负。

如果两个变量是统计独立的，那么二者之间的协方差就是0，因为
$$
E(XY)= E(X)E(Y)=uv
$$
但反之并不成立，即协方差为0，并不代表二者独立。

协方差的相关性：
$$
\eta = \frac{cov(X,Y)}{\sqrt{Var(X)*Var(y)}}
$$
更准确地说是[线性相关性](https://zh.wikipedia.org/wiki/線性相關性)，是一个衡量线性独立的[无量纲](https://zh.wikipedia.org/wiki/无量纲)数，其取值在[－1, 1]之间。相关性η = 1时称为“完全线性相关”（相关性η = －1时称为“完全线性负相关”），此时将Yi对Xi作Y-X [散点图](https://zh.wikipedia.org/w/index.php?title=散点图&action=edit&redlink=1)，将得到一组精确排列在直线上的点；相关性数值介于－1到1之间时，其绝对值越接近1表明线性相关性越好，作散点图得到的点的排布越接近一条直线。

相关性为0（因而协方差也为0）的两个随机变量又被称为是[不相关](https://zh.wikipedia.org/wiki/相关)的，或者更准确地说叫作“线性无关”、“线性不相关”，这仅仅表明*X* 与*Y* 两随机变量之间没有线性相关性，并非表示它们之间一定没有任何内在的（非线性）函数关系，和前面所说的“X、Y二者并不一定是统计独立的”说法一致。

### 通俗解释

首先从公式来看：
$$
Cov(X,Y) = E((X-u)(Y-v))
$$
代表每一个时刻X值与均值之差乘以Y值与均值之差，将所有时刻的乘积求和再算均值(求期望)，那么这个均值代表什么呢？

假设由两个随机变量，观察他们t1~t7七个时刻的值的变化情况：

<img src=imgs/a1_1.jpg width=200 /><img src=imgs/a1_2.jpg width=200 /><img src=imgs/a1_3.jpg width=180 />

- 对于图1，他们同一时刻X-u 与Y-u的符号一定相同（偏离自身均值的方向一直），那么最后累加后的均值也必为正，代表二者变化趋势相同
- 对于图2，他们每一时刻X-u 与Y-u的符号相反，最后累加后均值也必为负值，故代表二者变化趋势相反；
- 对于图3，表示的是一种一般的情况，有时二者“变化”相同，有时“变化”相反，故用总的误差的均值来表示总体变化趋势，若为负，代表大部分时间二者偏离程度相反，即二者变化趋势总的来说是相反的。

问题：我们知道了协方差的正负可以用来衡量两个变量变化的趋势是否一致，那么比较两个协方差的值的大小是否可以判定某两个变量比其他两个更加相关呢？答案是否定的，因为协方差值受各自偏离程度的影响，偏离程度大的协方差值就大，但两个变量的相关程度（变化趋势是否一致）并不受其值偏离程度的影响。由此引出相关系数的概念：

相关系数：
$$
\rho = \frac{cov(X,Y)}{\sigma_X \sigma_Y}
$$
标准差：$\sigma_X = \sqrt{E((X-\mu_x)^2)}$

即相关系数是一种剔除了两个变量量纲的影响，标准化后的特殊的协方差：它消除了两个变量变化幅度的影响（除以标准差），只反映两个变量每单位变化时的相似程度。



那么为何除以标准差就可以提出变化幅度的影响呢？从公式的角度来看：

标准差是每一时刻变量的值与均值的偏差，再平方，然后求和取平均，最后开方：

- 首先做差是为了计算偏离幅度
- 平方是为了消除反向偏离与正向偏离造成的正负值相互抵消，因为要计算总的累积偏差
- 求和取平均就是看平均偏差了多少
- 因为刚才有平方操作，为了回到真正的偏差量级，所以最后进行开方操作

综上，标准差描述了变量在整体变化过程中偏离均值的平均幅度，那么协方差除以两个变量各自的标准差，便消除了分子中偏离幅度的影响，只包含二者相对偏离趋势。相关系数值越大，代表二者变化越正相关。（任何不一致的行为反映到二者乘积上最后求和时减少其值）

最后，相关系数的取值范围在＋1到－1之间变化可以通过施瓦茨不等式来证明。

总结：

- 当他们的相关系数为1时，说明两个变量变化时的正向相似度最大，即，你变大一倍，我也变大一倍；你变小一倍，我也变小一倍。也即是完全正相关（以X、Y为横纵坐标轴，可以画出一条斜率为正数的直线，所以X、Y是线性关系的）。

- 随着他们相关系数减小，两个变量变化时的相似度也变小，当相关系数为0时，两个变量的变化过程没有任何相似度，也即两个变量无关。

- 当相关系数继续变小，小于0时，两个变量开始出现反向的相似度，随着相关系数继续变小，反向相似度会逐渐变大。

- 当相关系数为－1时，说明两个变量变化的反向相似度最大，即，你变大一倍，我变小一倍；你变小一倍，我变大一倍。也即是完全负相关（以X、Y为横纵坐标轴，可以画出一条斜率为负数的直线，所以X、Y也是线性关系的）。



举例：

<img src=imgs/a1_4.jpg />

情况1和情况2从图上看二者应该都是正相关的，都具有 "类似的"变化趋势，即相关度。只是由于图2中X量级比较小，但变化程度大家都是一样的。

那么首先计算方差：

情况1：
$$
[(100-0)*(70-0)+(-100-0)*(-70-0)+(-200-0)*(-200-0) \\
+(-100-0)*(-70-0)+(100-0)*(70-0)+(200-0)*(200-0) \\
+(0-0)*(0-0)]/7 =15428.5713
$$
情况2：
$$
[(0.01-0)*(70-0)+(-.01-0)*(-70-0)+(-0.02)*(-200-0)+(-0.01)*(-70-0)+(0.01)*(70-0)+(0.02)*(200-0)+(0.0)*(0-0)]/7= 1.54285713
$$
二者方差都为正，说明都是正相关，但值相差了10000倍，丝毫看不出两种情况谁的相关度更大。再计算相关系数：

情况1：

$\sigma_X = 130.9307, \sigma_Y=119.2836, \rho=0.9879$

情况2：

$\sigma_X = 0.01309307, \sigma_Y=119.2836, \rho=0.9879$

所以两种情况，两个变量的相关度相同