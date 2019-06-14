## 1. 概述

### 1.1 使用方法

损失函数（或称目标函数、优化评分函数）**是编译模型时所需的两个参数之一**：

***Remark:***

- 注意将其与评价指标 `metrics` 区分，训练模型仅仅需要损失函数和优化器参数就够了，优化算法根据损失函数值来反向传播更新参数；
- 但损失函数值有时并不能直观的反应网络模型的训练效果，因此我们需要使用恰当的评价指标（如准确率等）来直观显示训练过程的效果变化，或用来控制训练何时停止。

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

- loss:
  - 你可以传递一个现有的损失函数名
  - 或者一个 TensorFlow/Theano 符号函数。该符号函数为每个数据点返回一个标量，有以下两个参数:
    - __y_true__: 真实标签。TensorFlow/Theano 张量。
    - __y_pred__: 预测值。TensorFlow/Theano 张量，其 shape 与 y_true 相同。
    - 实际的优化目标是所有数据点的输出数组的平均值。

### 1.2 Regression Loss Functions

#### mean_squared_error

The mean squared error loss function can be used in Keras by specifying ‘*mse*‘ or ‘*mean_squared_error*‘ as the loss function when compiling the model.

```python
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
```

The Mean Squared Error, or MSE, loss is the default loss to use for regression problems.

- Mathematically, it is the preferred loss function under the inference framework of maximum likelihood **if the distribution of the target variable is Gaussian.** 
- It is the loss function to be evaluated first and only changed if you have a good reason.
- Mean squared error is calculated as **the average of the squared differences** between the predicted and actual values. The result is always positive regardless of the sign of the predicted and actual values and a perfect value is 0.0. 
- **The squaring means that larger mistakes result in more error than smaller mistakes**, meaning that the model is punished for making larger mistakes.
  - 这是当目标变量满足高斯分布的情况（大部分数据均分布在均值附近，因此大部分误差也产生在均值附近，预测准确的部分很大）
  - 如果目标值有的大有的小，5（目标）和4（预测）之间误差为1, 0.1 和0.3误差0.04，采用MSE，网络会认为前者误差较大，会尽量预测正确大目标值（大部分误差均产生在较大目标值附近），而忽略小的目标值。 但前者误差仅20%，后者误差200%！！ -- 会放大大误差，缩小小误差


#### mean_absolute_error


```python
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)
```

On some regression problems, the distribution of the target variable may be mostly Gaussian, but may have outliers, e.g. **large or small values far from the mean value**.

- The Mean Absolute Error, or MAE, loss is an appropriate loss function in this case as **it is more robust to outliers.** It is calculated as the average of the absolute difference between the actual and predicted values.
- 适用于处理数据分布总体上还是高斯的，但含有一些例外较大或较小的值，比如大部分数据都集中在均值为0的区域，误差为1，但突然来了一个5，这样总的误差增加了25，但适用MAE，总误差仅增加5

The model can be updated to use the ‘_mean_absolute_error_‘ loss function and keep the same configuration for the output layer.





#### mean_absolute_percentage_error


```python
def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)
```



#### mean_squared_logarithmic_error

```python
def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)
```

There may be regression problems in which the **target value has a spread of values and when predicting a large value**, you may not want to punish a model as heavily as mean squared error.

Instead, you can first calculate **the natural logarithm of each of the predicted values**, then calculate the mean squared error. This is called the Mean Squared Logarithmic Error loss, or MSLE for short.

- It has the effect of **relaxing the punishing effect of large differences in large predicted values.**

- As a loss measure, it may be more appropriate **when the model is predicting unscaled quantities directly**. Nevertheless, we can demonstrate this loss function using our simple regression problem.
- 误差比较小时，msle与mse二者区别不大；误差越大时，msle对大误差的惩罚相比与mse二者差距越大。

The model can be updated to use the ‘*mean_squared_logarithmic_error*‘ loss function and keep the same configuration for the output layer. We will also track the mean squared error as a metric when fitting the model so that we can use it as a measure of performance and plot the learning curve.

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



### 1.3 Binary Classification Loss Functions

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





### Multi-Class Classification Loss Functions

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



## 自定义损失函数

参考keras官方损失函数的原始定义[losses source](https://github.com/keras-team/keras/blob/master/keras/losses.py)， 我们亦可以定义自己的损失函数。

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

我们只要仿照这源码的定义形式，来定义自己的loss就可以了。例如举个最简单的例子，我们定义一个loss为预测值与真实值的差，则可写为：

```python
def my_loss(y_true,y_pred):
    return K.mean((y_pred-y_true),axis = -1)

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