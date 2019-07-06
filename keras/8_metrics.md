## 评价函数的用法

评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 `metrics` 的参数来输入(列表形式传入)。

***Remark:***

- 注意将其与评价指标 损失函数 `loss` 区分，训练模型仅仅需要损失函数和优化器参数就够了，优化算法根据损失函数值来反向传播更新参数；
- 但损失函数值有时并不能直观的反应网络模型的训练效果，因此我们需要使用恰当的评价指标（如准确率等）来直观显示训练过程的效果变化，或用来控制训练何时停止。

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

评价函数和 [损失函数](/losses) 相似，只不过**评价函数的结果不会用于训练过程中**。

- 我们可以传递已有的评价函数名称，
- 或者传递一个自定义的 Theano/TensorFlow 函数来使用（查阅[自定义评价函数](#custom-metrics)）。
  - __参数__：
    - __y_true__: 真实标签，Theano/Tensorflow 张量。
    - __y_pred__: 预测值。和 y_true 相同尺寸的 Theano/TensorFlow 张量。
  - __返回值__：  返回一个表示全部数据点平均值的张量。

----



## 可使用的评价函数


### binary_accuracy


```python
binary_accuracy(y_true, y_pred)
```

----

### categorical_accuracy


```python
categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

## 自定义评价函数

自定义评价函数应该在编译的时候（compile）传递进去。该函数需要以 `(y_true, y_pred)` 作为输入参数，并返回一个张量作为输出结果。

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```





# 使用Keras在训练深度学习模型时监控性能指标

- [教程概述](javascript:;)
- [Keras指标](javascript:;)
- [Keras为回归问题提供的性能评估指标](javascript:;)
- [Keras为分类问题提供的性能评估指标](javascript:;)
- [Keras中的自定义性能评估指标](javascript:;)
- [延伸阅读](javascript:;)
- [总结](javascript:;)

Keras库提供了一套供深度学习模型训练时的用于监控和汇总的标准性能指标并且开放了接口给开发者使用。

除了为分类和回归问题提供标准的指标以外，Keras还允许用户自定义指标。这使我们可以在模型训练的过程中实时捕捉模型的性能变化，为训练模型提供了很大的便利。

在本教程中，我会告诉你如何在使用Keras进行深度学习时添加内置指标以及自定义指标并监控这些指标。

完成本教程后，你将掌握以下知识：

- Keras计算模型指标的工作原理，以及如何在训练模型的过程中监控这些指标。
- 通过实例掌握Keras为分类问题和回归问题提供的性能评估指标的使用方法。
- 通过实例掌握Keras自定义指标的方法。



## 教程概述

本教程可以分为以下4个部分：

1. Keras指标（Metrics）
2. Keras为回归问题提供的性能评估指标
3. Keras为分类问题提供的性能评估指标
4. Keras中的自定义性能评估指标

## Keras指标

Keras允许你在训练模型期间输出要监控的指标。

您可以通过设定“ *metrics* ”参数并向模型的*compile（）*函数提供函数名（或函数别名）列表来完成此操作。

例如：

```js
model.compile(..., metrics=['mse'])
```

列出的具体指标可以是*Keras*函数的名称（如*mean_squared_error*）或这些函数的字符串别名（如' *mse* '）。

每当训练数据集中有一个epoch训练完成后，此时的性能参数会被记录下来。如果提供了验证数据集，验证数据集中的性能评估参数也会一并计算出来。

性能评估指标可以通过输出查看，也可以通过调用模型类的`fit()`方法获得。这两种方式里，性能评估函数都被当做关键字使用。如果要查看验证数据集的指标，只要在关键字前加上`val_`前缀即可。

损失函数和Keras明确定义的性能评估指标都可以当做训练中的性能指标使用。

## Keras为回归问题提供的性能评估指标

以下是Keras为回归问题提供的性能评估指标。

- **均方误差**：mean_squared_error，MSE或mse
- **平均绝对误差**：mean_absolute_error，MAE，mae
- **平均绝对误差百分比**：mean_absolute_percentage_error，MAPE，mape
- **Cosine距离**：cosine_proximity，cosine

下面通过演示来观察一下回归问题中这四个内建的性能评估指标随训练批次增加发生的变化。

```js
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['mean_squared_error'])
pyplot.plot(history.history['mean_absolute_error'])
pyplot.plot(history.history['mean_absolute_percentage_error'])
pyplot.plot(history.history['cosine_proximity'])
pyplot.show()
```

运行实例在每个epoch结束时打印性能评估指标。

```js
...
Epoch 96/100
0s - loss: 1.0596e-04 - mean_squared_error: 1.0596e-04 - mean_absolute_error: 0.0088 - mean_absolute_percentage_error: 3.5611 - cosine_proximity: -1.0000e+00
Epoch 97/100
0s - loss: 1.0354e-04 - mean_squared_error: 1.0354e-04 - mean_absolute_error: 0.0087 - mean_absolute_percentage_error: 3.5178 - cosine_proximity: -1.0000e+00
Epoch 98/100
0s - loss: 1.0116e-04 - mean_squared_error: 1.0116e-04 - mean_absolute_error: 0.0086 - mean_absolute_percentage_error: 3.4738 - cosine_proximity: -1.0000e+00
Epoch 99/100
0s - loss: 9.8820e-05 - mean_squared_error: 9.8820e-05 - mean_absolute_error: 0.0085 - mean_absolute_percentage_error: 3.4294 - cosine_proximity: -1.0000e+00
Epoch 100/100
0s - loss: 9.6515e-05 - mean_squared_error: 9.6515e-05 - mean_absolute_error: 0.0084 - mean_absolute_percentage_error: 3.3847 - cosine_proximity: -1.0000e+00
```

 在所有的epoch运行完毕后会创建这四个性能指标的折线图。

![img](https://ask.qcloudimg.com/http-save/410635/nmf5ocstyh.png?imageView2/2/w/1620)Keras为回归问题提供的四个性能评估指标随epoch完成个数变化的折线图

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

也可以使用损失函数作为度量标准。

如下所示，使用均方对数误差（*mean_squared_logarithmic_error*，*MSLE*或*msle*）损失函数作为度量标准：

```js
model.compile(loss='mse', optimizer='adam', metrics=['msle'])
```

## Keras为分类问题提供的性能评估指标

以下是Keras为分类问题提供的性能评估指标。

- **对二分类问题,计算在所有预测值上的平均正确率**：binary_accuracy，acc
- **对多分类问题,计算再所有预测值上的平均正确率**：categorical_accuracy，acc
- **在稀疏情况下，多分类问题预测值的平均正确率**：sparse_categorical_accuracy
- **计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确**：top_k_categorical_accuracy（需要手动指定k值）
- **在稀疏情况下的top-k正确率**：sparse_top_k_categorical_accuracy（需要手动指定k值）

准确性是一个特别的性能指标。

无论您的问题是二元还是多分类问题，都可以指定“ *acc* ”指标来评估准确性。

下面通过实例演示来观察Keras内置的准确度指标随训练批次增加的变化情况。

```js
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# train model
history = model.fit(X, y, epochs=400, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['acc'])
pyplot.show()
```

运行实例在每个epoch结束时打印当前的准确度。

```js
...
Epoch 396/400
 - 0s - loss: 0.5474 - acc: 1.0000
Epoch 397/400
 - 0s - loss: 0.5470 - acc: 1.0000
Epoch 398/400
 - 0s - loss: 0.5466 - acc: 1.0000
Epoch 399/400
 - 0s - loss: 0.5462 - acc: 1.0000
Epoch 400/400
 - 0s - loss: 0.5458 - acc: 1.0000
```

 在所有的epoch运行完毕后会创建精确度变化的折线图。

## Keras中的自定义性能评估指标

除了官方提供的标准性能评估指标之外，你还可以自定义自己的性能评估指标，然后再调用`compile()`函数时在`metrics`参数中指定函数名。

我经常喜欢增加的自定义指标是均方根误差（RMSE）。

你可以通过观察官方提供的性能评估指标函数来学习如何编写自定义指标。

下面展示的是[Keras中mean_squared_error损失函数（即均方差性能评估指标）](https://github.com/fchollet/keras/blob/master/keras/losses.py)的代码。

```js
def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
```

K是Keras使用的后端(例如TensorFlow)。

从这个例子以及其他损失函数和性能评估指标可以看出：需要使用后端提供的标准数学函数来计算我们感兴趣的性能评估指标。

现在我们可以尝试编写一个自定义性能评估函数来计算RMSE，如下所示：

```js
from keras import backend
 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
 
```

你可以看到除了用`sqrt()`函数封装结果之外，这个函数的代码和MSE是一样的。

我们可以通过一个简单的回归问题来测试这个性能评估函数。注意这里我们不再通过字符串提供给Keras来解析为对应的处理函数，而是直接设定为我们编写的自定义函数。

```js
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from keras import backend
 
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
 
# prepare sequence
X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# create model
model = Sequential()
model.add(Dense(2, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=[rmse])
# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
# plot metrics
pyplot.plot(history.history['rmse'])
pyplot.show()
```

同样地，在每个epoch结束时会打印均方误差值。

```js
...
Epoch 496/500
0s - loss: 1.2992e-06 - rmse: 9.7909e-04
Epoch 497/500
0s - loss: 1.2681e-06 - rmse: 9.6731e-04
Epoch 498/500
0s - loss: 1.2377e-06 - rmse: 9.5562e-04
Epoch 499/500
0s - loss: 1.2079e-06 - rmse: 9.4403e-04
Epoch 500/500
0s - loss: 1.1788e-06 - rmse: 9.3261e-04
```

在运行结束时可以得到自定义性能评估指标——均方误差的折线图。

![img](https://ask.qcloudimg.com/http-save/410635/6mrkadse8r.png?imageView2/2/w/1620)自定义性能评估指标——均方误差的折线图

你的自定义性能评估函数必须在Keras的内部数据结构上进行操作而不能直接在原始的数据进行操作，具体的操作方法取决于你使用的后端（如果使用TensorFlow，那么对应的就是`tensorflow.python.framework.ops.Tensor`）。

由于这个原因，我建议最好使用后端提供的数学函数来进行计算，这样可以保证一致性和运行速度。

## 延伸阅读

如果你想继续深入了解，下面有我推荐的一些资源以供参考。

- [Keras Metrics API文档](https://keras.io/metrics/)
- [Keras Metrics的源代码](https://github.com/fchollet/keras/blob/master/keras/metrics.py)
- [Keras Loss API文档](https://keras.io/losses/)
- [Keras Loss的源代码](https://github.com/fchollet/keras/blob/master/keras/losses.py)

## 总结

在本教程中，你应该已经了解到了如何在训练深度学习模型时使用Keras提供的性能评估指标接口。

具体来说，你应该掌握以下内容：

- Keras的性能评估指标的工作原理，以及如何配置模型在训练过程中输出性能评估指标。
- 如何使用Keras为分类问题和回归问题提供的性能评估指标。
- 如何有效地定义和使用自定义性能指标。



## 回归预测的评价指标

**问题1：**对于同一个数据集，两个模型，同样多的预测错误数，但前者大部分错在大样本值上，后者大部分错在小样本值上，如何比较二者优劣？？--预测误差大不相同，但可能实际效果差不多

#### 1. MSE: Mean Squared Error (均方误差)

$$
MSE = \frac{1}{n} \sum_{i=0}^{n-1}(y_{true}-y_{pred})^2
$$

#### 2. MAE: Mean Absolute Error (平均绝对误差)

$$
MAE = \frac{1}{n} \sum_{i=0}^{n-1}|y_{true}-y_{pred}|
$$

#### 3. MSLE: Mean Squared Logarithmic Error (均方指数误差)

$$
MSLE =\frac{1}{n} \sum_{i=0}^{n-1}(log_e(1+y_{true})-log_e(1+y_{pred}))^2
$$

优点：当目标实现指数增长时，例如人口数量、商品几年内的平均销量这种样本值差距巨大，或样本波动较大的情况，适合使用此指标，缩小Loss function的值，将较大值映射到一个较小的波动范围

#### 4. MedAE: Media Absolute Error (中位数绝对误差)

$$
MedAE = media(|y_{1true}-y_{1pred}|,\dots, |y_{ntrue}-y_{npred}|)
$$

优点：用于包含异常值的数据的衡量

 

#### 5. EVS: Explained Variance Score (解释方差分数)

$$
EVS = 1- \frac{Var(y_{true}-y_{pred})}{Var(y_{true})}
$$

解释：衡量模型对数据集预测结果的波动程度（用预测误差的方差，MSE,MAE差不多时，即误差的均值差不多时，看谁错误的波动大， [1,1,1,1] [0,0,0,4] ,但没准最后一个是坏数据呢？）

标准：越接近1，效果越好

**EVA: Explained Variance Ratio**
$$
EVA = \frac{Var[y_{true}-y_{pred}]}{Var[y_{true}]}
$$


#### 6. SSE: The Sum of Squares due to Error (误差/残差平方和)

$$
SSE = \sum(y_{true}-y_{pred})^2
$$

标准：同一数据集下，SSE值越小，误差越小，代表拟合效果越好 （不一定）

缺点：不同数据集之间，SSE的值没有意义，因为其值不仅取决于预测值，还取决于样本数量



#### 7. R-square: Coefficient of determination (决定系数)

$$
R^2 = 1-\frac{\sum (y_{true}-y_{pred})^2 }{\sum (y_{true}-y_{mean})^2}
$$

解释：分母为原始数据的离散程度，分子为预测数据与原始数据的误差，(分母一般肯定大于分子) 二者相除用来消除原始数据离散程度对结果造成的影响，这样当原始数据集离散程度过大或过小时，都能得到类似的$R^2$值。（貌似仍然不能解决问题1）

标准：越接近1越好 （经验值>0.4拟合效果好）

缺点：数据集样本数量越大，$R^2$越大，不同数据集模型结果比较会有一定误差 （可以用于交叉验证？样本数量相同但分布可能不同，防止由于数据分布不同产生不同的预测误差）--但当数据分布相同时（分母相同，有的错误值在大数据上，有的在小数据上，如何衡量？？看样本分布是小数据多还是大数据多，用误差的方差来衡量EVA; 但这样的前提是预测数据的分布，只要大部分数据预测正确就可以；如果占少量的样本也非常重要的话，比如急刹车，这样就要考虑新的评价标准，每一个 预测错误都同等重要，而不是预测的值）



#### 3. Adjusted R-square: Degree-of-freedom adjusted coefficient of determination (校正决定系数)

$$
R^2_{adjust} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
$$

解释：n为样本数量，p为模型变量数量；消除样本数据和变量数量的影响 （样本数较少时，对（1-R^2）放大）



评价指标面临的关键挑战：

- 数据值大小不同对误差的影响[1, 5] [1.1, 5.5] [1.5, 5.1]
- 坏数据的存在，不能因为少量坏数据造成的预测错误就放大此错误
- 每个数据都同等重要，不能因为少量数据不属于数据集主要分布，预测错误就不重要 （EVA值小，只能代表大部分数据预测符合数据集分布，并不能代表真的好）

解决方案：

- 
- 统计预测错误数，随着错误数的增加，对误差施加的权重/累积误差也增加，刚开始少数错误可以忽略，因为可能是坏数据

$$
\frac{1}{n} \sum_{i=0}^{n-1}\frac{(y_{true}-y_{pred})^2}{(y_{true})^2}
$$



## 分类模型评测指标

准确率：预测正确的样本占总样本的比例

问题：**假如有100个样本，其中1个正样本，99个负样本，如果模型的预测只输出0，那么正确率是99%，这时候用正确率来衡量模型的好坏显然是不对的。**

查准率： 预测值为1且真实值为1的样本在**所有预测值为1的样本**中所占的比例， 即挑出来的好瓜中有多大的比例是好瓜；

召回率：预测值为1且真实值为1的样本在所有真实值为1的样本中所占的比例，即真正的好瓜有多大的比例被挑出来了。

F1分数（F1-Score），又称为平衡F分数（BalancedScore），它被定义为精确率和召回率的调和平均数。

<https://blog.csdn.net/zjn295771349/article/details/84961596>