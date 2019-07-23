# Keras 核心网络层

所有 Keras 网络层都有很多共同的函数：

- `layer.get_weights()`: 以含有Numpy矩阵的列表形式返回层的权重。
- `layer.set_weights(weights)`: 从含有Numpy矩阵的列表中设置层的权重（与`get_weights`的输出形状相同）。
- `layer.get_config()`: 返回包含层配置的字典。此图层可以通过以下方式重置：

```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

或:

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

如果一个层具有单个节点 (i.e. 如果它不是共享层), 你可以得到它的输入张量、输出张量、输入尺寸和输出尺寸:

- `layer.input`
- `layer.output`
- `layer.input_shape`
- `layer.output_shape`

如果层有多个节点 (参见: [层节点和共享层的概念](/getting-started/functional-api-guide/#the-concept-of-layer-node)), 您可以使用以下函数:

- `layer.get_input_at(node_index)`
- `layer.get_output_at(node_index)`
- `layer.get_input_shape_at(node_index)`
- `layer.get_output_shape_at(node_index)`



<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L767)</span>

### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

就是你常用的的全连接层。

`Dense` 实现以下操作：
`output = activation(dot(input, kernel) + bias)`
其中 `activation` 是按逐个元素计算的激活函数，`kernel` 是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量 (只在 `use_bias` 为 `True` 时才有用)。

- __注意__: 如果该层的输入的秩大于2，那么它首先被展平然后再计算与 `kernel` 的点乘。

__例__

```python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
# 其输出数组的尺寸为 (*, 32)

# 在第一层之后，你就不再需要指定输入的尺寸了：
model.add(Dense(32))
```

__参数__

- __units__: 正整数，输出空间维度。
- __activation__: 激活函数(详见 [activations](../activations.md))。
  **若不指定，则不使用激活函数** (即，「线性」激活: `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (see [initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向的的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层的输出的正则化函数(它的 "activation")。
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

nD 张量，尺寸: `(batch_size, ..., input_dim)`。
最常见的情况是一个尺寸为 `(batch_size, input_dim)`
的 2D 输入。

__输出尺寸__

nD 张量，尺寸: `(batch_size, ..., units)`。
例如，对于尺寸为 `(batch_size, input_dim)` 的 2D 输入，
输出的尺寸为 `(batch_size, units)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L276)</span>

### Activation

```python
keras.layers.Activation(activation)
```

将激活函数应用于输出。

__参数__

- __activation__: 要使用的激活函数的名称
  (详见: [activations](../activations.md))，
  或者选择一个 Theano 或 TensorFlow 操作。

__输入尺寸__

任意尺寸。
当使用此层作为模型中的第一层时，使用参数 `input_shape`（整数元组，不包括样本数的轴）。

__输出尺寸__

与输入相同。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L80)</span>

### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

将 Dropout 应用于输入。

Dropout 包括在训练中每次更新时，将输入单元的按比率随机设置为 0，这有助于防止过拟合。

__参数__

- __rate__: 在 0 和 1 之间浮动。需要丢弃的输入比例。
- __noise_shape__: 1D 整数张量，
  表示将与输入相乘的二进制 dropout 掩层的形状。
  例如，如果你的输入尺寸为`(batch_size, timesteps, features)`，然后你希望 dropout 掩层在所有时间步都是一样的，你可以使用 `noise_shape=(batch_size, 1, features)`。
- __seed__: 一个作为随机种子的 Python 整数。

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L461)</span>

### Flatten

```python
keras.layers.Flatten(data_format=None)
```

将输入展平。不影响批量大小。

__参数__

- __data_format__：一个字符串，其值为 `channels_last`（默认值）或者 `channels_first`。它表明输入的维度的顺序。此参数的目的是当模型从一种数据格式切换到另一种数据格式时保留权重顺序。
  - `channels_last` 对应着尺寸为 `(batch, ..., channels)` 的输入，而 `channels_first` 对应着尺寸为 `(batch, channels, ...)` 的输入。
  - 默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__例__

```python
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# 现在：model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# 现在：model.output_shape == (None, 65536)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/input_layer.py#L114)</span>

### Input

```python
keras.engine.input_layer.Input()
```

`Input()` 用于实例化 Keras 张量。

Keras 张量是底层后端(Theano, TensorFlow 或 CNTK) 的张量对象，我们增加了一些特性，使得能够通过了解模型的输入和输出来构建 Keras 模型。

例如，如果 a, b 和 c 都是 Keras 张量，那么以下操作是可行的：
`model = Model(input=[a, b], output=c)`

添加的 Keras 属性是：

- __`_keras_shape`__: 通过 Keras端的尺寸推理
  进行传播的整数尺寸元组。
- __`_keras_history`__: 应用于张量的最后一层。
  整个网络层计算图可以递归地从该层中检索。

__参数__

- __shape__: 一个尺寸元组（整数），不包含批量大小。
  例如，`shape=(32,)` 表明期望的输入是按批次的 32 维向量。
- __batch_shape__: 一个尺寸元组（整数），包含批量大小。
  例如，`batch_shape=(10, 32)` 表明期望的输入是 10 个 32 维向量。`batch_shape=(None, 32)` 表明任意批次大小的 32 维向量。
- __name__: 一个可选的层的名称的字符串。
  在一个模型中应该是唯一的（不可以重用一个名字两次）。如未提供，将自动生成。
- __dtype__: 输入所期望的数据类型，字符串表示
  (`float32`, `float64`, `int32`...)
- __sparse__: 一个布尔值，指明需要创建的占位符是否是稀疏的。
- __tensor__: 可选的可封装到 `Input` 层的现有张量。
  如果设定了，那么这个层将不会创建占位符张量。

__返回__

一个张量。

__例__

```python
# 这是 Keras 中的一个逻辑回归
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L310)</span>

### Reshape

```python
keras.layers.Reshape(target_shape)

```

将输入重新调整为特定的尺寸。

__参数__

- __target_shape__: 目标尺寸。整数元组。
  **不包含表示批量的轴。**

__输入尺寸__

任意，尽管输入尺寸中的所有维度必须是固定的。
当使用此层作为模型中的第一层时，使用参数 `input_shape`（整数元组，不包括样本数的轴）。

__输出尺寸__

`(batch_size,) + target_shape`

__例__

```python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# 现在：model.output_shape == (None, 3, 4)
# 注意： `None` 是批表示的维度

# 作为 Sequential 模型的中间层
model.add(Reshape((6, 2)))
# 现在： model.output_shape == (None, 6, 2)

# 还支持使用 `-1` 表示维度的尺寸推断
model.add(Reshape((-1, 2, 2)))
# 现在： model.output_shape == (None, 3, 2, 2)

```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L409)</span>

### Permute

```python
keras.layers.Permute(dims)

```

根据给定的模式**置换输入的维度**。

在某些场景下很有用，例如将 RNN 和 CNN 连接在一起。

__例__

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 现在： model.output_shape == (None, 64, 10)
# 注意： `None` 是批表示的维度

```

__参数__

- __dims__: 整数元组。置换模式，不包含样本维度。
  索引从 1 开始。
  例如, `(2, 1)` 置换输入的第一和第二个维度。

__输入尺寸__

任意。当使用此层作为模型中的第一层时，使用参数 `input_shape`（整数元组，不包括样本数的轴）。

__输出尺寸__

与输入尺寸相同，但是维度根据指定的模式重新排列。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L523)</span>

### RepeatVector

```python
keras.layers.RepeatVector(n)

```

将输入重复 n 次。

__例__

```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# 现在： model.output_shape == (None, 32)
# 注意： `None` 是批表示的维度

model.add(RepeatVector(3))
# 现在： model.output_shape == (None, 3, 32)

```

__参数__

- __n__: 整数，重复次数。

__输入尺寸__

2D 张量，尺寸为 `(num_samples, features)`。

__输出尺寸__

3D 张量，尺寸为 `(num_samples, n, features)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L565)</span>

### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)

```

将任意表达式封装为 `Layer` 对象。

__例__

```python
# 添加一个 x -> x^2 层
model.add(Lambda(lambda x: x ** 2))

```

```python
# 添加一个网络层，返回输入的正数部分
# 与负数部分的反面的连接

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))

```

__参数__

- __function__: 需要封装的函数。
  将输入张量作为第一个参数。
- __output_shape__: 预期的函数输出尺寸。
  只在使用 Theano 时有意义。
  可以是元组或者函数。
  如果是元组，它只指定第一个维度；
  ​    样本维度假设与输入相同：
  ​    `output_shape = (input_shape[0], ) + output_shape`
  ​    或者，输入是 `None` 且样本维度也是 `None`：
  ​    `output_shape = (None, ) + output_shape`
  ​    如果是函数，它指定整个尺寸为输入尺寸的一个函数：
  ​    `output_shape = f(input_shape)`
- __arguments__: 可选的需要传递给函数的关键字参数。

__输入尺寸__

任意。当使用此层作为模型中的第一层时，使用参数 `input_shape`（整数元组，不包括样本数的轴）。

__输出尺寸__

由 `output_shape` 参数指定 (或者在使用 TensorFlow 时，自动推理得到)。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L911)</span>

### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)

```

网络层，对基于代价函数的输入活动应用一个更新。

__参数__

- __l1__: L1 正则化因子 (正数浮点型)。
- __l2__: L2 正则化因子 (正数浮点型)。

__输入尺寸__

任意。

当使用此层作为模型中的第一层时，使用参数 `input_shape`（整数元组，不包括样本数的轴）。

__输出尺寸__

与输入相同。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L28)</span>

### Masking

```python
keras.layers.Masking(mask_value=0.0)

```

使用覆盖值覆盖序列，以跳过时间步。

对于输入张量的每一个时间步（张量的第一个维度），如果所有时间步中输入张量的值与 `mask_value` 相等，
那么这个时间步将在所有下游层被覆盖 (跳过)（只要它们支持覆盖）。如果任何下游层不支持覆盖但仍然收到此类输入覆盖信息，会引发异常。

__例__

考虑将要喂入一个 LSTM 层的 Numpy 矩阵 `x`，尺寸为 `(samples, timesteps, features)`。
你想要覆盖时间步 #3 和 #5，因为你缺乏这几个时间步的数据。你可以：

- 设置 `x[:, 3, :] = 0.` 以及 `x[:, 5, :] = 0.`
- 在 LSTM 层之前，插入一个 `mask_value=0` 的 `Masking` 层：

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))

```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L140)</span>

### SpatialDropout1D

```python
keras.layers.SpatialDropout1D(rate)

```

Dropout 的 Spatial 1D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 1D 的特征图而不是丢弃单个元素。如果特征图中相邻的帧是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout1D 将有助于提高特征图之间的独立性，应该使用它来代替 Dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。

__输入尺寸__

3D 张量，尺寸为：`(samples, timesteps, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L177)</span>

### SpatialDropout2D

```python
keras.layers.SpatialDropout2D(rate, data_format=None)

```

Dropout 的 Spatial 2D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 2D 的特征图而不是丢弃单个元素。如果特征图中相邻的像素是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout2D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。
- __data_format__：`channels_first` 或者 `channels_last`。在 `channels_first`  模式中，通道维度（即深度）位于索引 1，在 `channels_last` 模式中，通道维度位于索引 3。默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__输入尺寸__

4D 张量，如果 data_format＝`channels_first`，尺寸为 `(samples, channels, rows, cols)`，如果 data_format＝`channels_last`，尺寸为 `(samples, rows, cols, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L227)</span>

### SpatialDropout3D

```python
keras.layers.SpatialDropout3D(rate, data_format=None)

```

Dropout 的 Spatial 3D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 3D 的特征图而不是丢弃单个元素。如果特征图中相邻的体素是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout3D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。
- __data_format__：`channels_first` 或者 `channels_last`。在 `channels_first`  模式中，通道维度（即深度）位于索引 1，在 `channels_last` 模式中，通道维度位于索引 4。默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__输入尺寸__

5D 张量，如果 data_format＝`channels_first`，尺寸为 `(samples, channels, dim1, dim2, dim3)`，如果 data_format＝`channels_last`，尺寸为 `(samples, dim1, dim2, dim3, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)



# Convolutional layer

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L241)</span>

### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 卷积层 (例如**时序卷积**)。

该层创建了一个卷积核，该卷积核以单个空间（或时间）维上的层输入进行卷积，以生成输出张量。
如果 `use_bias` 为 True，则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组或 `None`），例如，
`(10, 128)` 表示 10 个 128 维的向量组成的向量序列，
`(None, 128)` 表示 128 维的向量组成的变长序列。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
  指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
  指明卷积的步长。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"`, `"causal"` 或 `"same"` 之一 (大小写敏感)
  `"valid"` 表示「不填充」。
  `"same"` 表示填充输入以使输出具有与原始输入相同的长度。
  `"causal"` 表示因果（膨胀）卷积，
  例如，`output[t]` 不依赖于 `input[t+1:]`，
  在模型不应违反时间顺序的时间数据建模时非常有用。
  详见 [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)。
- __data_format__: 字符串,
  `"channels_last"` (默认) 或 `"channels_first"` 之一。输入的各个维度顺序。
  `"channels_last"` 对应输入尺寸为 `(batch, steps, channels)`
  (Keras 中时序数据的默认格式)
  而 `"channels_first"` 对应输入尺寸为 `(batch, channels, steps)`。
- __dilation_rate__: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。
  当前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如未指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

3D 张量 ，尺寸为 `(batch_size, steps, input_dim)`。

__输出尺寸__

3D 张量，尺寸为 `(batch_size, new_steps, filters)`。
由于填充或窗口按步长滑动，`steps` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L367)</span>

### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 卷积层 (例如对图像的空间卷积)。

该层创建了一个卷积核，该卷积核对层输入进行卷积，以生成输出张量。
如果 `use_bias` 为 True，则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 3)` 表示 128x128 RGB 图像，在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
  指明 2D 卷积窗口的宽度和高度。
  可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
  指明卷积沿宽度和高度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 `channels_last`。
- __dilation_rate__: 一个整数或 2 个整数的元组或列表，
  指定膨胀卷积的膨胀率。
  可以是一个整数，为所有空间维度指定相同的值。
  当前，指定任何 `dilation_rate` 值 != 1 与
  指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 4D 张量，尺寸为 `(samples, channels, rows, cols)`。
- 如果 data_format='channels_last'，
  输入 4D 张量，尺寸为 `(samples, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 4D 张量，尺寸为 `(samples, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
  输出 4D 张量，尺寸为 `(samples, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1420)</span>

### SeparableConv1D

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

深度方向的可分离 1D 卷积。

可分离的卷积的操作包括，首先执行深度方向的空间卷积（分别作用于每个输入通道），紧接一个将所得输出通道
混合在一起的逐点卷积。`depth_multiplier` 参数控制深度步骤中每个输入通道生成多少个输出通道。

直观地说，可分离的卷积可以理解为一种将卷积核分解成两个较小的卷积核的方法，或者作为 Inception 块的一个极端版本。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
  指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
  指明卷积的步长。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数，或者单个整数表示的元组或列表，
  为使用扩张（空洞）卷积指明扩张率。
  目前，指定任何 `dilation_rate` 值 != 1 与指定任何 `stride` 值 != 1 两者不兼容。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
  深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __pointwise_initializer__: 运用到逐点核矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __pointwise_regularizer__: 运用到逐点核矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __pointwise_constraint__: 运用到逐点核矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 3D 张量，尺寸为 `(batch, channels, steps)`。
- 如果 data_format='channels_last'，
  输入 3D 张量，尺寸为 `(batch, steps, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 3D 张量，尺寸为 `(batch, filters, new_steps)`。
- 如果 data_format='channels_last'，
  输出 3D 张量，尺寸为 `(batch, new_steps, filters)`。

由于填充的原因， `new_steps` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1552)</span>

### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

深度方向的可分离 2D 卷积。

可分离的卷积的操作包括，首先执行深度方向的空间卷积
（分别作用于每个输入通道），紧接一个将所得输出通道
混合在一起的逐点卷积。`depth_multiplier` 参数控
制深度步骤中每个输入通道生成多少个输出通道。

直观地说，可分离的卷积可以理解为一种将卷积核分解成
两个较小的卷积核的方法，或者作为 Inception 块的
一个极端版本。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
  指明 2D 卷积窗口的高度和宽度。
  可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
  指明卷积沿高度和宽度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数，或者 2 个整数表示的元组或列表，
  为使用扩张（空洞）卷积指明扩张率。
  目前，指定任何 `dilation_rate` 值 != 1 与指定任何 `stride` 值 != 1 两者不兼容。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
  深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
  详见 [initializers](../initializers.md))。
- __pointwise_initializer__: 运用到逐点核矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __pointwise_regularizer__: 运用到逐点核矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __pointwise_constraint__: 运用到逐点核矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
  输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
  输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1693)</span>

### DepthwiseConv2D

```python
keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
```

深度可分离 2D 卷积。

深度可分离卷积包括仅执行深度空间卷积中的第一步（其分别作用于每个输入通道）。
`depth_multiplier` 参数控制深度步骤中每个输入通道生成多少个输出通道。

__Arguments__

- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
  指明 2D 卷积窗口的高度和宽度。
  可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
  指明卷积沿高度和宽度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
  深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用「channels_last」。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
  详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
  输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
  输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L627)</span>

### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

转置卷积层 (有时被成为反卷积)。

对转置卷积的需求一般来自希望使用与正常卷积相反方向的变换，即，将具有卷积输出尺寸的东西转换为具有卷积输入尺寸的东西，同时保持与所述卷积相容的连通性模式。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 3)` 表示 128x128 RGB 图像，在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
  指明 2D 卷积窗口的高度和宽度。
  可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
  指明卷积沿高度和宽度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __output_padding__: 一个整数，或者 2 个整数表示的元组或列表，
  指定沿输出张量的高度和宽度的填充量。
  可以是单个整数，以指定所有空间维度的相同值。
  沿给定维度的输出填充量必须低于沿同一维度的步长。
  如果设置为 `None` (默认), 输出尺寸将自动推理出来。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。
- __dilation_rate__: 一个整数或 2 个整数的元组或列表，
  指定膨胀卷积的膨胀率。
  可以是一个整数，为所有空间维度指定相同的值。
  当前，指定任何 `dilation_rate` 值 != 1 与
  指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
  输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
  输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

如果指定了 `output_padding`:

```python
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```

__参考文献__

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L498)</span>

### Conv3D

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3D 卷积层 (例如立体空间卷积)。

该层创建了一个卷积核，该卷积核对层输入进行卷积，以生成输出张量。
如果 `use_bias` 为 True，则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 128, 1)` 表示 128x128x128 的单通道立体，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 3 个整数表示的元组或列表，
  指明 3D 卷积窗口的深度、高度和宽度。可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 3 个整数表示的元组或列表，
  指明卷积沿每一个空间维度的步长。
  可以是一个整数，为所有空间维度指定相同的步长值。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。
- __dilation_rate__: 一个整数或 3 个整数的元组或列表，
  指定膨胀卷积的膨胀率。
  可以是一个整数，为所有空间维度指定相同的值。
  当前，指定任何 `dilation_rate` 值 != 1 与
  指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 5D 张量，尺寸为 `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`。
- 如果 data_format='channels_last'，
  输入 5D 张量，尺寸为 `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
  输出 5D 张量，尺寸为 `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`。
- 如果 data_format='channels_last'，
  输出 5D 张量，尺寸为 `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)`。

由于填充的原因， `new_conv_dim1`, `new_conv_dim2` 和 `new_conv_dim3` 值可能已更改。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L900)</span>

### Conv3DTranspose

```python
keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

```

转置卷积层 (有时被成为反卷积)。

对转置卷积的需求一般来自希望使用
与正常卷积相反方向的变换，
即，将具有卷积输出尺寸的东西
转换为具有卷积输入尺寸的东西，
同时保持与所述卷积相容的连通性模式。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数
（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 128, 3)` 表示尺寸 128x128x128 的 3 通道立体，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度 
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 3 个整数表示的元组或列表，
  指明 3D 卷积窗口的深度、高度和宽度。
  可以是一个整数，为所有空间维度指定相同的值。 
- __strides__: 一个整数，或者 3 个整数表示的元组或列表，
  指明沿深度、高度和宽度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
  指定任何 `stride` 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __output_padding__: 一个整数，或者 3 个整数表示的元组或列表，
  指定沿输出张量的高度和宽度的填充量。
  可以是单个整数，以指定所有空间维度的相同值。
  沿给定维度的输出填充量必须低于沿同一维度的步长。
  如果设置为 `None` (默认), 输出尺寸将自动推理出来。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, depth, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, depth, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数或 3 个整数的元组或列表，
  指定膨胀卷积的膨胀率。
  可以是一个整数，为所有空间维度指定相同的值。
  当前，指定任何 `dilation_rate` 值 != 1 与
  指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

如果 data_format='channels_first'， 输入 5D 张量，尺寸为
`(batch, channels, depth, rows, cols)`，
如果 data_format='channels_last'， 输入 5D 张量，尺寸为
`(batch, depth, rows, cols, channels)`。

__Output shape__

如果 data_format='channels_first'， 输出 5D 张量，尺寸为
`(batch, filters, new_depth, new_rows, new_cols)`，
如果 data_format='channels_last'， 输出 5D 张量，尺寸为
`(batch, new_depth, new_rows, new_cols, filters)`。

`depth` 和 `rows` 和 `cols` 可能因为填充而改变。
如果指定了 `output_padding`：

```python
new_depth = ((depth - 1) * strides[0] + kernel_size[0]
             - 2 * padding[0] + output_padding[0])
new_rows = ((rows - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
new_cols = ((cols - 1) * strides[2] + kernel_size[2]
            - 2 * padding[2] + output_padding[2])

```

__参考文献__

- [A guide to convolution arithmetic for deep learning]
  (https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks]
  (http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2375)</span>

### Cropping1D

```python
keras.layers.Cropping1D(cropping=(1, 1))

```

1D 输入的裁剪层（例如时间序列）。

它沿着时间维度（第 1 个轴）裁剪。

__参数__

- __cropping__: 整数或整数元组（长度为 2）。
  在裁剪维度（第 1 个轴）的开始和结束位置应该裁剪多少个单位。
  如果只提供了一个整数，那么这两个位置将使用相同的值。

__输入尺寸__

3D 张量，尺寸为 `(batch, axis_to_crop, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, cropped_axis, features)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2407)</span>

### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)

```

2D 输入的裁剪层（例如图像）。

它沿着空间维度裁剪，即宽度和高度。

__参数__

- __cropping__: 整数，或 2 个整数的元组，或 2 个整数的 2 个元组。
  - 如果为整数： 将对宽度和高度应用相同的对称裁剪。
  - 如果为 2 个整数的元组：
    解释为对高度和宽度的两个不同的对称裁剪值：
    `(symmetric_height_crop, symmetric_width_crop)`。
  - 如果为 2 个整数的 2 个元组：
    解释为 `((top_crop, bottom_crop), (left_crop, right_crop))`。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

- 如果 data_format='channels_last'，
  输出 4D 张量，尺寸为 `(batch, rows, cols, channels)`。
- 如果 data_format='channels_first'，
  输出 4D 张量，尺寸为 `(batch, channels, rows, cols)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输出 4D 张量，尺寸为 `(batch, cropped_rows, cropped_cols, channels)`
- 如果 `data_format` 为 `"channels_first"`，
  输出 4D 张量，尺寸为 `(batch, channels, cropped_rows, cropped_cols)`。

__例子__

```python
# 裁剪输入的 2D 图像或特征图
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# 现在 model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same')) 
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# 现在 model.output_shape == (None, 20, 16. 64)

```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2490)</span>

### Cropping3D

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)

```

3D 数据的裁剪层（例如空间或时空）。

__参数__

- __cropping__: 整数，或 3 个整数的元组，或 2 个整数的 3 个元组。
  - 如果为整数： 将对深度、高度和宽度应用相同的对称裁剪。
  - 如果为 3 个整数的元组：
    解释为对深度、高度和高度的 3 个不同的对称裁剪值：
    `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`。
  - 如果为 2 个整数的 3 个元组：
    解释为 `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
  `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`
- 如果 `data_format` 为 `"channels_first"`: 
  `(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`

__输出尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
  `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`
- 如果 `data_format` 为 `"channels_first"`: 
  `(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1943)</span>

### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)

```

1D 输入的上采样层。

沿着时间轴重复每个时间步 `size` 次。

__参数__

- __size__: 整数。上采样因子。

__输入尺寸__

3D 张量，尺寸为 `(batch, steps, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, upsampled_steps, features)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1973)</span>

### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')

```

2D 输入的上采样层。

沿着数据的行和列分别重复 `size[0]` 和 `size[1]` 次。

__参数__

- __size__: 整数，或 2 个整数的元组。
  行和列的上采样因子。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。
- __interpolation__: 字符串，`nearest` 或 `bilinear` 之一。
  注意 CNTK 暂不支持 `bilinear` upscaling，
  以及对于 Theano，只可以使用 `size=(2, 2)`。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输入 4D 张量，尺寸为 
  `(batch, rows, cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输入 4D 张量，尺寸为 
  `(batch, channels, rows, cols)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输出 4D 张量，尺寸为 
  `(batch, upsampled_rows, upsampled_cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输出 4D 张量，尺寸为 
  `(batch, channels, upsampled_rows, upsampled_cols)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2031)</span>

### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)

```

3D 输入的上采样层。

沿着数据的第 1、2、3 维度分别重复 
`size[0]`、`size[1]` 和 `size[2]` 次。

__参数__

- __size__: 整数，或 3 个整数的元组。
  dim1, dim2 和 dim3 的上采样因子。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输入 5D 张量，尺寸为 
  `(batch, dim1, dim2, dim3, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输入 5D 张量，尺寸为 
  `(batch, channels, dim1, dim2, dim3)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输出 5D 张量，尺寸为 
  `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输出 5D 张量，尺寸为 
  `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2123)</span>

### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)

```

1D 输入的零填充层（例如，时间序列）。

__参数__

- __padding__: 整数，或长度为 2 的整数元组，或字典。
  - 如果为整数：
    在填充维度（第一个轴）的开始和结束处添加多少个零。
  - 如果是长度为 2 的整数元组：
    在填充维度的开始和结尾处添加多少个零 (`(left_pad, right_pad)`)。

__输入尺寸__

3D 张量，尺寸为 `(batch, axis_to_pad, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, padded_axis, features)`。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2158)</span>

### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)

```

2D 输入的零填充层（例如图像）。

该图层可以在图像张量的顶部、底部、左侧和右侧添加零表示的行和列。

__参数__

- __padding__: 整数，或 2 个整数的元组，或 2 个整数的 2 个元组。
  - 如果为整数：将对宽度和高度运用相同的对称填充。
  - 如果为 2 个整数的元组：
  - 如果为整数：: 解释为高度和高度的 2 个不同的对称裁剪值：
    `(symmetric_height_pad, symmetric_width_pad)`。
  - 如果为 2 个整数的 2 个元组：
    解释为 `((top_pad, bottom_pad), (left_pad, right_pad))`。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输入 4D 张量，尺寸为 
  `(batch, rows, cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输入 4D 张量，尺寸为 
  `(batch, channels, rows, cols)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
  输出 4D 张量，尺寸为 
  `(batch, padded_rows, padded_cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
  输出 4D 张量，尺寸为 
  `(batch, channels, padded_rows, padded_cols)`。



write your own tensor padding

<https://stackoverflow.com/questions/51810015/keras-pad-tensor-with-values-on-the-borders>

<https://stackoverflow.com/questions/49021356/how-to-set-arbitrary-padding-value-to-keras-conv2d-filter>

<https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2159>

<https://github.com/keras-team/keras/issues/9508>



------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2234)</span>

### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)

```

3D 数据的零填充层(空间或时空)。

__参数__

- __padding__: 整数，或 3 个整数的元组，或 2 个整数的 3 个元组。
  - 如果为整数：将对深度、高度和宽度运用相同的对称填充。
  - 如果为 3 个整数的元组：
    解释为深度、高度和宽度的三个不同的对称填充值：
    `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
  - 如果为 2 个整数的 3 个元组：解释为
    `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一，
  表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
  `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
  `channels_first` 对应输入尺寸为 
  `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
  `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`。
- 如果 `data_format` 为 `"channels_first"`: 
  `(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`。

__输出尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
  `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`。
- 如果 `data_format` 为 `"channels_first"`:
  `(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`。



# Pooling

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L69)</span>

### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对于时序数据的最大池化。

__参数__

- __pool_size__: 整数，最大池化的窗口大小。
- __strides__: 整数，或者是 `None`。作为缩小比例的因数。
  例如，2 会使得输入张量缩小一半。
  如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
  `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

__输入尺寸__

- 如果 `data_format='channels_last'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, features, steps)`

__输出尺寸__

- 如果 `data_format='channels_last'`，
  输出为 3D 张量，尺寸为：
  `(batch_size, downsampled_steps, features)`
- 如果 `data_format='channels_first'`，
  输出为 3D 张量，尺寸为：
  `(batch_size, features, downsampled_steps)`

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L217)</span>

### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

对于空间数据的最大池化。

__参数__

- __pool_size__: 整数，或者 2 个整数表示的元组，
  沿（垂直，水平）方向缩小比例的因数。
    （2，2）会把输入张量的两个维度都缩小一半。
  如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- __strides__: 整数，2 个整数表示的元组，或者是 `None`。
  表示步长值。
  如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L386)</span>

### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

对于 3D（空间，或时空间）数据的最大池化。

__参数__

- __pool_size__: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。
  (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
- __strides__: 3 个整数表示的元组，或者是 `None`。步长值。
- __padding__: `"valid"` 或者 `"same"`（区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L117)</span>

### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对于时序数据的平均池化。

__参数__

- __pool_size__: 整数，平均池化的窗口大小。
- __strides__: 整数，或者是 `None	`。作为缩小比例的因数。
  例如，2 会使得输入张量缩小一半。
  如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
  `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

__输入尺寸__

- 如果 `data_format='channels_last'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, features, steps)`

__输出尺寸__

- 如果 `data_format='channels_last'`，
  输出为 3D 张量，尺寸为：
  `(batch_size, downsampled_steps, features)`
- 如果 `data_format='channels_first'`，
  输出为 3D 张量，尺寸为：
  `(batch_size, features, downsampled_steps)`

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L272)</span>

### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

对于空间数据的平均池化。

__参数__

- __pool_size__: 整数，或者 2 个整数表示的元组，
  沿（垂直，水平）方向缩小比例的因数。
    （2，2）会把输入张量的两个维度都缩小一半。
  如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- __strides__: 整数，2 个整数表示的元组，或者是 `None`。
  表示步长值。
  如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L436)</span>

### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

对于 3D （空间，或者时空间）数据的平均池化。

__参数__

- __pool_size__: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。
  (2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
- __strides__: 3 个整数表示的元组，或者是 `None`。步长值。
- __padding__: `"valid"` 或者 `"same"`（区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557)</span>

### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D(data_format='channels_last')
```

对于时序数据的全局最大池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
  `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

__输入尺寸__

尺寸是 `(batch_size, steps, features)` 的 3D 张量。

__输出尺寸__

尺寸是 `(batch_size, features)` 的 2D 张量。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L455)</span>

### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D()
```

对于时序数据的全局平均池化。

__输入尺寸__

- 如果 `data_format='channels_last'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
  输入为 3D 张量，尺寸为：
  `(batch_size, features, steps)`

__输出尺寸__

尺寸是 `(batch_size, features)` 的 2D 张量。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L647)</span>

### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)

```

对于空域数据的全局最大池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L510)</span>

### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)

```

对于空域数据的全局平均池化。

__参数__

- __data_format__: 一个字符串，`channels_last` （默认值）或者 `channels_first`。
  输入张量中的维度顺序。
  `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L742)</span>

### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)

```

对于 3D 数据的全局最大池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L707)</span>

### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)

```

对于 3D 数据的全局平均池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
  表示输入各维度的顺序。
  `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
  而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
  默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
  如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
  尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
  尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量



# local

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L19)</span>

### LocallyConnected1D

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 输入的局部连接层。

`LocallyConnected1D` 层与 `Conv1D` 层的工作方式相同，除了权值不共享外，
也就是说，在输入的每个不同部分应用不同的一组过滤器。

__例子__

```python
# 将长度为 3 的非共享权重 1D 卷积应用于
# 具有 10 个时间步长的序列，并使用 64个 输出滤波器
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# 现在 model.output_shape == (None, 8, 64)
# 在上面再添加一个新的 conv1d
model.add(LocallyConnected1D(32, 3))
# 现在 model.output_shape == (None, 6, 32)
```

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
  指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
  指明卷积的步长。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: 当前仅支持 `"valid"` (大小写敏感)。
  `"same"` 可能会在未来支持。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

3D 张量，尺寸为： `(batch_size, steps, input_dim)`。

__输出尺寸__

3D 张量 ，尺寸为：`(batch_size, new_steps, filters)`，
`steps` 值可能因填充或步长而改变。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L182)</span>

### LocallyConnected2D

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 输入的局部连接层。

`LocallyConnected2D` 层与 `Conv2D` 层的工作方式相同，除了权值不共享外，
也就是说，在输入的每个不同部分应用不同的一组过滤器。

__例子__

```python
# 在 32x32 图像上应用 3x3 非共享权值和64个输出过滤器的卷积
# 数据格式 `data_format="channels_last"`：
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# 现在 model.output_shape == (None, 30, 30, 64)
# 注意这一层的参数数量为 (30*30)*(3*3*3*64) + (30*30)*64 （bias)
'''
正常的卷积生成的每个feature map用的是同一个卷积核和同样的权值;
而局部卷积每次滑动卷积核参数不共享，相当于不滑动
'''

# 在上面再加一个 3x3 非共享权值和 32 个输出滤波器的卷积：
model.add(LocallyConnected2D(32, (3, 3)))
# 现在 model.output_shape == (None, 28, 28, 32)
```

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
  指明 2D 卷积窗口的宽度和高度。
  可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
  指明卷积沿宽度和高度方向的步长。
  可以是一个整数，为所有空间维度指定相同的值。
- __padding__: 当前仅支持 `"valid"` (大小写敏感)。
  `"same"` 可能会在未来支持。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一。
  输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
  `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 "channels_last"。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果你不指定，则不使用激活函数
  (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
  (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
  (详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。

__输入尺寸__

4D 张量，尺寸为：
`(samples, channels, rows, cols)`，如果 data_format='channels_first'；
或者 4D 张量，尺寸为：
`(samples, rows, cols, channels)`，如果 data_format='channels_last'。

__输出尺寸__

4D 张量，尺寸为：
`(samples, filters, new_rows, new_cols)`，如果 data_format='channels_first'；
或者 4D 张量，尺寸为：
`(samples, new_rows, new_cols, filters)`，如果 data_format='channels_last'。
`rows` 和 `cols` 的值可能因填充而改变。



![](http://xiaosheng.me/img/article/article_65_3.png)

**(上)局部连接**：每一接受域有两个像素的局部连接层，每条边都有自身的权重参数。**(中)卷积**：核宽度为两个像素的卷积层，模型与局部连接层具有完全相同的连接，但是参数是共享的，卷积层在整个输入上重复使用相同的两个权重。**(下)全连接**：全连接层类似于局部连接层，它的每条边都有其自身的参数，但是它不具有局部连接层的连接受限的特征。





# recurrent

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L237)</span>

### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

循环神经网络层基类。

__参数__

- __cell__: 一个 RNN 单元实例。RNN 单元是一个具有以下几项的类：

  - 一个 `call(input_at_t, states_at_t)` 方法，
    它返回 `(output_at_t, states_at_t_plus_1)`。
    单元的调用方法也可以采引入可选参数 `constants`，
    详见下面的小节「关于给 RNN 传递外部常量的说明」。
  - 一个 `state_size` 属性。这可以是单个整数（单个状态），
    在这种情况下，它是循环层状态的大小（应该与单元输出的大小相同）。
    这也可以是整数表示的列表/元组（每个状态一个大小）。
  - 一个 `output_size` 属性。 这可以是单个整数或者是一个 TensorShape，
    它表示输出的尺寸。出于向后兼容的原因，如果此属性对于当前单元不可用，
    则该值将由 `state_size` 的第一个元素推断。

  `cell` 也可能是 RNN 单元实例的列表，在这种情况下，RNN 的单元将堆叠在另一个单元上，实现高效的堆叠 RNN。

- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。

- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。

- __go_backwards__: 布尔值 (默认 False)。
  如果为 True，则向后处理输入序列并返回相反的序列。

- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态将用作下一批次中索引 i 样品的初始状态。

- __unroll__: 布尔值 (默认 False)。
  如果为 True，则网络将展开，否则将使用符号循环。
  展开可以加速 RNN，但它往往会占用更多的内存。
  展开只适用于短序列。

- __input_dim__: 输入的维度（整数）。
  将此层用作模型中的第一层时，此参数（或者，关键字参数 `input_shape`）是必需的。

- __input_length__: 输入序列的长度，在恒定时指定。
  如果你要在上游连接 `Flatten` 和 `Dense` 层，
  则需要此参数（如果没有它，无法计算全连接输出的尺寸）。
  请注意，如果循环神经网络层不是模型中的第一层，
  则需要在第一层的层级指定输入长度（例如，通过 `input_shape` 参数）。

__输入尺寸__

3D 张量，尺寸为 `(batch_size, timesteps, input_dim)`。

__输出尺寸__

- 如果 `return_state`：返回张量列表。
  第一个张量为输出。剩余的张量为最后的状态，
  每个张量的尺寸为 `(batch_size, units)`。
- 如果 `return_sequences`：返回 3D 张量，
  尺寸为 `(batch_size, timesteps, units)`。
- 否则，返回尺寸为 `(batch_size, units)` 的 2D 张量。

__Masking__

该层支持以可变数量的时间步对输入数据进行 masking。
要将 masking 引入你的数据，请使用 [Embedding](embeddings.md) 层，
并将 `mask_zero` 参数设置为 `True`。

__关于在 RNN 中使用「状态（statefulness）」的说明__

你可以将 RNN 层设置为 `stateful`（有状态的），
这意味着针对一个批次的样本计算的状态将被重新用作下一批样本的初始状态。
这假定在不同连续批次的样品之间有一对一的映射。

为了使状态有效：

- 在层构造器中指定 `stateful=True`。
- 为你的模型指定一个固定的批次大小，
  如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)` 参数。
- 为你的模型指定一个固定的批次大小，
  如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)`。
  如果是带有 1 个或多个 Input 层的函数式模型，为你的模型的所有第一层传递一个 `batch_shape=(...)`。
  这是你的输入的预期尺寸，*包括批量维度*。
  它应该是整数的元组，例如 `(32, 10, 100)`。
- 在调用 `fit()` 是指定 `shuffle=False`。

要重置模型的状态，请在特定图层或整个模型上调用 `.reset_states()`。

__关于指定 RNN 初始状态的说明__

您可以通过使用关键字参数 `initial_state` 调用它们来符号化地指定 RNN 层的初始状态。
`initial_state` 的值应该是表示 RNN 层初始状态的张量或张量列表。

您可以通过调用带有关键字参数 `states` 的 `reset_states` 方法来数字化地指定 RNN 层的初始状态。
`states` 的值应该是一个代表 RNN 层初始状态的 Numpy 数组或者 Numpy 数组列表。

__关于给 RNN 传递外部常量的说明__

你可以使用 `RNN.__call__`（以及 `RNN.call`）的 `constants` 关键字参数将「外部」常量传递给单元。
这要求 `cell.call` 方法接受相同的关键字参数 `constants`。
这些常数可用于调节附加静态输入（不随时间变化）上的单元转换，也可用于注意力机制。

__例子__

```python
# 首先，让我们定义一个 RNN 单元，作为网络层子类。

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 让我们在 RNN 层使用这个单元：

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 以下是如何使用单元格构建堆叠的 RNN的方法：

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L944)</span>

### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

全连接的 RNN，其输出将被反馈到输入。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  默认：双曲正切（`tanh`）。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
  如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品
  的最后状态将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
  如果为 True，则网络将展开，否则将使用符号循环。
  展开可以加速 RNN，但它往往会占用更多的内存。
  展开只适用于短序列。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1482)</span>

### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

门限循环单元网络（Gated Recurrent Unit） - Cho et al. 2014.

有两种变体。默认的是基于 1406.1078v3 的实现，同时在矩阵乘法之前将复位门应用于隐藏状态。
另一种则是基于 1406.1078v1 的实现，它包括顺序倒置的操作。

第二种变体与 CuDNNGRU(GPU-only) 兼容并且允许在 CPU 上进行推理。
因此它对于 `kernel` 和 `recurrent_kernel` 有可分离偏置。
使用 `'reset_after'=True` 和 `recurrent_activation='sigmoid'` 。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  默认：双曲正切 (`tanh`)。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
  (详见 [activations](../activations.md))。
  默认：分段线性近似 sigmoid (`hard_sigmoid`)。
  如果传入 None，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
  模式 1 将把它的操作结构化为更多的小的点积和加法操作，
  而模式 2 将把它们分批到更少，更大的操作中。
  这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
  如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态
  将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
  如果为 True，则网络将展开，否则将使用符号循环。
  展开可以加速 RNN，但它往往会占用更多的内存。
  展开只适用于短序列。
- __reset_after__: 
- GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。
  False =「之前」(默认)，Ture =「之后」( CuDNN 兼容)。

__参考文献__

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L2034)</span>

### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

长短期记忆网络层（Long Short-Term Memory） - Hochreiter 1997.

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
  (详见 [activations](../activations.md))。
  默认：分段线性近似 sigmoid (`hard_sigmoid`)。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
  如果为 True，初始化时，将忘记门的偏置加 1。
  将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
  这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
  模式 1 将把它的操作结构化为更多的小的点积和加法操作，
  而模式 2 将把它们分批到更少，更大的操作中。
  这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
  如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态
  将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
  如果为 True，则网络将展开，否则将使用符号循环。
  展开可以加速 RNN，但它往往会占用更多的内存。
  展开只适用于短序列。

__参考文献__

- [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L788)</span>

### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

卷积 LSTM。

它类似于 LSTM 层，但输入变换和循环变换都是卷积的。

__参数__

- __filters__: 整数，输出空间的维度
  （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 n 个整数表示的元组或列表，
  指明卷积窗口的维度。
- __strides__: 一个整数，或者 n 个整数表示的元组或列表，
  指明卷积的步长。
  指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` 之一 (大小写敏感)。
- __data_format__: 字符串，
  `channels_last` (默认) 或 `channels_first` 之一。
  输入中维度的顺序。
  `channels_last` 对应输入尺寸为 `(batch, time, ..., channels)`，
  `channels_first` 对应输入尺寸为 `(batch, time, channels, ...)`。
  它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
  找到的 `image_data_format` 值。
  如果你从未设置它，将使用 `"channels_last"`。
- __dilation_rate__: 一个整数，或 n 个整数的元组/列表，指定用于膨胀卷积的膨胀率。
  目前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  如果传入 None，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
  (详见 [activations](../activations.md))。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
  如果为 True，初始化时，将忘记门的偏置加 1。
  将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
  这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __go_backwards__: 布尔值 (默认 False)。
  如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态
  将用作下一批次中索引 i 样品的初始状态。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。

__输入尺寸__

- 如果 data_format='channels_first'，
  输入 5D 张量，尺寸为：
  `(samples,time, channels, rows, cols)`。
- 如果 data_format='channels_last'，
  输入 5D 张量，尺寸为：
  `(samples,time, rows, cols, channels)`。

__输出尺寸__

- 如果 `return_sequences`，
  - 如果 data_format='channels_first'，返回 5D 张量，尺寸为：`(samples, time, filters, output_row, output_col)`。
  - 如果 data_format='channels_last'，返回 5D 张量，尺寸为：`(samples, time, output_row, output_col, filters)`。
- 否则，
  - 如果 data_format ='channels_first'，返回 4D 张量，尺寸为：`(samples, filters, output_row, output_col)`。
  - 如果 data_format='channels_last'，返回 4D 张量，尺寸为：`(samples, output_row, output_col, filters)`。

o_row 和 o_col 取决于 filter 和 padding 的尺寸。

__异常__

- __ValueError__: 无效的构造参数。

__参考文献__

- [Convolutional LSTM Network: A Machine Learning Approach for
  Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)。
  当前的实现不包括单元输出的反馈回路。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L779)</span>

### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SimpleRNN 的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  默认：双曲正切 (`tanh`)。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1163)</span>

### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False)
```

GRU 层的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  默认：双曲正切 (`tanh`)。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
  (详见 [activations](../activations.md))。
  默认：分段线性近似 sigmoid (`hard_sigmoid`)。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
  模式 1 将把它的操作结构化为更多的小的点积和加法操作，
  而模式 2 将把它们分批到更少，更大的操作中。
  这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __reset_after__: 
- GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。
  False = "before" (默认)，Ture = "after" ( CuDNN 兼容)。
- __reset_after__: GRU convention (whether to apply reset gate after or
  before matrix multiplication). False = "before" (default),
  True = "after" (CuDNN compatible).

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1756)</span>

### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)

```

LSTM 层的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
  (详见 [activations](../activations.md))。
  默认：双曲正切（`tanh`）。
  如果传入 `None`，则不使用激活函数
  (即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
  (详见 [activations](../activations.md))。
  默认：分段线性近似 sigmoid (`hard_sigmoid`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
  如果为 True，初始化时，将忘记门的偏置加 1。
  将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
  这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
  单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
  模式 1 将把它的操作结构化为更多的小的点积和加法操作，
  而模式 2 将把它们分批到更少，更大的操作中。
  这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L135)</span>

### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

```

由 [CuDNN](https://developer.nvidia.com/cudnn) 支持的快速 GRU 实现。

只能以 TensorFlow 后端运行在 GPU 上。

__参数__

- __units__: 正整数，输出空间的维度。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: Regularizer function applied to
  the output of the layer (its "activation").
  (see [regularizer](../regularizers.md)).
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态
  将用作下一批次中索引 i 样品的初始状态。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L328)</span>

### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)

```

由 [CuDNN](https://developer.nvidia.com/cudnn) 支持的快速 LSTM 实现。

只能以 TensorFlow 后端运行在 GPU 上。

__参数__

- __units__: 正整数，输出空间的维度。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
  用于输入的线性转换
  (详见 [initializers](../initializers.md))。
- __unit_forget_bias__: 布尔值。
  如果为 True，初始化时，将忘记门的偏置加 1。
  将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
  这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
  的初始化器，用于循环层状态的线性转换
  (详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
  (详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
  (详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
  (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
  (详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __stateful__: 布尔值 (默认 False)。
  如果为 True，则批次中索引 i 处的每个样品的最后状态
  将用作下一批次中索引 i 样品的初始状态。



# embeddings

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/embeddings.py#L16)</span>

### Embedding

```python
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

- 将正整数（索引值）转换为固定尺寸的稠密向量。
- 该层只能用作模型中的第一层。

__例子__

```python
from keras import Sequential
from keras.layers import Embedding
import numpy as np

model = Sequential()
# 输入: 大小为 (batch_size, input_length) 的整数矩阵
# 且输入中的最大整数在0-999之间(词汇表的大小input_dim=1000)
model.add(Embedding(1000, 64, input_length=10))
# 输出：大小为(batch_size, 10, 64)

# 构造输入数据
# input_dim=1000, batch_size=32, input_length=10 
input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

具体例子：

```python
 # 输入：
 [[4], [20]] 
 
 # 输出
 [[0.25, 0.1], [0.6, -0.2]]
```



__参数__

- __input_dim__: int > 0。词汇表大小，
  即，最大整数 index + 1。
- __output_dim__: int >= 0。词向量的维度。
- __embeddings_initializer__: `embeddings` 矩阵的初始化方法
  (详见 [initializers](../initializers.md))。
- __embeddings_regularizer__: `embeddings` matrix 的正则化方法
  (详见 [regularizer](../regularizers.md))。
- __embeddings_constraint__: `embeddings` matrix 的约束函数
  (详见 [constraints](../constraints.md))。
- __mask_zero__: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。
  这对于可变长的 [循环神经网络层](recurrent.md) 十分有用。
  如果设定为 `True`，那么接下来的所有层都必须支持 masking，否则就会抛出异常。
  如果 mask_zero 为 `True`，作为结果，索引 0 就不能被用于词汇表中
  （input_dim 应该与 vocabulary + 1 大小相同）。
- __input_length__: 输入序列的长度，当它是固定的时。
  如果你需要连接 `Flatten` 和 `Dense` 层，则这个参数是必须的
  （没有它，dense 层的输出尺寸就无法计算）。

__输入尺寸__

尺寸为 `(batch_size, sequence_length)` 的 2D 张量。

__输出尺寸__

尺寸为 `(batch_size, sequence_length, output_dim)` 的 3D 张量。

__参考文献__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)



# merge

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L200)</span>

### Add

```python
keras.layers.Add()
```

计算输入张量列表的和。

它接受一个张量的列表，所有的张量必须有相同的输入尺寸，然后返回一个张量（和输入张量尺寸相同）。

__例子__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# 相当于 added = keras.layers.add([x1, x2])
added = keras.layers.Add()([x1, x2])  

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L231)</span>

### Subtract

```python
keras.layers.Subtract()
```

计算两个输入张量的差。

它接受一个长度为 2 的张量列表，两个张量必须有相同的尺寸，然后返回一个值为 (inputs[0] - inputs[1]) 的张量，输出张量和输入张量尺寸相同。

__例子__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# 相当于 subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L268)</span>

### Multiply

```python
keras.layers.Multiply()
```

计算输入张量列表的（逐元素间的）乘积。

它接受一个张量的列表，所有的张量必须有相同的输入尺寸，然后返回一个张量（和输入张量尺寸相同）。

------

<span style="float:right;">[[source]]<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L283)</span>

### Average

```python
keras.layers.Average()
```

计算输入张量列表的平均值。

它接受一个张量的列表，所有的张量必须有相同的输入尺寸，然后返回一个张量（和输入张量尺寸相同）。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L298)</span>

### Maximum

```python
keras.layers.Maximum()
```

计算输入张量列表的（逐元素间的）最大值。

它接受一个张量的列表，所有的张量必须有相同的输入尺寸，然后返回一个张量（和输入张量尺寸相同）。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L320)</span>

### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

连接一个输入张量的列表。

它接受一个张量的列表，除了连接轴之外，其他的尺寸都必须相同，然后返回一个由所有输入张量连接起来的输出张量。

__参数__

- __axis__: 连接的轴。
- __**kwargs__: 层关键字参数。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L416)</span>

### Dot

```python
keras.layers.Dot(axes, normalize=False)

```

计算两个张量之间样本的点积。

例如，如果作用于输入尺寸为 `(batch_size, n)` 的两个张量 `a` 和 `b`，那么输出结果就会是尺寸为 `(batch_size, 1)` 的一个张量。在这个张量中，每一个条目 `i` 是 `a[i]` 和 `b[i]` 之间的点积。

__参数__

- __axes__: 整数或者整数元组，
  一个或者几个进行点积的轴。
- __normalize__: 是否在点积之前对即将进行点积的轴进行 L2 标准化。
  如果设置成 `True`，那么输出两个样本之间的余弦相似值。
- __**kwargs__: 层关键字参数。

------

### add

```python
keras.layers.add(inputs)

```

`Add` 层的函数式接口。

__参数__

- __inputs__: 一个输入张量的列表（列表大小至少为 2）。
- __**kwargs__: 层关键字参数。

__返回__

一个张量，所有输入张量的和。

__例子__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

```

------

### subtract

```python
keras.layers.subtract(inputs)

```

`Subtract` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小准确为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，两个输入张量的差。

__例子__

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)

```

------

### multiply

```python
keras.layers.multiply(inputs)

```

`Multiply` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量的逐元素乘积。

------

### average

```python
keras.layers.average(inputs)

```

`Average` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量的平均值。

------

### maximum

```python
keras.layers.maximum(inputs)

```

`Maximum` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有张量的逐元素的最大值。

------

### concatenate

```python
keras.layers.concatenate(inputs, axis=-1)

```

`Concatenate` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __axis__: 串联的轴。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量通过 `axis` 轴串联起来的输出张量。

------

### dot

```python
keras.layers.dot(inputs, axes, normalize=False)

```

`Dot` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __axes__: 整数或者整数元组，
  一个或者几个进行点积的轴。
- __normalize__: 是否在点积之前对即将进行点积的轴进行 L2 标准化。
  如果设置成 True，那么输出两个样本之间的余弦相似值。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量样本之间的点积。



# advanced activations

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>

### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

带泄漏的 ReLU。

当神经元未激活时，它仍允许赋予一个很小的梯度：
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__输入尺寸__

可以是任意的。如果将该层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha__: float >= 0。负斜率系数。

__参考文献__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L59)</span>

### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

参数化的 ReLU。

形式：
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`,
其中 `alpha` 是一个可学习的数组，尺寸与 x 相同。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha_initializer__: 权重的初始化函数。
- __alpha_regularizer__: 权重的正则化方法。
- __alpha_constraint__: 权重的约束。
- __shared_axes__: 激活函数共享可学习参数的轴。
  例如，如果输入特征图来自输出形状为 `(batch, height, width, channels)`的 2D 卷积层，而且你希望跨空间共享参数，以便每个滤波器只有一组参数，可设置 `shared_axes=[1, 2]`。

__参考文献__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L153)</span>

### ELU

```python
keras.layers.ELU(alpha=1.0)
```

指数线性单元。

形式：
`f(x) =  alpha * (exp(x) - 1.) for x < 0`,
`f(x) = x for x >= 0`.

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha__: 负因子的尺度。

__参考文献__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L193)</span>

### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

带阈值的修正线性单元。

形式：
`f(x) = x for x > theta`,
`f(x) = 0 otherwise`.

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __theta__: float >= 0。激活的阈值位。

__参考文献__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>

### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax 激活函数。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __axis__: 整数，应用 softmax 标准化的轴。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>

### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

ReLU 激活函数。

使用默认值时，它返回逐个元素的 `max(x，0)`。

否则：

- 如果 `x >= max_value`，返回 `f(x) = max_value`，
- 如果 `threshold <= x < max_value`，返回 `f(x) = x`,
- 否则，返回 `f(x) = negative_slope * (x - threshold)`。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __max_value__: 浮点数，最大的输出值。
- __negative_slope__: float >= 0. 负斜率系数。
- __threshold__: float。"thresholded activation" 的阈值。



# normalization

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16)</span>

### BatchNormalization

```python
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

批量标准化层 (Ioffe and Szegedy, 2014)。

在每一个批次的数据中标准化前一层的激活项，即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。

__参数__

- __axis__: 整数，需要标准化的轴（通常是特征轴-channel轴）。
  - 例如，在 `data_format="channels_first"` 的 `Conv2D` 层之后，在 `BatchNormalization` 中设置 `axis=1`。
- __momentum__: 移动均值和移动方差的动量。
- __epsilon__: 增加到方差的小的浮点数，以避免除以零。
- __center__: 如果为 True，把 `beta` 的偏移量加到标准化的张量上。
  如果为 False， `beta` 被忽略。
- __scale__: 如果为 True，乘以 `gamma`。
  如果为 False，`gamma` 不使用。
  当下一层为线性层（或者例如 `nn.relu`），
  这可以被禁用，因为缩放将由下一层完成。
- __beta_initializer__: beta 权重的初始化方法。
- __gamma_initializer__: gamma 权重的初始化方法。
- __moving_mean_initializer__: 移动均值的初始化方法。
- __moving_variance_initializer__: 移动方差的初始化方法。
- __beta_regularizer__: 可选的 beta 权重的正则化方法。
- __gamma_regularizer__: 可选的 gamma 权重的正则化方法。
- __beta_constraint__: 可选的 beta 权重的约束方法。
- __gamma_constraint__: 可选的 gamma 权重的约束方法。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层， 则需要指定 `input_shape` 参数 （整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)



# nosie

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L14)</span>

### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

应用以 0 为中心的加性高斯噪声。

这对缓解过拟合很有用（你可以将其视为随机数据增强的一种形式）。高斯噪声（GS）是对真实输入的腐蚀过程的自然选择。由于它是一个正则化层，因此它只在训练时才被激活。

__参数__

- __stddev__: float，噪声分布的标准差。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L58)</span>

### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

应用以 1 为中心的 乘性高斯噪声。

由于它是一个正则化层，因此它只在训练时才被激活。

__参数__

- __rate__: float，丢弃概率（与 `Dropout` 相同）。
  这个乘性噪声的标准差为 `sqrt(rate / (1 - rate))`。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L106)</span>

### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

将 Alpha Dropout 应用到输入。

Alpha Dropout 是一种 `Dropout`，
它保持输入的平均值和方差与原来的值不变，
以确保即使在 dropout 后也能实现自我归一化。
通过随机将激活设置为负饱和值，
Alpha Dropout 非常适合按比例缩放的指数线性单元（SELU）。

__参数__

- __rate__: float，丢弃概率（与 `Dropout` 相同）。
  这个乘性噪声的标准差为 `sqrt(rate / (1 - rate))`。
- __seed__: 用作随机种子的 Python 整数。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)



# wrappers

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L114)</span>

### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

这个封装器将一个层应用于输入的每个时间片。

输入至少为 3D，且第一个维度应该是时间所表示的维度。

考虑 32 个样本的一个 batch，
其中每个样本是 10 个 16 维向量的序列。
那么这个 batch 的输入尺寸为 `(32, 10, 16)`，
而 `input_shape` 不包含样本数量的维度，为 `(10, 16)`。

你可以使用 `TimeDistributed` 来将 `Dense` 层独立地应用到
这 10 个时间步的每一个：

```python
# 作为模型第一层
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# 现在 model.output_shape == (None, 10, 8)
```

输出的尺寸为 `(32, 10, 8)`。

在后续的层中，将不再需要 `input_shape`：

```python
model.add(TimeDistributed(Dense(32)))
# 现在 model.output_shape == (None, 10, 32)
```

输出的尺寸为 `(32, 10, 32)`。

`TimeDistributed` 可以应用于任意层，不仅仅是 `Dense`，
例如运用于 `Conv2D` 层：

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__参数__

- __layer__: 一个网络层实例。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L333)</span>

### Bidirectional

```python
keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

RNN 的双向封装器，对序列进行前向和后向计算。

__参数__

- __layer__: `Recurrent` 实例。
- __merge_mode__: 前向和后向 RNN 的输出的结合模式。
  为 {'sum', 'mul', 'concat', 'ave', None} 其中之一。
  如果是 None，输出不会被结合，而是作为一个列表被返回。

__异常__

- __ValueError__: 如果参数 `merge_mode` 非法。

__例__

```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```





# writing your own keras layers

# 编写你自己的 Keras 层

对于简单、无状态的自定义操作，你也许可以通过 `layers.core.Lambda` 层来实现。但是对于那些包含了可训练权重的自定义层，你应该自己实现这种层。

这是一个 **Keras2.0** 中，Keras 层的骨架（如果你用的是旧的版本，请更新到新版）。你只需要实现三个方法即可:

- `build(input_shape)`: 这是你定义权重的地方。这个方法必须设 `self.built = True`，可以通过调用 `super([Layer], self).build()` 完成。
- `call(x)`: 这里是编写层的功能逻辑的地方。你只需要关注传入 `call` 的第一个参数：输入张量，除非你希望你的层支持masking。
- `compute_output_shape(input_shape)`: 如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

还可以定义具有多个输入张量和多个输出张量的 Keras 层。
为此，你应该假设方法 `build(input_shape)`，`call(x)` 
和 `compute_output_shape(input_shape)` 的输入输出都是列表。
这里是一个例子，与上面那个相似：

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
```

已有的 Keras 层就是实现任何层的很好例子。不要犹豫阅读源码！



