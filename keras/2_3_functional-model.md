# Keras 函数式 API

Keras 函数式 API 是定义复杂模型（如多输出模型、有向无环图，或具有共享层的模型）的方法。

## 基本用法

### 全连接网络

`Sequential` 模型可能是实现这种网络的一个更好选择，但这个例子能够帮助我们进行一些简单的理解。

- 网络层的实例是可调用的，它以张量为参数，并且返回一个张量
- 输入和输出均为张量，它们都可以用来定义一个模型（`Model`）
- 这样的模型同 Keras 的 `Sequential` 模型一样，都可以被训练

```python
from keras.layers import Input, Dense
from keras.models import Model

# 这部分返回一个张量
inputs = Input(shape=(784,))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # 开始训练
```

-----

### 模型调用

利用函数式 API，可以轻易地重用训练好的模型：

- 可以将任何模型看作是一个层，然后通过传递一个张量来调用它。
- 注意，在调用模型时，您不仅重用模型的*结构*，还重用了它的权重。

```python
x = Input(shape=(784,))
# 这是可行的，并且返回上面定义的 10-way softmax。
y = model(x) # 重用之前定义好的model,可看做一层
```

这种方式能允许我们快速创建可以处理*序列输入*的模型。只需一行代码，你就将图像分类模型转换为视频分类模型。

```python
from keras.layers import TimeDistributed

# 输入张量是 20 个时间步的序列，
# 每一个时间为一个 784 维的向量
input_sequences = Input(shape=(20, 784))

# 这部分将我们之前定义的模型应用于 输入序列中的 每个时间步。
# 之前定义的模型的输出是一个 10-way softmax，
# 因而下面的层的输出将是维度为 10 的 20 个向量的序列。
# 即 产生时序输出 
processed_sequences = TimeDistributed(model)(input_sequences)
```

-----

### 多输入多输出模型

以下是函数式 API 的一个很好的例子：具有多个输入和输出的模型。函数式 API 使处理大量交织的数据流变得容易。

来考虑下面的模型。我们试图预测 Twitter 上的一条新闻标题有多少转发和点赞数。模型的主要输入将是新闻标题本身，即一系列词语，但是为了增添趣味，我们的模型还添加了其他的辅助输入来接收额外的数据，例如新闻标题的发布的时间等。
该模型也将通过两个损失函数进行监督学习。较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法。

模型结构如下图所示：

<img src="imgs/multi-input-multi-output-graph.png" width=400/>

让我们用函数式 API 来实现它。

主要输入接收新闻标题本身:

- 即**一个整数序列（每个整数编码一个词）**, 这些整数在 1 到 10,000 之间（10,000 个词的词汇表），
- 序列长度为 100 个词。

```python
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

# 标题输入：接收一个含有 100 个整数的序列，每个整数在 1 到 10000 之间。
# 注意我们可以通过传递一个 "name" 参数来命名任何层。
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Embedding 层将输入序列编码为一个稠密向量的序列，
# input_dim 是词汇表的大小，输入中的整数值范围为[0,1000)
# input_length: 输入序列长度
# output_dim: 要编码成的向量的维度
# input: (batch_size, input_length)
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
# output: (batch_size, input_length, output_dim)

# LSTM 层把向量序列转换成单个向量，
# 它包含整个序列的上下文信息
lstm_out = LSTM(32)(x)

# 在这里，我们插入辅助损失，使得即使在模型主损失很高的情况下，LSTM 层和 Embedding 层都能被平稳地训练。
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# 此时，我们将辅助输入数据与 LSTM 层的输出拼接起来，输入到模型中：
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 堆叠多个全连接网络层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# 最后添加主要的逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

#最后定义一个具有两个输入和两个输出的模型：
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# 编译模型，并给辅助损失分配一个 0.2 的权重
# 如果要为不同的输出指定不同的 `loss_weights` 或 `loss`，可以使用列表或字典。
# 在这里，我们给 `loss` 参数传递单个损失函数，这个损失将用于所有的输出。
# 模型将最小化的误差值是由 `loss_weights` 系数加权的*加权总和*误差。
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

# 我们可以通过传递输入数组和目标数组的列表来训练模型：
model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
```



```python
# 由于输入和输出均被命名了（在定义时传递了一个 `name` 参数），我们也可以通过以下方式编译模型：
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# 然后使用以下方式训练：
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
```



-----

### 共享网络层

函数式 API 的另一个用途是使用共享网络层的模型。我们来看看共享层。

来考虑推特推文数据集。我们想要建立一个模型来分辨两条推文是否来自同一个人（例如，通过推文的相似性来对用户进行比较）。

实现这个目标的一种方法是建立一个模型，将两条推文编码成两个向量，连接向量，然后添加逻辑回归层；这将输出两条推文来自同一作者的概率。模型将接收一对对正负表示的推特数据。

由于这个问题是对称的，编码第一条推文的机制应该被完全重用来编码第二条推文（权重及其他全部）。这里我们使用一个共享的 LSTM 层来编码推文。

让我们使用函数式 API 来构建它。首先我们将一条推特转换为一个尺寸为 `(280, 256)` 的矩阵，即每条推特 280 字符，每个字符为 256 维的 one-hot 编码向量 （取 256 个常用字符）。

```python
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(280, 256))
tweet_b = Input(shape=(280, 256))
```

要在不同的输入上共享同一个层，只需实例化该层一次，然后根据需要传入你想要的输入即可：

```python
# 这一层可以输入一个矩阵，并返回一个 64 维的向量
shared_lstm = LSTM(64)

# 当我们重用相同的图层实例多次，图层的权重也会被重用 (它其实就是同一层)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 然后再连接两个向量：
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# 再在上面添加一个逻辑回归层
predictions = Dense(1, activation='sigmoid')(merged_vector)

# 定义一个连接推特输入和预测的可训练的模型
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)
```

那么如何读取共享层的输出或输出尺寸呢？因为其对应多个不同的输入，也会产生不同的输出与输出尺寸。

-----

### 层「节点」的概念

每当你在某个输入上调用一个层时，都将创建一个新的张量（层的输出），并且为该层添加一个「节点」，将输入张量连接到输出张量。当多次调用同一个图层时，该图层将拥有多个节点索引 (0, 1, 2...)。

在之前版本的 Keras 中，可以通过 `layer.get_output()` 来获得层实例的输出张量，或者通过 `layer.output_shape` 来获取其输出形状。现在你依然可以这么做（除了 `get_output()` 已经被 `output` 属性替代）。但是如果一个层与多个输入连接呢？

只要一个层仅仅连接到一个输入，就不会有困惑，`.output` 会返回层的唯一输出：

```python
a = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a
```

但是如果该层有多个输入，那就会出现问题：

```python
a = Input(shape=(280, 256))
b = Input(shape=(280, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

lstm.output
```

```python
>> AttributeError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.
```

通过下面的方法可以解决：

```python
assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b
```

`input_shape` 和 `output_shape` 这两个属性也是如此：只要该层只有一个节点，或者只要所有节点具有相同的输入/输出尺寸，那么「层输出/输入尺寸」的概念就被很好地定义，并且将由 `layer.output_shape` / `layer.input_shape` 返回。

但是比如说，如果将一个 `Conv2D` 层先应用于尺寸为 `(32，32，3)` 的输入，再应用于尺寸为 `(64, 64, 3)` 的输入，那么这个层就会有多个输入/输出尺寸，你将不得不通过指定它们所属节点的索引来获取它们：

```python
a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# 到目前为止只有一个输入，以下可行：
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# 现在 `.input_shape` 属性不可行，但是这样可以：
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
```

-----



## Model 类方法

在函数式 API 中，给定一些输入张量和输出张量，可以通过以下方式实例化一个 `Model`：

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
```

这个模型将包含从 `a` 到 `b` 的计算的所有网络层。 

在多输入或多输出模型的情况下，你也可以使用列表：

```python
model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])
```

有关 `Model` 的详细介绍，请阅读 [Keras 函数式 API 指引](/getting-started/functional-api-guide)。

### compile

```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

用于配置训练模型。

__参数__

- __optimizer__: 字符串（优化器名）或者优化器实例。
  详见 [optimizers](/optimizers)。
- __loss__: 字符串（目标函数名）或目标函数。
  详见 [losses](/losses)。
  如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。
  模型将最小化的损失值将是所有单个损失的总和。
- __metrics__: 在训练和测试期间的模型评估标准。
  通常你会使用 `metrics = ['accuracy']`。
  要为多输出模型的不同输出指定不同的评估标准，
  还可以传递一个字典，如 `metrics = {'output_a'：'accuracy'}`。
- __loss_weights__: 可选的指定标量系数（Python 浮点数）的列表或字典，
  用以衡量损失函数对不同的模型输出的贡献。
  模型将最小化的误差值是由 `loss_weights` 系数加权的*加权总和*误差。
  如果是列表，那么它应该是与模型输出相对应的 1:1 映射。
  如果是张量，那么应该把输出的名称（字符串）映到标量系数。
- __sample_weight_mode__: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 `temporal`。
  默认为 `None`，为采样权重（1D）。
  如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 `sample_weight_mode`。
- __weighted_metrics__: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
- __target_tensors__: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。
  相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据），
  您可以通过 `target_tensors` 参数指定它们。
  它可以是单个张量（单输出模型），张量列表，或一个映射输出名称到目标张量的字典。
- __**kwargs__: 当使用 Theano/CNTK 后端时，这些参数被传入 `K.function`。
  当使用 TensorFlow 后端时，这些参数被传递到 `tf.Session.run`。

__异常__

- __ValueError__:  如果 `optimizer`, `loss`, `metrics` 或 `sample_weight_mode` 这些参数不合法。

------

### fit

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

以给定数量的轮次（数据集上的迭代）训练模型。

__参数__

- __x__: 训练数据的 Numpy 数组（如果模型只有一个输入），
  或者是 Numpy 数组的列表（如果模型有多个输入）。
  如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，`x` 可以是 `None`（默认）。
- __y__: 目标（标签）数据的 Numpy 数组（如果模型只有一个输出），
  或者是 Numpy 数组的列表（如果模型有多个输出）。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`。每次梯度更新的样本数。如果未指定，默认为 32。
- __epochs__: 整数。训练模型迭代轮次。一个轮次是在整个 `x` 和 `y` 上的一轮迭代。
  请注意，与 `initial_epoch` 一起，`epochs` 被理解为 「最终轮次」。模型并不是训练了 `epochs` 轮，而是到第 `epochs` 轮停止训练。
- __verbose__: 0, 1 或 2。日志显示模式。
  0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
- __callbacks__: 一系列的 `keras.callbacks.Callback` 实例。一系列可以在训练时使用的回调函数。
  详见 [callbacks](/callbacks)。
- __validation_split__: 0 和 1 之间的浮点数。用作验证集的训练数据的比例。
  模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。
  验证数据是混洗之前 `x` 和`y` 数据的最后一部分样本中。
- __validation_data__: 元组 `(x_val，y_val)` 或元组 `(x_val，y_val，val_sample_weights)`，
  用来评估损失，以及在每轮结束时的任何模型度量指标。
  模型将不会在这个数据上进行训练。这个参数会覆盖 `validation_split`。
- __shuffle__: 布尔值（是否在每轮迭代之前混洗数据）或者 字符串 (`batch`)。
  `batch` 是处理 HDF5 数据限制的特殊选项，它对一个 batch 内部的数据进行混洗。
  当 `steps_per_epoch` 非 `None` 时，这个参数无效。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。
  这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1:1 映射），
  或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。
  在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __initial_epoch__: 整数。开始训练的轮次（有助于恢复之前的训练）。
- __steps_per_epoch__: 整数或 `None`。
  在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。
  使用 TensorFlow 数据张量等输入张量进行训练时，默认值 `None` 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
- __validation_steps__: 只有在指定了 `steps_per_epoch` 时才有用。停止前要验证的总步数（批次样本）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。 

__异常__

- __RuntimeError__: 如果模型从未编译。
- __ValueError__: 在提供的输入数据与模型期望的不匹配的情况下。

------

### evaluate

```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```

在测试模式下返回模型的误差值和评估标准值。

计算是分批进行的。

__参数__

- __x__: 测试数据的 Numpy 数组（如果模型只有一个输入），
  或者是 Numpy 数组的列表（如果模型有多个输入）。
  如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 `None`（默认）。
- __y__: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`。每次评估的样本数。如果未指定，默认为 32。
- __verbose__: 0 或 1。日志显示模式。
  0 = 安静模式，1 = 进度条。
- __sample_weight__: 测试样本的可选 Numpy 权重数组，用于对损失函数进行加权。
  您可以传递与输入样本长度相同的扁平（1D）Numpy 数组（权重和样本之间的 1:1 映射），
  或者在时序数据的情况下，传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。
  在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __steps__: 整数或 `None`。
  声明评估结束之前的总步数（批次样本）。默认值 `None`。

__返回__

标量测试误差（如果模型只有一个输出且没有评估标准）
或标量列表（如果模型具有多个输出 和/或 评估指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

------

### predict

```python
predict(x, batch_size=None, verbose=0, steps=None)
```

为输入样本生成输出预测。

计算是分批进行的

__参数__

- __x__: 输入数据，Numpy 数组
  （或者 Numpy 数组的列表，如果模型有多个输出）。
- __batch_size__: 整数。如未指定，默认为 32。
- __verbose__: 日志显示模式，0 或 1。
- __steps__: 声明预测结束之前的总步数（批次样本）。默认值 `None`。

__返回__

预测的 Numpy 数组（或数组列表）。

__异常__

- __ValueError__: 在提供的输入数据与模型期望的不匹配的情况下，
  或者在有状态的模型接收到的样本不是 batch size 的倍数的情况下。

------

### train_on_batch

```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

运行一批样品的单次梯度更新。

__参数_

- __x__: 训练数据的 Numpy 数组（如果模型只有一个输入），
  或者是 Numpy 数组的列表（如果模型有多个输入）。
  如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
- __y__: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
- __sample_weight__: 可选数组，与 x 长度相同，包含应用到模型损失函数的每个样本的权重。
  如果是时域数据，你可以传递一个尺寸为 (samples, sequence_length) 的 2D 数组，
  为每一个样本的每一个时间步应用不同的权重。
  在这种情况下，你应该在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，以在训练时对模型的损失函数加权。
  这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。

__返回__

标量训练误差（如果模型只有一个输入且没有评估标准），
或者标量的列表（如果模型有多个输出 和/或 评估标准）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

------

### test_on_batch

```python
test_on_batch(x, y, sample_weight=None)
```

在一批样本上测试模型。

__参数__

- __x__: 测试数据的 Numpy 数组（如果模型只有一个输入），
  或者是 Numpy 数组的列表（如果模型有多个输入）。
  如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
- __y__: 目标（标签）数据的 Numpy 数组，或 Numpy 数组的列表（如果模型具有多个输出）。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
- __sample_weight__: 可选数组，与 x 长度相同，包含应用到模型损失函数的每个样本的权重。
  如果是时域数据，你可以传递一个尺寸为 (samples, sequence_length) 的 2D 数组，
  为每一个样本的每一个时间步应用不同的权重。

__返回__

标量测试误差（如果模型只有一个输入且没有评估标准），
或者标量的列表（如果模型有多个输出 和/或 评估标准）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

------

### predict_on_batch

```python
predict_on_batch(x)

```

返回一批样本的模型预测值。

__参数__

- __x__: 输入数据，Numpy 数组。

__返回__

预测值的 Numpy 数组（或数组列表）。

------

### fit_generator

```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

```

使用 Python 生成器（或 `Sequence` 实例）逐批生成的数据，按批次训练模型。

生成器与模型并行运行，以提高效率。
例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。

`keras.utils.Sequence` 的使用可以保证数据的顺序，
以及当 `use_multiprocessing=True` 时 ，保证每个输入在每个 epoch 只使用一次。

__参数__

- __generator__: 一个生成器，或者一个 `Sequence` (`keras.utils.Sequence`) 对象的实例，
  以在使用多进程时避免数据的重复。
  生成器的输出应该为以下之一：

  - 一个 `(inputs, targets)` 元组
  - 一个 `(inputs, targets, sample_weights)` 元组。

  这个元组（生成器的单个输出）组成了单个的 batch。
  因此，这个元组中的所有数组长度必须相同（与这一个 batch 的大小相等）。
  不同的 batch 可能大小不同。
  例如，一个 epoch 的最后一个 batch 往往比其他 batch 要小，
  如果数据集的尺寸不能被 batch size 整除。
  生成器将无限地在数据集上循环。当运行到第 `steps_per_epoch` 时，记一个 epoch 结束。

- __steps_per_epoch__: 在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。
  它通常应该等于你的数据集的样本数量除以批量大小。
  对于 `Sequence`，它是可选的：如果未指定，将使用`len(generator)` 作为步数。

- __epochs__: 整数。训练模型的迭代总轮数。一个 epoch 是对所提供的整个数据的一轮迭代，如 `steps_per_epoch` 所定义。注意，与 `initial_epoch` 一起使用，epoch 应被理解为「最后一轮」。模型没有经历由 `epochs` 给出的多次迭代的训练，而仅仅是直到达到索引 `epoch` 的轮次。

- __verbose__: 0, 1 或 2。日志显示模式。
  0 = 安静模式, 1 = 进度条, 2 = 每轮一行。

- __callbacks__: `keras.callbacks.Callback` 实例的列表。在训练时调用的一系列回调函数。

- __validation_data__: 它可以是以下之一：

  - 验证数据的生成器或 `Sequence` 实例
  - 一个 `(inputs, targets)` 元组
  - 一个 `(inputs, targets, sample_weights)` 元组。

  

在每个 epoch 结束时评估损失和任何模型指标。该模型不会对此数据进行训练。
    

- __validation_steps__: 仅当 `validation_data` 是一个生成器时才可用。
  在停止前 `generator` 生成的总步数（样本批数）。
  对于 `Sequence`，它是可选的：如果未指定，将使用 `len(generator)` 作为步数。
- __class_weight__: 可选的将类索引（整数）映射到权重（浮点）值的字典，用于加权损失函数（仅在训练期间）。
  这可以用来告诉模型「更多地关注」来自代表性不足的类的样本。
- __max_queue_size__: 整数。生成器队列的最大尺寸。
  如未指定，`max_queue_size` 将默认为 10。
- __workers__: 整数。使用的最大进程数量，如果使用基于进程的多线程。
  如未指定，`workers` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 布尔值。如果 True，则使用基于进程的多线程。
  如未指定， `use_multiprocessing` 将默认为 False。
  请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __shuffle__: 是否在每轮迭代之前打乱 batch 的顺序。
  只能与 `Sequence` (keras.utils.Sequence) 实例同用。
- __initial_epoch__: 开始训练的轮次（有助于恢复之前的训练）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。 

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

__例__

```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
            # 从文件中的每一行生成输入数据和标签的 numpy 数组，
            x1, x2, y = process_line(line)
            yield ({'input_1': x1, 'input_2': x2}, {'output': y})
        f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)

```

------

### evaluate_generator

```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

```

在数据生成器上评估模型。

这个生成器应该返回与 `test_on_batch` 所接收的同样的数据。

__参数__

- __generator__: 一个生成 `(inputs, targets)` 或 `(inputs, targets, sample_weights)` 的生成器，
  或一个 `Sequence` (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时数据的重复。
- __steps__: 在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。
  它通常应该等于你的数据集的样本数量除以批量大小。
  对于 `Sequence`，它是可选的：如果未指定，将使用`len(generator)` 作为步数。
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用的最大进程数量，如果使用基于进程的多线程。
  如未指定，`workers` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 布尔值。如果 True，则使用基于进程的多线程。
  请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__: 日志显示模式，0 或 1。

__返回__

标量测试误差（如果模型只有一个输入且没有评估标准），
或者标量的列表（如果模型有多个输出 和/或 评估标准）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

------

### predict_generator

```python
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)

```

为来自数据生成器的输入样本生成预测。

这个生成器应该返回与 `predict_on_batch` 所接收的同样的数据。

__参数__

- __generator__: 生成器，返回批量输入样本，
  或一个 `Sequence` (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时数据的重复。
- __steps__: 在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。
  它通常应该等于你的数据集的样本数量除以批量大小。
  对于 `Sequence`，它是可选的：如果未指定，将使用`len(generator)` 作为步数。
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用的最大进程数量，如果使用基于进程的多线程。
  如未指定，`workers` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。
  请注意，由于此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__: 日志显示模式，0 或 1。

__返回__

预测值的 Numpy 数组（或数组列表）。

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

------

### get_layer

```python
get_layer(self, name=None, index=None)

```

根据名称（唯一）或索引值查找网络层。

如果同时提供了 `name` 和 `index`，则 `index` 将优先。

索引值来自于水平图遍历的顺序（自下而上）。

__参数__

- __name__: 字符串，层的名字。
- __index__: 整数，层的索引。

__返回__

一个层实例。

__异常__

- __ValueError__: 如果层的名称或索引不正确。



## 经典样例

代码示例仍然是起步的最佳方式，所以这里还有更多的例子。

### Inception 模型

有关 Inception 结构的更多信息，请参阅 [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842)。

```python
from keras.layers import Conv2D, MaxPooling2D, Input

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
```

### 卷积层上的残差连接

有关残差网络 (Residual Network) 的更多信息，请参阅 [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)。

```python
from keras.layers import Conv2D, Input

# 输入张量为 3 通道 256x256 图像
x = Input(shape=(256, 256, 3))
# 3 输出通道（与输入通道相同）的 3x3 卷积核
y = Conv2D(3, (3, 3), padding='same')(x)
# 返回 x + y
z = keras.layers.add([x, y])
```

### 共享视觉模型

该模型在两个输入上重复使用同一个图像处理模块，以判断两个 MNIST 数字是否为相同的数字。

```python
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

# 首先，定义视觉模型
digit_input = Input(shape=(27, 27, 1))
x = Conv2D(64, (3, 3))(digit_input)
x = Conv2D(64, (3, 3))(x)
x = MaxPooling2D((2, 2))(x)
out = Flatten()(x)

vision_model = Model(digit_input, out)

# 然后，定义区分数字的模型
digit_a = Input(shape=(27, 27, 1))
digit_b = Input(shape=(27, 27, 1))

# 视觉模型将被共享，包括权重和其他所有
out_a = vision_model(digit_a)
out_b = vision_model(digit_b)

concatenated = keras.layers.concatenate([out_a, out_b])
out = Dense(1, activation='sigmoid')(concatenated)

classification_model = Model([digit_a, digit_b], out)
```

### 视觉问答模型

当被问及关于图片的自然语言问题时，该模型可以选择正确的单词作答。

它通过将问题和图像编码成向量，然后连接两者，在上面训练一个逻辑回归，来从词汇表中挑选一个可能的单词作答。

```python
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential

# 首先，让我们用 Sequential 来定义一个视觉模型。
# 这个模型会把一张图像编码为向量。
vision_model = Sequential()
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# 现在让我们用视觉模型来得到一个输出张量：
image_input = Input(shape=(224, 224, 3))
encoded_image = vision_model(image_input)

# 接下来，定义一个语言模型来将问题编码成一个向量。
# 每个问题最长 100 个词，词的索引从 1 到 9999.
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

# 连接问题向量和图像向量：
merged = keras.layers.concatenate([encoded_question, encoded_image])

# 然后在上面训练一个 1000 词的逻辑回归模型：
output = Dense(1000, activation='softmax')(merged)

# 最终模型：
vqa_model = Model(inputs=[image_input, question_input], outputs=output)

# 下一步就是在真实数据上训练模型。
```

### 视频问答模型

现在我们已经训练了图像问答模型，我们可以很快地将它转换为视频问答模型。在适当的训练下，你可以给它展示一小段视频（例如 100 帧的人体动作），然后问它一个关于这段视频的问题（例如，「这个人在做什么运动？」 -> 「足球」）。

```python
from keras.layers import TimeDistributed

video_input = Input(shape=(100, 224, 224, 3))
# 这是基于之前定义的视觉模型（权重被重用）构建的视频编码
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # 输出为向量的序列
encoded_video = LSTM(256)(encoded_frame_sequence)  # 输出为一个向量

# 这是问题编码器的模型级表示，重复使用与之前相同的权重：
question_encoder = Model(inputs=question_input, outputs=encoded_question)

# 让我们用它来编码这个问题：
video_question_input = Input(shape=(100,), dtype='int32')
encoded_video_question = question_encoder(video_question_input)

# 这就是我们的视频问答模式：
merged = keras.layers.concatenate([encoded_video, encoded_video_question])
output = Dense(1000, activation='softmax')(merged)
video_qa_model = Model(inputs=[video_input, video_question_input], outputs=output)
```

