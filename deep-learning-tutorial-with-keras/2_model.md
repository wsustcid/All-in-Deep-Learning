# Model 类

## Model类的方法和属性

在 Keras 中有两类主要的模型：[Sequential 顺序模型](/models/sequential) 和 [使用函数式 API 的 Model 类模型](/models/model)。

这些模型有许多共同的方法和属性：

- `model.layers` 是包含模型网络层的展平列表。

- `model.inputs` 是模型输入张量的列表。

- `model.outputs` 是模型输出张量的列表。

- `model.summary()` 打印出模型概述信息。 它是 [utils.print_summary](/utils/#print_summary) 的简捷调用。

  ```python
  from keras.applications.vgg16 import VGG16 
  base_model = VGG16(weights='imagenet', include_top=False)
  
  print(len(base_model.layers))
  print()
  print(base_model.inputs) # 模型输入张量的列表，可能由多个输入
  print(base_model.input)
  print()
  print(base_model.outputs) # 模型输入张量的列表，可能由多个输入
  print(base_model.output)
  print()
  print(base_model.input_shape)
  print(base_model.output_shape)
  print()
  print(base_model.summary())
  
  # =================== #
  19
  
  [<tf.Tensor 'input_1:0' shape=(?, ?, ?, 3) dtype=float32>]
  Tensor("input_1:0", shape=(?, ?, ?, 3), dtype=float32)
  
  [<tf.Tensor 'block5_pool/MaxPool:0' shape=(?, ?, ?, 512) dtype=float32>]
  Tensor("block5_pool/MaxPool:0", shape=(?, ?, ?, 512), dtype=float32)
  
  (None, None, None, 3)
  (None, None, None, 512)
  
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #   
  =================================================================
  input_1 (InputLayer)         (None, None, None, 3)     0         
  _________________________________________________________________
  block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
  _________________________________________________________________
  block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
  _________________________________________________________________
  block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
  _________________________________________________________________
  block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
  _________________________________________________________________
  block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
  _________________________________________________________________
  block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
  _________________________________________________________________
  block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
  _________________________________________________________________
  block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
  _________________________________________________________________
  block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
  _________________________________________________________________
  block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
  _________________________________________________________________
  block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
  _________________________________________________________________
  block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
  _________________________________________________________________
  block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
  _________________________________________________________________
  block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
  _________________________________________________________________
  block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
  _________________________________________________________________
  block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
  _________________________________________________________________
  block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
  _________________________________________________________________
  block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
  =================================================================
  Total params: 14,714,688
  Trainable params: 14,714,688
  Non-trainable params: 0
  _________________________________________________________________
  None
  ```

  

- `model.get_config()` 返回包含模型配置信息的字典。通过以下代码，就**可以根据这些配置信息重新实例化模型：**

```python
config = model.get_config()
# 对于Model类模型
model = Model.from_config(config)
# 或者，对于 Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()` 返回模型中所有权重张量的列表，类型为 Numpy 数组。
- `model.set_weights(weights)` 从 Numpy 数组中为模型设置权重。列表中的数组必须与 `get_weights()` 返回的权重具有相同的尺寸。
- `model.to_json()` 以 JSON 字符串的形式返回模型的表示。请注意，**该表示不包括权重，仅包含结构**。你可以通过以下方式从 JSON 字符串**重新实例化同一模型**（使用重新初始化的权重）：

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml()` 以 YAML 字符串的形式返回模型的表示。请注意，该表示不包括权重，只包含结构。你可以通过以下代码，从 YAML 字符串中重新实例化相同的模型（使用重新初始化的权重）：

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)` 将模型权重存储为 HDF5 文件。
- `model.load_weights(filepath, by_name=False)`: 从 HDF5 文件（由 `save_weights` 创建）中加载权重。默认情况下，模型的结构应该是不变的。 如果想将权重载入不同的模型（部分层相同）， 设置 `by_name=True` 来载入那些名字相同的层的权重。

注意：另请参阅[如何安装 HDF5 或 h5py 以保存 Keras 模型](/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)，在常见问题中了解如何安装 `h5py` 的说明。

## Model 类继承

除了这两类模型之外，你还可以通过继承 `Model` 类并在 `call` 方法中实现你自己的前向传播，以创建你自己的完全定制化的模型，（`Model` 类继承 API 引入于 Keras 2.2.0）。

这里是一个用 `Model` 类继承写的简单的多层感知器的例子：

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

网络层定义在 `__init__(self, ...)` 中，前向传播在 `call(self, inputs)` 中指定。在 `call` 中，你可以指定自定义的损失函数，通过调用 `self.add_loss(loss_tensor)` （就像你在自定义层中一样）。

在类继承模型中，模型的拓扑结构是由 Python 代码定义的（而不是网络层的静态图）。这意味着该模型的拓扑结构不能被检查或序列化。因此，以下方法和属性**不适用于类继承模型**：

- `model.inputs` 和 `model.outputs`。
- `model.to_yaml()` 和 `model.to_json()`。
- `model.get_config()` 和 `model.save()`。

**关键点**：为每个任务使用正确的 API。`Model` 类继承 API 可以为实现复杂模型提供更大的灵活性，但它需要付出代价（比如缺失的特性）：它更冗长，更复杂，并且有更多的用户错误机会。如果可能的话，尽可能使用函数式 API，这对用户更友好。







# Sequential 模型

## 基本使用

顺序模型是多个网络层的**线性堆叠**。

你可以通过将**网络层实例的列表**传递给 `Sequential` 的构造器，来创建一个 `Sequential` 模型：

```python
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([Dense(32, input_shape=(784,)),
                    Activation('relu'),
                    Dense(10),
                    Activation('softmax')])
```

也可以简单地使用 `.add()` 方法将各层添加到模型中：

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```



1. **指定输入数据的尺寸**

模型需要知道它所期望的输入的尺寸。出于这个原因，顺序模型中的第一层（且只有第一层，因为下面的层可以自动地推断尺寸）需要接收关于其输入尺寸的信息。有几种方法来做到这一点：

- 传递一个 `input_shape` 参数给第一层。它是一个表示**尺寸的元组** (一个整数或 `None` 的元组，其中 `None` 表示可能为任何正整数)。在 `input_shape` 中**不包含数据的 batch 大小**。
- 某些 2D 层，例如 `Dense`，支持通过参数 `input_dim` 指定输入尺寸，某些 **3D 时序层**支持 `input_dim` 和 `input_length` 参数。
- 如果你需要为你的输入指定一个固定的 batch 大小（这对 stateful RNNs 很有用），你可以传递一个 `batch_size` 参数给一个层。如果你同时将 `batch_size=32` 和 `input_shape=(6, 8)` 传递给一个层，那么每一批输入的尺寸就为 `(32，6，8)`。

因此，下面的代码片段是等价的：

```python
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```

```python
model = Sequential()
model.add(Dense(32, input_dim=784))
```



2. **模型编译**

在训练模型之前，您需要配置学习过程，这是通过 `compile` 方法完成的。它接收三个参数：

- 优化器 optimizer。它可以是现有优化器的字符串标识符，如 `rmsprop` 或 `adagrad`，也可以是 Optimizer 类的实例。详见：[optimizers](/optimizers)。
- 损失函数 loss，模型试图最小化的目标函数。它可以是现有损失函数的字符串标识符，如 `categorical_crossentropy` 或 `mse`，也可以是一个目标函数。详见：[losses](/losses)。
- **评估标准 metrics**。对于任何分类问题，你都希望将其设置为 `metrics = ['accuracy']`。评估标准可以是现有的标准的字符串标识符，也可以是**自定义的评估标准函数**。

```python
# 多分类问题
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 二分类问题
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 均方误差回归问题
model.compile(optimizer='rmsprop',
              loss='mse')

# 自定义评估标准函数
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```



3. **模型训练**

Keras 模型在输入数据和标签的 Numpy 矩阵上进行训练。为了训练一个模型，你通常会使用 `fit` 函数。[文档详见此处](/models/sequential)。

```python
# 对于具有 2 个类的单输入模型（二进制分类）：

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100)) # 产生0~1之间的随机数，元组参数为数组维度
labels = np.random.randint(2, size=(1000, 1)) # 产生固定范围的随机整数，第一个参数是必须的[0,2)

# 训练模型，以 32 个样本为一个 batch 进行迭代
model.fit(data, labels, epochs=10, batch_size=32)
```

```python
# 对于具有 10 个类的单输入模型（多分类分类）：

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 生成虚拟数据
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# 将标签转换为分类的 one-hot 编码 (1000, 10)
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# 训练模型，以 32 个样本为一个 batch 进行迭代
model.fit(data, one_hot_labels, epochs=10, batch_size=32)
```

------



## 模型方法

### compile

```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

用于配置训练模型。

__参数__

- __optimizer__: 字符串（优化器名）或者优化器对象。详见 [optimizers](/optimizers)。
- __loss__: 字符串（目标函数名）或目标函数。详见 [losses](/losses)。
  如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。模型将最小化的损失值将是所有单个损失的总和。
- __metrics__: 在训练和测试期间的模型评估标准。通常你会使用 `metrics = ['accuracy']`。
  要为多输出模型的不同输出指定不同的评估标准，还可以传递一个字典，如 `metrics = {'output_a'：'accuracy'}`。
- __loss_weights__: 指定标量系数（Python浮点数）的可选列表或字典，用于加权不同模型输出的损失贡献。
  模型将要最小化的损失值将是所有单个损失的加权和，由 `loss_weights` 系数加权。
  如果是列表，则期望与模型的输出具有 1:1 映射。
  如果是张量，则期望将输出名称（字符串）映射到标量系数。
- __sample_weight_mode__: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 `temporal`。
  默认为 `None`，为采样权重（1D）。如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 `sample_weight_mode`。
- __weighted_metrics__: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
- __target_tensors__: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据），您可以通过 `target_tensors` 参数指定它们。它应该是单个张量（对于单输出 Sequential 模型）。
- __**kwargs__: 当使用 Theano/CNTK 后端时，这些参数被传入 `K.function`。当使用 TensorFlow 后端时，这些参数被传递到 `tf.Session.run`。

__异常__

- __ValueError__:  如果 `optimizer`, `loss`, `metrics` 或 `sample_weight_mode` 这些参数不合法。

------

### fit

```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

以固定数量的轮次（数据集上的迭代）训练模型。

update:

````python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1)
````

- **validation_freq**: Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=2` runs validation every 2 epochs. If a list, tuple, or set, specifies the epochs on which to run validation, e.g. `validation_freq=[1, 2, 10]` runs validation at the end of the 1st, 2nd, and 10th epochs.



__参数__

- __x__: 训练数据的 Numpy 数组
- 如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组
- 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 `None`（默认）
- __y__: 目标（标签）数据的 Numpy 数组
- 如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
- 如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`: 每次梯度更新的样本数
  - 如果未指定，默认为 32.
- __epochs__: 整数。训练模型迭代轮次。一个轮次是在整个 `x` 或 `y` 上的一轮迭代。
  - 请注意，与 `initial_epoch` 一起，`epochs` 被理解为 「最终轮次」。模型并不是训练了 `epochs` 轮，而是到第 `epochs` 轮停止训练。
- __verbose__: 0, 1 或 2。日志显示模式：
  - 0 = 安静模式, 
  - 1 = 进度条, 
  - 2 = 每轮一行。
- __callbacks__: 一系列的 `keras.callbacks.Callback` 实例。一系列可以在训练时使用的回调函数。详见 [callbacks](/callbacks)。
- __validation_split__: 在 0 和 1 之间浮动。用作验证集的训练数据的比例。模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。
  - 验证数据是混洗之前 `x` 和`y` 数据的最后一部分样本中（所以让他划分之前最好自己要混洗一下？）
- __validation_data__: **元组** `(x_val，y_val)` 或元组 `(x_val，y_val，val_sample_weights)`，用来评估损失，以及在每轮结束时的任何模型度量指标。模型将不会在这个数据上进行训练。
  - 这个参数会覆盖 `validation_split`。
- __shuffle__: 布尔值（是否在每轮迭代之前混洗数据）或者 字符串 (`batch`)。
  - `batch` 是处理 HDF5 数据限制的特殊选项，它对一个 batch 内部的数据进行混洗。当 `steps_per_epoch` 非 `None` 时，这个参数无效。
- __class_weight__: 可选的**字典**，用来映射类**索引**（整数）到**权重**（浮点）值，用于加权损失函数（仅在训练期间）。
  - 这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  - 您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），
  - 或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __initial_epoch__: 开始训练的轮次（有助于恢复之前的训练）。==????==
- __steps_per_epoch__: 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。
  - 使用 TensorFlow 数据张量等输入张量进行训练时，默认值 `None` 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
- __validation_steps__: 只有在指定了 `steps_per_epoch`时才有用。停止前要验证的总步数（批次样本）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失值和评估值，以及验证集损失值和评估值的记录（如果适用）。 

```python
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0)

# plot training history
# 如果定义了评价指标metrics,也可以通过 “acc”, "val_acc" 访问历史记录
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```



__异常__

- __RuntimeError__: 如果模型从未编译。
- __ValueError__: 在提供的输入数据与模型期望的不匹配的情况下。

------

### evaluate

```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```

在测试模式，返回**误差值**和评估标准值。

计算逐批次进行。

update:

```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None, callbacks=None)
```

- **callbacks**: List of `keras.callbacks.Callback` instances. List of callbacks to apply during evaluation. See [callbacks](https://keras.io/callbacks).

__参数__

- __x__: 训练数据的 Numpy 数组。
  如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 `None`（默认）。
- __y__: 目标（标签）数据的 Numpy 数组。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`。每次提度更新的样本数。如果未指定，默认为 32.
- __verbose__: 0, 1。日志显示模式。0 = 安静模式, 1 = 进度条。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __steps__: 整数或 `None`。
  声明评估结束之前的总步数（批次样本）。默认值 `None`。

__返回__

- 标量测试误差（如果模型只有单个输出且没有评估指标）
- 或标量列表（如果模型具有多个输出和/或指标）。属性 `model.metrics_names` 将提供标量输出的显示标签。

```python
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```



------

### predict

```python
predict(x, batch_size=None, verbose=0, steps=None)
```

为输入样本生成输出预测。

计算逐批次进行。

update:

```python
predict(x, batch_size=None, verbose=0, steps=None, callbacks=None)
```

- **callbacks**: List of `keras.callbacks.Callback` instances. List of callbacks to apply during prediction. See [callbacks](https://keras.io/callbacks).

__参数__

- __x__: 输入数据，Numpy 数组（或者如果模型有多个输入，则为 Numpy 数组列表）。
- __batch_size__: 整数。如未指定，默认为 32。
- __verbose__: 日志显示模式，0 或 1。
- __steps__: 声明预测结束之前的总步数（批次样本）。默认值 `None`。

__返回__

预测的 Numpy 数组。

__异常__

- __ValueError__: 如果提供的输入数据与模型的期望数据不匹配，或者有状态模型收到的数量不是批量大小的倍数。

------

### train_on_batch

```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

一批样品的单次梯度更新。

__Arguments__

- __x__: 训练数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输入都已命名，你还可以传入输入名称到 Numpy 数组的映射字典。
- __y__: 目标数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输出都已命名，你还可以传入输出名称到 Numpy 数组的映射字典。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。

__返回__

标量训练误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

------

### test_on_batch

```python
test_on_batch(x, y, sample_weight=None)

```

在一批样本上评估模型。

__参数__

- __x__: 训练数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输入都已命名，你还可以传入输入名称到 Numpy 数组的映射字典。
- __y__: 目标数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输出都已命名，你还可以传入输出名称到 Numpy 数组的映射字典。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。

__返回__

标量测试误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

------

### predict_on_batch

```python
predict_on_batch(x)
```

返回一批样本的模型预测值。

__参数__

- __x__: 输入数据，Numpy 数组或列表（如果模型有多输入）。

__返回__

预测值的 Numpy 数组。

------

### fit_generator

```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

使用 Python 生成器或 `Sequence` 实例逐批生成的数据，按批次训练模型。

生成器与模型并行运行，以提高效率。
例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。

`keras.utils.Sequence` 的使用可以保证数据的顺序， 以及当 `use_multiprocessing=True` 时 ，保证每个输入在每个 epoch 只使用一次。

update:

```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, validation_freq=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

- **validation_freq**: Only relevant if validation data is provided. Integer or `collections.Container`instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. `validation_freq=2` runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. `validation_freq=[1, 2, 10]`runs validation at the end of the 1st, 2nd, and 10th epochs.

__参数__

- __generator__: 一个生成器或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
  生成器的输出应该为以下之一：
  - 一个 `(inputs, targets)` 元组
  - 一个 `(inputs, targets, sample_weights)` 元组。
    这个元组（生成器的单个输出）表示一个独立批次。因此，此元组中的所有数组必须具有相同的长度（等于此批次的大小）。不同的批次可能具有不同的大小。例如，如果数据集的大小不能被批量大小整除，则最后一批时期通常小于其他批次。生成器将无限地在数据集上循环。当运行到第 `steps_per_epoch` 时，记一个 epoch 结束。
- __steps_per_epoch__: 整数。在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。它通常应该等于你的数据集的样本数量除以批量大小。可选参数 `Sequence`：如果未指定，将使用 `len(generator)` 作为步数。
- __epochs__: 整数，数据的迭代总轮数。一个 epoch 是对所提供的整个数据的一轮迭代，由 `steps_per_epoch` 所定义。请注意，与 `initial_epoch` 一起，参数 `epochs` 应被理解为 「最终轮数」。模型并不是训练了 `epochs` 轮，而是到第 `epochs` 轮停止训练。
- __verbose__: 日志显示模式。0，1 或 2。0 = 安静模式，1 = 进度条，2 = 每轮一行。
- __callbacks__: `keras.callbacks.Callback` 实例列表。在训练时调用的一系列回调。详见 [callbacks](/callbacks)。
- __validation_data__: 它可以是以下之一：
  - 验证数据的生成器或 `Sequence` 实例
  - 一个 `(inputs, targets)` 元组
  - 一个 `(inputs, targets, sample_weights)` 元组。
- __validation_steps__: 仅当 `validation_data` 是一个生成器时才可用。
  每个 epoch 结束时验证集生成器产生的步数。它通常应该等于你的数据集的样本数量除以批量大小。可选参数 `Sequence`：如果未指定，将使用 `len(generator)` 作为步数。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
- __max_queue_size__: 整数。生成器队列的最大尺寸。如果未指定，`max_queue_size` 将默认为 10。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。如果未指定，`use_multiprocessing` 将默认为 `False`。请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __shuffle__: 布尔值。是否在每轮迭代之前打乱 batch 的顺序。只能与 `Sequence` (`keras.utils.Sequence`) 实例同用。在 `steps_per_epoch` 不为 `None` 是无效果。
- __initial_epoch__: 整数。开始训练的轮次（有助于恢复之前的训练）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。 

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。



#### 数据生成器

##### Example 1:

Here is an example of data generator:

Assume `features` is an array of data with shape (100,64,64,3) and `labels` is an array of data with shape (100,1). We use data from `features` and `labels`to train our model.

```python
def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 64, 64, 3))
 batch_labels = np.zeros((batch_size,1))
 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choice(len(features),1)
     batch_features[i] = some_processing(features[index])
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels

```

With the generator above, if we define `batch_size = 10` , that means it will randomly taking out 10 samples from `features` and `labels` to feed into each epoch until an epoch hits 50 sample limit. Then fit_generator() destroys the used data and move on repeating the same process in new epoch.

One great advantage about **fit_generator()** besides saving memory is user **can integrate random augmentation inside the generator**, so it will always provide model with new data to train on the fly.

##### Example 2:

```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # 从文件中的每一行生成输入数据和标签的 numpy 数组
                x1, x2, y = process_line(line)
                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)

```



##### 解析

在python中，当你定义一个函数，使用了yield关键字时，这个函数就是一个__生成器__ (也就是说，只要有yield这个词出现，你在用def定义函数的时候，系统默认这就不是一个函数啦，而是一个生成器）。如果需要生成器返回（下一个）值，需要调用.next()函数。其实当系统判断def是生成器时，就会自动支持.next()函数，例如：

```python
def fibonacci(max_n):
    a, b = 1, 1
    while a <= max_n:
        
        yield a
        a, b = b, a+b 
        
if __name__ == '__main__':
    
    n = []
    for i in fibonacci(15):
        n.append(i)
        
    print n 
        
    m = fibonacci(13)
    print m
    
    print m.next()
    print m.next()
    print m.next()
    
## output:
[1, 1, 2, 3, 5, 8, 13]
<generator object fibonacci at 0x7f751b544820>
1
1
2

```



1. 每个生成器只能使用一次。比如上个例子中的m生成器，一旦打印完m的7个值，就没有办法再打印m的值了，因为已经吐完了。生成器每次运行之后都会在运行到yield的位置时候，保存暂时的状态，跳出生成器函数，在下次执行生成器函数的时候会从上次截断的位置继续开始执行循环。
2. yield一般都在def生成器定义中搭配一些循环语句使用，比如for或者while，以防止运行到生成器末尾跳出生成器函数，就不能再yield了。有时候，为了保证生成器函数永远也不会执行到函数末尾，会用while True: 语句，这样就会保证只要使用next()，这个生成器就会生成一个值，是处理无穷序列的常见方法。

拿第一个例子为例， 每次继续开始执行上次没处理完成的位置，但后面的每次循环都只在while True这个循环体内部运行，之前的非循环体batch_feature...  batch_label ...并没有执行，因为它们只在第一次进入生成其函数的时候才有效地运行过一次。



### evaluate_generator

```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```

在数据生成器上评估模型。

这个生成器应该返回与 `test_on_batch` 所接收的同样的数据。

```python
evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, 
workers=1, use_multiprocessing=False, verbose=0)
```

- **callbacks**: List of `keras.callbacks.Callback` instances. List of callbacks to apply during training. See [callbacks](https://keras.io/callbacks).

__参数__

- __generator__: 生成器，生成 (inputs, targets)
  或 (inputs, targets, sample_weights)，或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
- __steps__: Total number of steps (batches of samples) to yield from `generator` before stopping. （暂时理解为迭代完整个数据集所需的步数：nb_samples/batch）
- Optional for `Sequence`: if unspecified, will use the `len(generator)` as a number of steps（只有generator是自带的keras.utils.Sequence类才可以不指定，否则必须指定）
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。
  请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__：日志显示模式，0 或 1。

__返回__

标量测试误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
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

update:

```python
predict_generator(generator, steps=None, callbacks=None, max_queue_size=10, 
workers=1, use_multiprocessing=False, verbose=0)
```

- **callbacks**: List of `keras.callbacks.Callback` instances. List of callbacks to apply during training. See [callbacks](https://keras.io/callbacks).

__参数__

- __generator__: 返回批量输入样本的生成器，或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
- __steps__: 在停止之前，来自 `generator` 的总步数 (样本批次)。
  可选参数 `Sequence`：如果未指定，将使用`len(generator)` 作为步数。
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。
  请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__: 日志显示模式， 0 或 1。

__返回__

预测值的 Numpy 数组。

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

------

### get_layer

```python
get_layer(name=None, index=None)
```

根据名称（唯一）或索引值查找网络层。

如果同时提供了 `name` 和 `index`，则 `index` 将优先。

根据网络层的名称（唯一）或其索引返回该层。索引是基于水平图遍历的顺序（自下而上）。

__参数__

- __name__: 字符串，层的名字。
- __index__: 整数，层的索引。

__返回__

一个层实例。

__异常__

- __ValueError__: 如果层的名称或索引不正确。





## 典型样例

这里有几个可以帮助你起步的例子！

在 [examples 目录](https://github.com/keras-team/keras/tree/master/examples) 中，你可以找到真实数据集的示例模型：

- CIFAR10 小图片分类：具有实时数据增强的卷积神经网络 (CNN)
- IMDB 电影评论情感分类：基于词序列的 LSTM
- Reuters 新闻主题分类：多层感知器 (MLP)
- MNIST 手写数字分类：MLP & CNN
- 基于 LSTM 的字符级文本生成

...以及更多。

### 

```python
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# 生成虚拟数据
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
# 在第一层必须指定所期望的输入数据尺寸：
# 在这里，是一个 20 维的向量。
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)
```

### 基于多层感知器的二分类：

```python
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

import numpy as np

# 生成虚拟数据
x_train = np.random.random((1000, 20))
y_train = to_categorical(np.random.randint(10, size=(1000,1)), 
                         num_classes=10)

x_test = np.random.random((100,20))
y_test = to_categorical(np.random.randint(10, size=(100,1)), 
                         num_classes=10)

# 创建模型
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 编译模型
model.compile(loss='categorical_crossentropy',
             optimizer=sgd,
             metrics=['accuracy'])

# 拟合模型
model.fit(x_train, y_train,
         epochs=20,
         batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)

print(score)
```

### 类似 VGG 的卷积神经网络：

```python
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# 生成虚拟数据
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

model = Sequential()
# 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
# 使用 32 个大小为 3x3 的卷积滤波器。
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=32)

```

### 基于 LSTM 的序列分类：

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

max_features = 1024

model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 基于 1D 卷积的序列分类：

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

seq_length = 64

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, activation='relu'))
model.add(Conv1D(128, 3, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=16, epochs=10)
score = model.evaluate(x_test, y_test, batch_size=16)
```

### 基于栈式 LSTM 的序列分类

在这个模型中，我们将 3 个 LSTM 层叠在一起，使模型能够学习更高层次的时间表示。

前两个 LSTM 返回完整的输出序列，但最后一个只返回输出序列的最后一步，从而降低了时间维度（即将输入序列转换成单个向量）。

<img src="/home/ubuntu16/Deep-learning-tutorial/keras/imgs/regular_stacked_lstm.png" alt="stacked LSTM" style="width: 300px;"/>

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # 返回维度为 32 的向量序列
model.add(LSTM(32, return_sequences=True))  # 返回维度为 32 的向量序列
model.add(LSTM(32))  # 返回维度为 32 的单个向量
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 生成虚拟训练数据
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# 生成虚拟验证数据
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))
```

### "stateful" 渲染的的栈式 LSTM 模型

有状态 (stateful) 的循环神经网络模型中，在一个 batch 的样本处理完成后，其内部状态（记忆）会被记录并作为下一个 batch 的样本的初始状态。这允许处理更长的序列，同时保持计算复杂度的可控性。

[你可以在 FAQ 中查找更多关于 stateful RNNs 的信息。](/getting-started/faq/#how-can-i-use-stateful-rnns)

```python
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# 期望输入数据尺寸: (batch_size, timesteps, data_dim)
# 请注意，我们必须提供完整的 batch_input_shape，因为网络是有状态的。
# 第 k 批数据的第 i 个样本是第 k-1 批数据的第 i 个样本的后续。
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 生成虚拟训练数据
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# 生成虚拟验证数据
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
```



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

------

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

------

### 多输入多输出模型

以下是函数式 API 的一个很好的例子：具有多个输入和输出的模型。函数式 API 使处理大量交织的数据流变得容易。

来考虑下面的模型。我们试图预测 Twitter 上的一条新闻标题有多少转发和点赞数。模型的主要输入将是新闻标题本身，即一系列词语，但是为了增添趣味，我们的模型还添加了其他的辅助输入来接收额外的数据，例如新闻标题的发布的时间等。
该模型也将通过两个损失函数进行监督学习。较早地在模型中使用主损失函数，是深度学习模型的一个良好正则方法。

模型结构如下图所示：

<img src="/home/ubuntu16/Deep-learning-tutorial/keras/imgs/multi-input-multi-output-graph.png" width=400/>

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



------

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

------

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

------



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
  - If steps_per_epoch is set, the `batch_size` must be None.
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



# 模型保存

TensorFlow的模型格式有很多种，针对不同场景可以使用不同的格式，只要符合规范的模型都可以轻易部署到在线服务或移动设备上，这里简单列举一下。

- Checkpoint： 用于保存模型的权重，主要用于模型训练过程中参数的备份和模型训练热启动。
- GraphDef：用于保存模型的Graph，不包含模型权重，加上checkpoint后就有模型上线的全部信息。
- ExportModel：使用exportor接口导出的模型文件，包含模型Graph和权重可直接用于上线，但官方已经标记为deprecated推荐使用SavedModel。
- **SavedModel**：使用saved_model接口导出的模型文件，包含模型Graph和权限可直接用于上线，TensorFlow和Keras模型推荐使用这种模型格式。
- FrozenGraph：使用freeze_graph.py对checkpoint和GraphDef进行整合和优化，可以直接部署到Android、iOS等移动设备上。
- TFLite：基于flatbuf对模型进行优化，可以直接部署到Android、iOS等移动设备上，使用接口和FrozenGraph有些差异。

**模型格式**

目前建议TensorFlow和Keras模型都导出成SavedModel格式，这样就可以直接使用通用的TensorFlow Serving服务，模型导出即可上线不需要改任何代码。

不同的模型导出时只要指定输入和输出的signature即可，**其中字符串的key可以任意命名只会在客户端请求时用到**，可以参考下面的代码示例。

注意，目前使用tf.py_func()的模型导出后不能直接上线，模型的所有结构建议都用op实现。

**TensorFlow模型导出**

```python
import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat

model_path = "model"
model_version = 1
model_signature = signature_def_utils.build_signature_def(
    inputs={
        "keys": utils.build_tensor_info(keys_placeholder),
        "features": utils.build_tensor_info(inference_features)
    },
    outputs={
        "keys": utils.build_tensor_info(keys_identity),
        "prediction": utils.build_tensor_info(inference_op),
        "softmax": utils.build_tensor_info(inference_softmax),
    },
    method_name=signature_constants.PREDICT_METHOD_NAME)

export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
 
builder = saved_model_builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
    sess, [tag_constants.SERVING],
    clear_devices=True,
    signature_def_map={
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
        model_signature,
    },
    legacy_init_op=legacy_init_op)
 
builder.save() 
```

**Keras模型导出**

```python
import os
import tensorflow as tf
from tensorflow.python.util import compat
 
def export_savedmodel(model):
  model_path = "model"
  model_version = 1
  model_signature = tf.saved_model.signature_def_utils.predict_signature_def(
      inputs={'input': model.input}, outputs={'output': model.output})
  export_path = os.path.join(compat.as_bytes(model_path), compat.as_bytes(str(model_version)))
 
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)
  builder.add_meta_graph_and_variables(
      sess=K.get_session(),
      tags=[tf.saved_model.tag_constants.SERVING],
      clear_devices=True,
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          model_signature
      })
  builder.save()
```

**SavedModel模型结构**

使用TensorFlow的API导出SavedModel模型后，可以检查模型的目录结构如下，然后就可以直接使用开源工具来加载服务了。

![img](https://pic2.zhimg.com/80/v2-f6acd33963b0691f725b942d5567f159_hd.jpg)

------

## 模型上线

**部署在线服务**

使用HTTP接口可参考 [tobegit3hub/simple_tensorflow_serving](https://link.zhihu.com/?target=https%3A//github.com/tobegit3hub/simple_tensorflow_serving) 。

使用gRPC接口可参考 [tensorflow/serving](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/serving) 。

**部署离线设备**

部署到Android可参考 [https://medium.com/@tobe_ml/all-tensorflow-models-can-be-embedded-into-mobile-devices-1932e80579e5](https://link.zhihu.com/?target=https%3A//medium.com/%40tobe_ml/all-tensorflow-models-can-be-embedded-into-mobile-devices-1932e80579e5) 。

部署到iOS可参考 <https://zhuanlan.zhihu.com/p/33715219> 。



### keras 模型部署 tensorflow serving

这两天算法同事那边用keras训练了个二分类的模型。有一个新的需求是把keras模型跑到 tensorflow serving上 （TensorFlow Serving 系统用于在生产环境中运行模型）。

在这之前我并没有接触过keras、tensorflow ， 官方教程和一堆的博客论坛资料有些过时，（keras 模型转 tensorflow 模型的示例代码跑不动），过程不太顺利，于是花了一天学习 keras、 tensorlow, 写个小demo，再追踪一下keras和tensorflow源代码，耗时两天终于把这个需求实现了。这里记录填坑过程。

#### keras模型转 tensorflow模型

我把 keras模型转tensorflow serving模型所使用的方法如下：

**1、要拿到算法训练好的keras模型文件（一个HDF5文件）**

该文件应该包含：

- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始

**2、编写 keras模型转tensorflow serving模型的代码**

```python
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from os.path import isfile

def build_model():
    model = Sequential()
    # 省略这部分代码，根据算法实际情况填写
    return model
# 保存keras模型为pb模型
def save_model_to_serving(model, export_version, export_path='prod_models'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
        inputs={'voice': model.input}, outputs={'scores': model.output})
    
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            'voice_classification': signature,                                                                                                                     
        },
        legacy_init_op=legacy_init_op)
    builder.save()

if __name__ == '__main__':
    # 1.构建模型
    model = build_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='xxx', # 用实际算法情况替换这里的xxx
                  metrics=['xxx'])
    model.summary()
    
    # 2. 导入模型权重
    checkpoint_filepath = 'weights.hdf5'
    if (isfile(checkpoint_filepath)):
        print('Checkpoint file detected. Loading weights.')
        model.load_weights(checkpoint_filepath) # 加载模型
    else:
        print('No checkpoint file detected.  Starting from scratch.')
    # 3. 保存为pb模型
    export_path = "test_model"
    save_model_to_serving(model, "1", export_path)
```

上面的例子将模型保存到 test_model目录下
test_model目录结构如下：

```python
test_model/
└── 1
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

saved_model.pb 是能在 tensorflow serving跑起来的模型。

**3、跑模型**

```python
tensorflow_model_server --port=8500 --model_name="voice" --model_base_path="/home/yu/workspace/test/test_model/"
```

标准输出如下（算法模型已成功跑起来了）：

```python
2018-02-08 16:28:02.641662: I tensorflow_serving/model_servers/main.cc:149] Building single TensorFlow model file config:  model_name: voice model_base_path: /home/yu/workspace/test/test_model/
2018-02-08 16:28:02.641917: I tensorflow_serving/model_servers/server_core.cc:439] Adding/updating models.
2018-02-08 16:28:02.641976: I tensorflow_serving/model_servers/server_core.cc:490]  (Re-)adding model: voice
2018-02-08 16:28:02.742740: I tensorflow_serving/core/basic_manager.cc:705] Successfully reserved resources to load servable {name: voice version: 1}
2018-02-08 16:28:02.742800: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: voice version: 1}
2018-02-08 16:28:02.742815: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: voice version: 1}
2018-02-08 16:28:02.742867: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:360] Attempting to load native SavedModelBundle in bundle-shim from: /home/yu/workspace/test/test_model/1
2018-02-08 16:28:02.742906: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:236] Loading SavedModel from: /home/yu/workspace/test/test_model/1
2018-02-08 16:28:02.755299: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-02-08 16:28:02.795329: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:155] Restoring SavedModel bundle.
2018-02-08 16:28:02.820146: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:190] Running LegacyInitOp on SavedModel bundle.
2018-02-08 16:28:02.832832: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:284] Loading SavedModel: success. Took 89481 microseconds.
2018-02-08 16:28:02.834804: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: voice version: 1}
2018-02-08 16:28:02.836855: I tensorflow_serving/model_servers/main.cc:290] Running ModelServer at 0.0.0.0:8500 ...
```

**4、客户端代码**

```python
from __future__ import print_function
from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import numpy as np

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
tf.app.flags.DEFINE_string('vocie', '', 'path to voice in wav format')
FLAGS = tf.app.flags.FLAGS

def get_melgram(path):
    melgram = .... # 这里省略
    return melgram

def main(_):
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    # Send request

    # See prediction_service.proto for gRPC request/response details.
    data = get_melgram("T_1000001.wav")
    data = data.astype(np.float32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'voice' # 这个name跟tensorflow_model_server  --model_name="voice" 对应
    request.model_spec.signature_name = 'voice_classification' # 这个signature_name  跟signature_def_map 对应
    request.inputs['voice'].CopyFrom(
          tf.contrib.util.make_tensor_proto(data, shape=[1, 1, 96, 89])) # shape跟 keras的model.input类型对应
    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
  tf.app.run()
```

客户端跑出的结果是：

```python
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 2
      }
    }
    float_val: 0.0341101661325
    float_val: 0.965889811516
  }
}
```

`float_val: 0.0341101661325`和`float_val: 0.965889811516`就是我们需要的结果。

#### keras模型转 tensorflow模型的一些说明

1、 keras 保存模型

可以使用`model.save(filepath)`将Keras模型和权重保存在一个HDF5文件中，该文件将包含：

- 模型的结构，以便重构该模型
- 模型的权重
- 训练配置（损失函数，优化器等）
- 优化器的状态，以便于从上次训练中断的地方开始

当然这个 HDF5 也可以是用下面的代码生成

```python
from keras.callbacks import ModelCheckpoint

checkpoint_filepath = 'weights.hdf5'
checkpointer = ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)
```

2、 keras 加载模型

keras 加载模型像下面这样子(中间部分代码省略了)：

```python
from keras.models import Sequential, Model

model = Sequential()
.....
model.compile(loss='categorical_crossentropy',
       optimizer='xxx', # 用实际算法情况替换这里的xxx
       metrics=['xxx'])
model.summary()
model.load_weights("xxx.h5") # 加载keras模型(一个HDF5文件)
```

Remark:

1：过时的生成方法

有些方法已经过时了（例如下面这种）:

```python
from tensorflow_serving.session_bundle import exportefrom tensorflow_serving.session_bundle import e expxporotetr

export_path = ... # where to save the exported graph
export_version = ... # version number (integer)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)
```

如果使用这种过时的方法，用tensorflow serving 跑模型的时候会提示：

```python
WARNING:tensorflow:From test.py:107: Exporter.export (from tensorflow.contrib.session_bundle.exporter) is deprecated and will be removed after 2017-06-30.
Instructions for updating:
No longer supported. Switch to SavedModel immediately.
```

从warning中 显然可以知道这种方法要被抛弃了，不再支持这种方法了， 建议我们转用 SaveModel方法。

填坑大法： 使用 SaveModel

```python
def save_model_to_serving(model, export_version, export_path='prod_models'):
    print(model.input, model.output)
    signature = tf.saved_model.signature_def_utils.predict_signature_def(                                                                        
        inputs={'voice': model.input}, outputs={'scores': model.output})
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),                                                                                                                    
        tags=[tf.saved_model.tag_constants.SERVING],                                                                                             
        signature_def_map={                                                                                                                      
            'classification': signature,                                                                                                                     
        },
        legacy_init_op=legacy_init_op)
    builder.save()
```



## 保存tensorflow模型为pb文件

通常训练模型的时候是保存ckpt方便接着训练，但是上线可以保存为pb模型，加载的时候不需要重新定义模型，只用输入输出来调用模型。

```python
'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2019-08-25 10:18:46
@LastEditTime: 2019-08-25 10:50:04
'''
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

tf.app.flags.DEFINE_string('mode', 'train', 'train or test')

FLAGS = tf.app.flags.FLAGS

class model():
    def __init__(self):
        self.a = tf.placeholder(tf.float32, [None])
        self.w = tf.Variable(tf.constant(2.0, shape=[1]), name="w")
        b = tf.Variable(tf.constant(0.5, shape=[1]), name="b")
        self.y = self.a * self.w + b

#模型保存为ckpt
def save_model():
    graph1 = tf.Graph()
    with graph1.as_default():
        m = model()
    with tf.Session(graph=graph1) as session:
        session.run(tf.global_variables_initializer())
        update = tf.assign(m.w, [10])
        session.run(update)
        predict_y = session.run(m.y,feed_dict={m.a:[3.0]})
        print(predict_y)

        saver = tf.train.Saver()
        saver.save(session,"model_pb/model.ckpt")


#保存为pb模型
def export_model(session, m):


   #只需要修改这一段，定义输入输出，其他保持默认即可
   # 定义模型输入输出名
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)
    
    # 定义模型保存路径
    export_path = "pb_model/1"
    if os.path.exists(export_path):
        os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))
    
    # 模型保存
    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))

##加载pb模型
def load_pb():
    # 导入pb模型
    session = tf.Session(graph=tf.Graph())
    model_file_path = "pb_model/1"
    meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_file_path)
    
    # 导出模型输出名（保存pb模型时定义的）
    model_graph_signature = list(meta_graph.signature_def.items())[0][1]
    output_tensor_names = [] # tensor_name 是模型定义是给tensor起的别名（未指定时会tensorflow会自动定义）
    # 测试pb模型时只需要用到tensor名
    # 但op_name 与tensor_name 是相关的，也可以通过op_name 得到tensor_name
    #output_op_names = [] # op_name 是在模型保存pb模型时model_signature中名称
    for output_item in model_graph_signature.outputs.items():
        #output_op_name = output_item[0] 
        #output_op_names.append(output_op_name)
        output_tensor_name = output_item[1].name
        output_tensor_names.append(output_tensor_name)
    #print("output_tensor_names: {}; output_op_names: {}".format(output_tensor_names, output_op_names))
    print("load model finish!")
    
    
    sentences = {} # 测试数据
    ## 测试pb模型
    # 模型输入名要与保存pb模型时定义的保持一致
    for test_x in [[1],[2],[3],[4],[5]]:
        sentences["input"] = test_x # key值与保存pb模型时定义保持一致
        feed_dict_map = {}
        for input_item in model_graph_signature.inputs.items():
            #input_op_name = input_item[0]
            input_tensor_name = input_item[1].name
            #print("input_tensor_name: {}; input_op_name: {}".format(input_tensor_name, input_op_name))
            #feed_dict_map[input_tensor_name] = sentences['input_op_name']
            feed_dict_map[input_tensor_name] = test_x
            
        predict_y = session.run(output_tensor_names, feed_dict=feed_dict_map)
        
        print("predict pb y:",predict_y)

def main(argv=None):
    
    if FLAGS.mode == 'train':
        # 1. 训练时保存模型权重
        save_model()
    
        # 2. 定义模型并恢复模型权重
        graph2 = tf.Graph()
        with graph2.as_default():
            m = model()
            saver = tf.train.Saver()
        with tf.Session(graph=graph2) as session:
            saver.restore(session, "model_pb/model.ckpt") #加载ckpt模型
            # 3. 保存为pb模型
            export_model(session, m)
    
    elif FLAGS.mode == 'test':
        # 3. 导入pb模型并测试
        load_pb()
    else:
        print("ERROR")
    
if __name__ == "__main__":
    tf.app.run()
    

```

save_model 和load_pb两个函数要分开执行，第一次注释掉load_pb，只save，第二次load的时候注释掉save。因为声明模型的时候都是用默认图，变量命名会依次是0,1，load的时候名字对应不上。

保存好的pb模型路径文件格式为



![img](https://upload-images.jianshu.io/upload_images/13198972-fe62cd7917238b87.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/293/format/webp)

测试输出：

```python
output_tensor_names: ['add:0']; output_op_names: ['output']
load model finish!
input_tensor_name: Placeholder:0; input_op_name: input
predict pb y: [array([10.5], dtype=float32)]
input_tensor_name: Placeholder:0; input_op_name: input
predict pb y: [array([20.5], dtype=float32)]
input_tensor_name: Placeholder:0; input_op_name: input
predict pb y: [array([30.5], dtype=float32)]
input_tensor_name: Placeholder:0; input_op_name: input
predict pb y: [array([40.5], dtype=float32)]
input_tensor_name: Placeholder:0; input_op_name: input
predict pb y: [array([50.5], dtype=float32)]

```





还有一种保存pb模型的方法，保存的东西只有model.pb没有variables。

```python
#保存为pb模型，只有model.pb 没有variables
def export_model_one(session, m, graph):

    output_names = [m.y.op.name] # 定义模型输出

    input_graph_def = graph.as_graph_def()
    print(m.y.op.name, m.a.op.name)
    output_graph_def = convert_variables_to_constants(session, input_graph_def, output_names)
    output_graph = 'pb_model/model.pb'  # 保存地址
    with tf.gfile.GFile(output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

#加载pb模型, 只有model.pb 没有variables
def load_pb_one():
    graph = tf.Graph()
    with graph.as_default():
        output_graph_def = tf.GraphDef()

        with open( 'pb_model/model.pb' , "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        input = sess.graph.get_tensor_by_name("Placeholder:0")
        output = sess.graph.get_tensor_by_name("add:0")

        # 测试pb模型
        for test_x in [[1], [2], [3], [4], [5]]:
            predict_y = sess.run(output, feed_dict={input: test_x})
            print("predict pb y:", predict_y)
```

如果需要发布模型成服务，请看：

<https://www.jianshu.com/p/d673c9507988>

<https://www.jianshu.com/p/5b74f1bc0178>



先看Learning TensorFlow A GUIDE TO BUILDING DEEP LEARNING SYSTEMS

书中的最后一章，完善概念理解

其他相关资料：

TensorFlow 模型保存与恢复 - 简书
https://www.jianshu.com/p/c9fd5c01715e

TensorFlow 保存模型为 PB 文件 - 知乎
https://zhuanlan.zhihu.com/p/32887066

(4条消息)Tensorflow-pb保存与导入 - wc781708249的博客 - CSDN博客
https://blog.csdn.net/wc781708249/article/details/78043099

(4条消息)将tensorflow网络模型（图+权值）保存为.pb文件，并从.pb文件中还原网络模型 - 妙猪的专栏 - CSDN博客
https://blog.csdn.net/guvcolie/article/details/77478973

Saving and Loading a TensorFlow model using the SavedModel API
https://medium.com/@jsflo.dev/saving-and-loading-a-tensorflow-model-using-the-savedmodel-api-17645576527



# 模型融合

模型融合 - Google 搜索
https://www.google.com/search?sourceid=chrome&ie=UTF-8&q=%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88

(5条消息)机器学习模型优化之模型融合 - yeal - CSDN博客
https://blog.csdn.net/u014248127/article/details/78993753

模型融合：kaggle比赛胜出的杀手锏 - 知乎
https://zhuanlan.zhihu.com/p/40131797

【机器学习】模型融合方法概述 - 知乎
https://zhuanlan.zhihu.com/p/25836678



# 使用多GPU加速训练

## 构建多GPU模型

### keras中的multi_gpu_model

https://blog.csdn.net/MachineRandy/article/details/80040765

```python
keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```

**作用：**

- 将模型复制到不同的 GPU 上，具体来说，该功能实现了单机多 GPU 数据并行性。

**原理：**

 下图是两个GPU进行数据并行的示意图，CPU负责梯度平均和参数更新，GPU1和GPU2训练模型副本。

<img src=imgs/multi-gpu.png >

具体步骤如下：

- 将模型的输入分成多个子批次，并根据GPU数量定义模型副本（副本之间要能够共享变量tf.get_variable()）
- 对于单独的GPU，分别从数据管道读取不同的数据块（子批次），然后做forward propagation来计算出loss，再计算在当前variables下的gradients
- 把所有GPU输出的梯度数据转移到CPU上，先进行梯度取平均操作，然后进行模型参数的更新, Concatenate the results (on CPU) into one big batch.（loss求和）
- 重复1-3，直到模型收敛

**Remark：**

- 例如， 如果你的 `batch_size` 是 64，且你使用 `gpus=2`，那么我们将把输入分为两个 32 个样本的子批次，
  在 1 个 GPU 上处理 1 个子批次，然后返回完整批次的 64 个处理过的样本
- 最多支持 8 个 GPU 的准线性加速，此功能目前仅适用于 TensorFlow 后端。



**关于数据并行实现加速的理解：**

- 从数据集迭代的角度：假设我们总共有100个数据，我们预计迭代10个epoch能到达最优点。如果我们只有一个GPU，单个GPU每次最多处理10个数据，那么我们总共需要迭代100次。但如果我们有两个GPU，每个GPU也是处理10个数据，那么我们把100个数据迭代10个epoch只需要50次迭代，这样就实现了线性加速；
- 关于对同一个参数，使用不同的变量求得的梯度，取平均值后进行参数更新：我的理解是反正都是用一批数据的梯度来近似整个数据集的梯度，一次采用20个数据求的梯度和两个10个数据梯度的平均应该也差不多，毕竟你10个10个连续算相对与真实梯度也很不准，还花费了两倍的时间

__参数__：

- __model__: 一个 Keras 模型实例。为了避免OOM错误，该模型可以建立在 CPU 上，详见下面的使用样例。
- __gpus__: 整数 >= 2 或整数列表，创建模型副本的 GPU 数量，或 GPU ID 的列表。
- __cpu_merge__: 一个布尔值，用于标识是否强制合并 CPU 范围内的模型权重。
- __cpu_relocation__: 一个布尔值，用来确定是否在 CPU 的范围内创建模型的权重。如果模型没有在任何一个设备范围内定义，您仍然可以通过激活这个选项来拯救它。

__返回__：

一个 Keras `Model` 实例，它可以像初始 `model` 参数一样使用，但它将工作负载分布在多个 GPU 上。

__例子__：

例 1 - 训练在 CPU 上合并权重的模型

```python
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# 实例化基础模型（或者「模版」模型）。
# 我们推荐在 CPU 设备范围内做此操作，
# 这样模型的权重就会存储在 CPU 内存中。
# 否则它们会存储在 GPU 上，与权重共享冲突。
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# 复制模型到 8 个 GPU 上。
# 这假设你的机器有 8 个可用 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 生成虚拟数据
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# 这个 `fit` 调用将分布在 8 个 GPU 上。
# 由于 batch size 是 256, 每个 GPU 将处理 32 个样本。
parallel_model.fit(x, y, epochs=20, batch_size=256)

# 通过模版模型存储模型（共享相同权重）：
model.save('my_model.h5')
```

**例 2** - 训练在 CPU 上利用 cpu_relocation 合并权重的模型

```python
..
# 不需要更改模型定义的设备范围：
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except ValueError:
    parallel_model = model
    print("Training using single GPU or CPU..")
parallel_model.compile(..)
..
```

例 3 - 训练在 GPU 上合并权重的模型（建议用于 NV-link）

```python
..
# 不需要更改模型定义的设备范围：
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..")

parallel_model.compile(..)
..
```

#### __关于模型保存__

要保存多 GPU 模型，请通过模板模型（传递给 `multi_gpu_model` 的参数）调用 `.save(fname)` 或 `.save_weights(fname)` 以进行存储，而不是通过 `multi_gpu_model` 返回的模型。



在使用 `callbacks.ModelCheckpoint()` 并进行多 gpu 并行计算时，`callbacks` 函数会报错：

```
TypeError: can't pickle ...(different text at different situation) objects
```

这个问题在我之前的文章中也有提到：[[Keras\] 使用Keras调用多GPU，并保存模型
 ](https://www.jianshu.com/p/d57595dac5a9)。显然，在使用检查点时，默认还是使用了 `paralleled_model.save()` ，进而导致错误。为了解决这个问题，我们需要自己定义一个召回函数。

法1：

```python
original_model = ...
parallel_model = multi_gpu_model(original_model, gpus=n)

class MyCbk(keras.callbacks.Callback):

    def __init__(self, model):
         self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save('model_at_epoch_%d.h5' % epoch)

cbk = MyCbk(original_model)
parallel_model.fit(..., callbacks=[cbk])
```

法二：

```python
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

check_point = ParallelModelCheckpoint(single_model ,'best.hd5')
```

法三：

```python
class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss

model.fit(X_train, y_train,
              batch_size=batch_size*G, epochs=nb_epoch, verbose=0, shuffle=True,
              validation_data=(X_valid, y_valid),
              callbacks=[CustomModelCheckpoint(model, '/path/to/save/model.h5')])
```

详见：

<https://www.jianshu.com/p/1d7977599e90>

runtimeerror you must compile your model before using it. keras - Google Search
https://www.google.com/search?newwindow=1&safe=active&ei=oglSXc6OFYH8wQOltprQBA&q=runtimeerror+you+must+compile+your+model+before+using+it.+keras&oq=RuntimeError%3A+You+must+compile+your+model+before+using+it&gs_l=psy-ab.1.2.35i39j0l2.62006.94348..96401...5.0..0.180.830.0j6......0....1..gws-wiz.......0i71j0i22i30j33i299.O-eGCnG6Vl0

ParallelModelCheckpoint keras - Google Search
https://www.google.com/search?sourceid=chrome&ie=UTF-8&q=ParallelModelCheckpoint+keras

(3条消息)[Keras] 使用多 gpu 并行训练并使用 ModelCheckpoint() 可能遇到的问题 - DexterLei - CSDN博客
https://blog.csdn.net/u012862372/article/details/80367607

ModelCheckpoint callback with multi_gpu fails to save the model, throws error after 1st epoch · Issue #8764 · keras-team/keras
https://github.com/keras-team/keras/issues/8764

call_back_error when using multi_gpu_model · Issue #10218 · keras-team/keras
https://github.com/keras-team/keras/issues/10218

Multi_gpu in keras not working with callbacks, but works fine if callback is removed · Issue #8649 · keras-team/keras
https://github.com/keras-team/keras/issues/8649

## 其他方式

### 提升GPU-Util的利用率

numpy在数据创建等操作中默认使用np.float64作为浮点数的数据类型，但GPU只支持32位的数据计算，将所有输入的数据类型设定为np.float32后，GPU利用率将稍微提升。

<https://blog.csdn.net/qing101hua/article/details/78978791>



### 显存管理

```python
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os  
# 指定使用的gpu 
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'    

config = tf.ConfigProto()  
# 设定程序最多只能占用指定gpu50%的显存
config.gpu_options.per_process_gpu_memory_fraction = 0.5   
# or 程序按需申请内存  
config.gpu_options.allow_growth = True      
set_session(tf.Session(config=config))
```

