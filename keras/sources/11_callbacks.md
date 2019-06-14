## 概述

回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。

**源码：**

- [[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L148)

**使用方法：**

你可以传递一个列表的回调函数（作为 `callbacks` 关键字的参数）到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。在训练时，相应的回调函数的方法就会被在各自的阶段被调用。

```python
#First, callbacks must be instantiated.
cb = Callback(...)

# Then, one or more callbacks that you intend to use must be added to a Python list.
cb_list = [cb, ...]

# Finally, the list of callbacks is provided to the callback argument when fitting the model.
model.fit(..., callbacks=cb_list)
```

**使用回调函数监控模型性能：**

在每一轮训练的最后，都会计算我们选择优化的损失函数的值，我们可以通过多种方式监控此值：

- 在回调函数中通过监控`“loss”`, 我们可以得到训练集上的损失函数值

- 如果我们使用了验证集，可以通过监控`“val_loss”` 的到验证集上的损失值

- 我们同样可以监控其他值，如评价指标，但要在 model.compile() 函数中事先指定

  ```python
  model.compile(..., metrics=['acc'])
  ```

  - 一旦我们指定了此评价指标，此指标就可以在训练过程中被监控
  - 同样，在回调函数中指定 `“acc”` 或`“mse”`可以监控训练集的精度或最小均方误差
  - 指定 `“val_acc”` 或`“val_mse”`可以监控验证集的精度或最小均方误差



## 回调函数的种类

### Callback

```python
keras.callbacks.Callback()
```

用来组建新的回调函数的抽象基类。

__属性__

- __params__: 字典。训练参数，
(例如，verbosity, batch size, number of epochs...)。
- __model__: `keras.models.Model` 的实例。
指代被训练模型。

被回调函数作为参数的 `logs` 字典，它会含有于当前批量或训练轮相关数据的键。

目前，`Sequential` 模型类的 `.fit()` 方法会在传入到回调函数的 `logs` 里面包含以下的数据：

- __on_epoch_end__: 包括 `acc` 和 `loss` 的日志， 也可以选择性的包括 `val_loss`（如果在 `fit` 中启用验证），和 `val_acc`（如果启用验证和监测精确值）。
- __on_batch_begin__: 包括 `size` 的日志，在当前批量内的样本数量。
- __on_batch_end__: 包括 `loss` 的日志，也可以选择性的包括 `acc`（如果启用监测精确值）。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L204)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger(stateful_metrics=None)
```

会积累训练轮平均评估的回调函数。

这个回调函数被自动应用到每一个 Keras 模型上面。

__参数__

__stateful_metrics__: 可重复使用不应在一个 epoch 上平均的指标的字符串名称。
此列表中的度量标准将按原样记录在 `on_epoch_end` 中。
所有其他指标将在 `on_epoch_end` 中取平均值。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L251)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

当遇到 NaN 损失会停止训练的回调函数。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L264)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
```

会把评估以标准输出打印的回调函数。

__参数__

- __count_mode__: "steps" 或者 "samples"。
进度条是否应该计数看见的样本或步骤（批量）。
__stateful_metrics__: 可重复使用不应在一个 epoch 上平均的指标的字符串名称。
此列表中的度量标准将按原样记录在 `on_epoch_end` 中。
所有其他指标将在 `on_epoch_end` 中取平均值。


__异常__

- __ValueError__: 如果 `count_mode`

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L341)</span>
### History

```python
keras.callbacks.History()
```

把所有事件都记录到 `History` 对象的回调函数。

这个回调函数被自动启用到每一个 Keras 模型。`History` 对象会被模型的 `fit` 方法返回。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L360)</span>



### EarlyStopping()

This callback allows you to specify the performance measure to monitor, the trigger, and once triggered, it will stop the training process.

**Requirements：**

- 早停法要求在训练时提供验证集，有两种提供验证集的方法：

  - 将训练数据手动分为训练集和验证集后（比如在数据集读取阶段分）分别提供给 model.fit() 函数

    ```python
    model.fit(train_X, train_y, validation_data=(val_x, val_y))
    ```

  - 直接提供完整训练数据集，并指定`validation_split`参数一个0-1之间的值，依靠model.fit()函数自动划分

    ```python
    model.fit(train_X, train_y, validation_split=0.3)
    ```

  - 通过以上两种方式，模型将在每个epoch利用训练集进行训练(loss)，依靠验证集进行评估（`metrics`）

**使用：**

The *EarlyStopping* callback is configured when instantiated via arguments.

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```

当被监测的数量不再提升，则停止训练。

__参数__

- __monitor__: 指定被监测性能指标来停止训练
  - 要监测验证集上的指标，一般都要加“val_”前缀，如监测验证集上的损失时，使用'val_loss'
- __mode__: {auto, min, max} 其中之一
  - 在 `min` 模式中，当被监测的数据停止下降，训练就会停止；
  - 在 `max` 模式中，当被监测的数据停止上升，训练就会停止；
  - 在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来

*以上两个参数是使用早停法的最简实现，当选择监控的性能参数首次停止提升时，训练将会停止。*

- __verbose__: 详细信息模式
  - 当需要输出训练过程在哪一个epoch停止时，将此参数设置为1

***但是，性能参数首次停止提升便停止训练并不是最优做法，因为参数可能陷入局部最优。为此，我们可以给早停法的触发通过patience参数设定一个延时：***

- __patience__: 指定没有进步的训练轮数，在这之后训练才被停止

*我们亦可指定多大的提升才算做提升（默认大于0即算）*

- __min_delta__: 在被监测的数据中被认为是提升的最小变化，
  - 例如，小于 min_delta 的绝对变化会被认为没有提升。

*if you have familiarity with the training of the model (e.g. learning curves) and know that once a validation loss of a given value is achieved that there is no point in continuing training. This can be specified by setting the “*baseline*” argument.*

- __baseline__: 要监控的数量的基准值
  - 仅当性能指标优于基准值时，训练停止



- __restore_best_weights__: 是否从具有监测数量的最佳值的时期恢复模型权重
- 如果为 False，则使用在训练的最后一步获得的模型权重。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L574)</span>

### ModelCheckpoint()

The *EarlyStopping* callback will stop training once triggered, but the model at the end of training may not be the model with best performance on the validation dataset.

An additional callback is required that will save the best model observed during training for later use. This is the *ModelCheckpoint* callback.

**Requirements:**

Saving and loading models requires that HDF5 support has been installed on your workstation. For example, using the *pip* Python installer, this can be achieved as follows:

```python
sudo pip install h5py
```

You can learn more from the [h5py Installation documentation](http://docs.h5py.org/en/latest/build.html).



**Usage:**

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

在每个训练期之后保存模型。

__参数__

- __filepath__: 字符串，保存模型的路径:
  - 若使用固定的文件名如 `save_dir + "best_model.h5"` ，则 ==实验==
  - `filepath` 可以包括命名格式选项，可以由 `epoch` 的值和 `logs` 的键（由 `on_epoch_end` 参数传递）来填充。例如：如果 `filepath` 是 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`，那么模型被保存的的文件名就会有训练轮数和验证集损失。
- __monitor__: 被监测的数据。默认是验证集的损失函数值
- __verbose__: 是否启用详细信息模式，0 或者 1 。
  - 为1时会输出保存信息：Epoch 00001: saving model to ..
- __save_best_only__: 
- Finally, we are interested in only the very best model observed during training, rather than the best compared to the previous epoch, which might not be the best overall if training is noisy. This can be achieved by setting the “*save_best_only*” argument to *True*.
  - 默认为False，==实验==
  - 如果 `save_best_only=True`，仅当此次回调时，监控指标比上一次好时才保存模型，性能没有提升则不会保存此轮训练模型。（使用固定文件名时，仅保存整个训练过程中最佳的模型）
- __mode__: {auto, min, max} 的其中之一。
  如果 `save_best_only=True`，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。
  对于 `val_acc`，模式就会是 `max`，而对于 `val_loss`，模式就需要是 `min`，等等。
  在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。
- __save_weights_only__: 
  - 如果 True，那么只有模型的权重会被保存 (`model.save_weights(filepath)`)，
  - 否则的话，整个模型会被保存 (`model.save(filepath)`)。
- __period__: 每个检查点之间的间隔（训练轮数）。

------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L460)</span>

#### Example

```python
# mlp overfit on the moons dataset with patient early stopping and model checkpointing
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot
from keras.models import load_model
# generate 2d classification dataset
X, y = make_moons(n_samples=100, noise=0.2, random_state=1)
# split into train and test
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# simple early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(trainX, trainy, verbose=0)
_, test_acc = saved_model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

**Questions:**

1. Why not monitor validation accuracy for early stopping?

   This is a good question. The main reason is that accuracy is a coarse measure of model performance during training and that loss provides more nuance when using early stopping with classification problems. The same measure may be used for early stopping and model checkpointing in the case of regression, such as mean squared error.

### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
```

将事件数据流到服务器的回调函数。

需要 `requests` 库。
事件被默认发送到 `root + '/publish/epoch/end/'`。
采用 HTTP POST ，其中的 `data` 参数是以 JSON 编码的事件数据字典。
如果 send_as_json 设置为 True，请求的 content type 是 application/json。否则，将在表单中发送序列化的 JSON。

__参数__

- __root__: 字符串；目标服务器的根地址。
- __path__: 字符串；相对于 `root` 的路径，事件数据被送达的地址。
- __field__: 字符串；JSON ，数据被保存的领域。
- __headers__: 字典；可选自定义的 HTTP 的头字段。
- __send_as_json__: 布尔值；请求是否应该以 application/json 格式发送。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L633)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

学习速率定时器。

__参数__

- __schedule__: 一个函数，接受轮索引数作为输入（整数，从 0 开始迭代）
然后返回一个学习速率作为输出（浮点数）。
- __verbose__: 整数。 0：安静，1：更新信息。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L669)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
```

Tensorboard 基本可视化。

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
是由 Tensorflow 提供的一个可视化工具。

这个回调函数为 Tensorboard 编写一个日志，
这样你可以可视化测试和训练的标准评估的动态图像，
也可以可视化模型中不同层的激活值直方图。

如果你已经使用 pip 安装了 Tensorflow，你应该可以从命令行启动 Tensorflow：

```sh
tensorboard --logdir=/full_path_to_your_logs
```

__参数__

- __log_dir__: 用来保存被 TensorBoard 分析的日志文件的文件名。
- __histogram_freq__: 对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。
如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
- __write_graph__: 是否在 TensorBoard 中可视化图像。
如果 write_graph 被设置为 True，日志文件会变得非常大。
- __write_grads__: 是否在 TensorBoard  中可视化梯度值直方图。
`histogram_freq` 必须要大于 0 。
- __batch_size__: 用以直方图计算的传入神经元网络输入批的大小。
- __write_images__: 是否在 TensorBoard 中将模型权重以图片可视化。
- __embeddings_freq__: 被选中的嵌入层会被保存的频率（在训练轮中）。
- __embeddings_layer_names__: 一个列表，会被监测层的名字。
如果是 None 或空列表，那么所有的嵌入层都会被监测。
- __embeddings_metadata__: 一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字。
查看 [详情](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
关于元数据的数据格式。
以防同样的元数据被用于所用的嵌入层，字符串可以被传入。
- __embeddings_data__: 要嵌入在 `embeddings_layer_names` 指定的层的数据。
Numpy 数组（如果模型有单个输入）或 Numpy 数组列表（如果模型有多个输入）。
[Learn ore about embeddings](https://www.tensorflow.org/programmers_guide/embedding)。
- __update_freq__: `'batch'` 或 `'epoch'` 或 整数。当使用 `'batch'` 时，在每个 batch 之后将损失和评估值写入到 TensorBoard 中。同样的情况应用到 `'epoch'` 中。如果使用整数，例如 `10000`，这个回调会在每 10000 个样本之后将损失和评估值写入到 TensorBoard 中。注意，频繁地写入到 TensorBoard 会减缓你的训练。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1017)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

当标准评估停止提升时，降低学习速率。

当学习停止时，模型总是会受益于降低 2-10 倍的学习速率。
这个回调函数监测一个数据并且当这个数据在一定「有耐心」的训练轮之后还没有进步，
那么学习速率就会被降低。

__例子__


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__参数__

- __monitor__: 被监测的数据。
- __factor__: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
- __patience__: 没有进步的训练轮数，在这之后训练速率会被降低。
- __verbose__: 整数。0：安静，1：更新信息。
- __mode__: {auto, min, max} 其中之一。如果是 `min` 模式，学习速率会被降低如果被监测的数据已经停止下降；
在 `max` 模式，学习塑料会被降低如果被监测的数据已经停止上升；
在 `auto` 模式，方向会被从被监测的数据中自动推断出来。
- __min_delta__: 对于测量新的最优化的阀值，只关注巨大的改变。
- __cooldown__: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
- __min_lr__: 学习速率的下边界。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1138)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

把训练轮结果数据流到 csv 文件的回调函数。

支持所有可以被作为字符串表示的值，包括 1D 可迭代数据，例如，np.ndarray。

__例子__


```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__参数__

- __filename__: csv 文件的文件名，例如 'run/log.csv'。
- __separator__: 用来隔离 csv 文件中元素的字符串。
- __append__: True：如果文件存在则增加（可以被用于继续训练）。False：覆盖存在的文件。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1226)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

在训练进行中创建简单，自定义的回调函数的回调函数。

这个回调函数和匿名函数在合适的时间被创建。
需要注意的是回调函数要求位置型参数，如下：

- `on_epoch_begin` 和 `on_epoch_end` 要求两个位置型的参数：
`epoch`, `logs`
- `on_batch_begin` 和 `on_batch_end` 要求两个位置型的参数：
`batch`, `logs`
- `on_train_begin` 和 `on_train_end` 要求一个位置型的参数：
`logs`

__参数__

- __on_epoch_begin__: 在每轮开始时被调用。
- __on_epoch_end__: 在每轮结束时被调用。
- __on_batch_begin__: 在每批开始时被调用。
- __on_batch_end__: 在每批结束时被调用。
- __on_train_begin__: 在模型训练开始时被调用。
- __on_train_end__: 在模型训练结束时被调用。

__例子__


```python
# 在每一个批开始时，打印出批数。
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# 把训练轮损失数据流到 JSON 格式的文件。文件的内容
# 不是完美的 JSON 格式，但是时每一行都是 JSON 对象。
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# 在完成模型训练之后，结束一些进程。
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```

---


# 创建一个回调函数

你可以通过扩展 `keras.callbacks.Callback` 基类来创建一个自定义的回调函数。
通过类的属性 `self.model`，回调函数可以获得它所联系的模型。

下面是一个简单的例子，在训练时，保存一个列表的批量损失值：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### 例: 记录损失历史

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# 输出
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### 例: 模型检查点

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
如果验证损失下降， 那么在每个训练轮之后保存模型。
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```
