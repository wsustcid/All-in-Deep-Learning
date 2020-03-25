# 1. Keras中的回调函数

**定义：<https://keras.io/callbacks/>**

回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。

**源码：**

- [[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L148)

**用法：**

你可以传递一个 *列表* 的回调函数（作为 `callbacks` 关键字的参数）到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。在训练时，相应的回调函数的方法就会被在各自的阶段或相应的触发条件下被调用。

```python
#First, callbacks must be instantiated.
cb = Callback(...)

# Then, one or more callbacks that you intend to use 
# must be added to a Python list.
cb_list = [cb, ...]

# Finally, the list of callbacks is provided to the callbacks argument 
# when fitting the model.
model.fit(..., callbacks=cb_list)
```

**Remark: 关于回调函数中的monitor 参数**

- 很多种类型的回调函数中都会用到monitor参数，如 `History`, `EarlyStopping()`, `ModelCheckpoint()`, `RemoteMonitor`, `ReduceLROnPlateau`等， 用来指定回调函数要监控的值。

- fit() 函数*在每一轮训练的最后*，都会计算在 训练集上的 损失函数值；如果我们指定了验证集，验证集上的 损失函数值也会被计算（加val_前缀），这些值都可以指定给monitor参数来被监控进而执行相应的操作。
  - `monitor='loss'`, 可以监控训练集上的损失函数值
  - `monitor='val_loss'` 监控验证集上的损失值

- 如果我们在模型编译时指定了评价指标（因为损失函数值往往无法直观体现训练效果），其值也会被fit函数计算进而可以被回调函数监控，如

  ```python
  model.compile(..., metrics=['acc'])
  ```

  - `monitor='acc'`可以监控训练集上在此轮的精度
  - `monitor='val_acc'` 可以监控验证集上在此轮的精度

  

**完整示例：**

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TerminateOnNaN, EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ProgbarLogger, CSVLogger, TensorBoard

from keras import backend as K

from keras.models import load_model

#from matplotlib import pyplot

epochs = 4000

# generate 2d binary classification dataset
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)

# splitting dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=2)


# define model
model = Sequential()
model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# Callbacks
terminate = TerminateOnNaN()
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=200, min_delta=0,verbose=1)

#model_path = 'model_{epoch:02d}.h5'
model_path = 'best_model.h5'
model_checkpoint = ModelCheckpoint(model_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# 尽量不使用
#def lr_sche(epoch):
#    lr=float(K.get_value(model.optimizer.lr))
#    return lr
#learning_rate_scheduler = LearningRateScheduler(schedule=lr_sche, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=150)

# pbl = ProgbarLogger(count_mode='samples', stateful_metrics='acc')
log_path='train_log.csv'
csv_logger = CSVLogger(log_path)
tensor_board = TensorBoard(log_dir='./logs')

callbacks = [terminate, early_stopping, model_checkpoint, reduce_lr, csv_logger, tensor_board]

History = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=epochs, verbose=1, callbacks=callbacks)


# load the saved model
saved_model = load_model('best_model.h5')
# evaluate the model
_, train_acc = saved_model.evaluate(X_train, Y_train, verbose=1)
_, test_acc = saved_model.evaluate(X_test, Y_test, verbose=1)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```

```python
for key in History.history:
    print key
    
print
for key in History.params:
    print("{}: {}".format(key, History.params[key]))
    
    
==== output ====
acc
loss
val_acc
val_loss
lr

metrics: ['loss', 'acc', 'val_loss', 'val_acc']
samples: 80
batch_size: 32
epochs: 4000
steps: None
do_validation: True
verbose: 1

```



## 1.1 回调函数的种类

### 1.1.1 早停工具

#### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

**作用：**

- 当遇到 NaN 损失时会停止训练



#### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
```

**作用：**

- 当被监测的值不再提升时，则停止训练。

__参数__

- __monitor__: 指定被监测性能指标来停止训练
  - 默认监控验证集上的损失值（对于分类任务，不宜监控精确度）
  - 一般需要监测验证集上的指标，在指标名称前加“val_”前缀。（前提是通过手动，或自动划分的方式为fit函数指定了验证集）
- __mode__: {auto, min, max} 其中之一
  - 在 `min` 模式中，当被监测的数据停止下降，训练就会停止；
  - 在 `max` 模式中，当被监测的数据停止上升，训练就会停止；
  - 在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来

- ***Remark:*** 
  - 以上两个参数是使用早停法的最少设置参数，当选择监控的性能参数**首次**停止提升时，训练将会停止。
  - 但是，这种并不是最优做法，因为参数可能陷入局部最优。为此，我们还有一下三种方式来触发停止训练。
  - patience和min_delta可以同时设置，来灵活的控制停止条件：
    - 默认 0,0, 首次停止最小停止，即停止
    - patience=a, min_delta=b, 只要是连续a轮没有出现大于b的提升，即停止（防止停止条件过于严格，因为可能对某种模型来说，为了0.00001的提升再多训练a轮已经没有太多意义）
    - 因此，我总结为：为了保证度过 局部极值， patience可以设置的大一些，但此时，min_delta也要略微提升，防止训练过久(但结果肯定比min_delta=0时差一些，这是训练时间与精度的trade off)。

- __patience__: 
  - 指定容许监控指标没有进步的**训练轮数**，大于此轮数后训练才被停止
  - 实际测试表明，系统应该是实时记录监控指标的最优值，当最优值在patience个轮次都没有更新时，即早停，因此在最优值之后的patience个轮次中，是允许数据波动的，并不是看相邻两个点之间有没有进步；

- __min_delta__: 
  - 指定监测数据被认为是提升的最小变化，即小于或等于 min_delta 的绝对变化会被认为没有提升。
  - 实际测试表明，只要首次出现小于或等于min_delta的提升，便会停止训练。此值设置过大，会造成程序过早停止，且不容许数据波动。

- __baseline__: 

  - 指定监控指标的基准值，仅当性能指标优于基准值时，训练停止。

  

- __restore_best_weights__: 
  - 是否从具有监测数据最佳值的时期恢复模型权重。如果为 False，则使用在训练的最后一步获得的模型权重。（这样此时模型的参数将不是最佳参数）

- __verbose__: 是否启用详细信息模式

  - 当需要输出训练过程在哪一个epoch停止时，将此参数设置为1。

    ```Python
    Epoch 00227: early stopping
    ```

**Remark:**

- **Why not monitor validation accuracy for early stopping?**

  This is a good question. The main reason is that accuracy is a coarse measure of model performance during training and that loss provides more nuance when using early stopping with classification problems. The same measure may be used for early stopping and model checkpointing in the case of regression, such as mean squared error. (实验结果标明，监控val_loss（需要的时间更久），训练集测试集准确率最后都能达到1，但监控val_acc，测试集很快达到1，但训练集没有)

- 使用val_acc作为模型检查的监控指标，只会保存首次为准确率最高的模型，之后patience个epoch准确率虽然没有提升，但val_loss指标仍在在继续下降，这个要根据具体情况做一下思考，这种方式是否合适。因为相同的准确率可以对应多种参数情况，那种最优不好说。但越往后肯定有

### 1.1.2 检查点

#### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

作用：

- 在每个epoch之后保存模型
- 当我们使用早停法或使用固定训练轮次时，停止训练时并不是最佳模型，因此我们需要设置模型检查点，根据监控指标适时保存模型/模型参数。

__参数__

- __filepath__: 字符串，保存模型的路径:
  - `filepath` 可以包括命名格式选项，可以由 `epoch` 的值和 `logs` 的键（由 `on_epoch_end` 参数传递）来填充。例如：如果 `filepath` 是 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`，那么模型被保存的的文件名就会有训练轮数和验证集损失。
- __monitor__: 被监测的指标值
  - 默认是验证集的损失函数值
- __verbose__: 是否启用详细信息模式，0 或者 1 。
  - 为1时会输出保存信息：Epoch 00001: saving model to ..
- __save_best_only__: Finally, we are interested in only the very best model observed during training, rather than the best compared to the previous epoch, which might not be the best overall if training is noisy. This can be achieved by setting the “*save_best_only*” argument to *True*.
  - False(默认)：每个epoch的训练模型均保存
  - True: 仅当(监控指标下)训练效果比之前的所有都好时进行保存
  - 如果使用固定名称：如 `save_dir + "best_model.h5"` ，则最后仅会有一个模型被保存，当使用动态名称时，会有多个被保存
- __mode__: {auto, min, max} 的其中之一。
  - 如果 `save_best_only=True`，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。
    对于 `val_acc`，模式就会是 `max`，而对于 `val_loss`，模式就需要是 `min`，等等。
  - 在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。
- __save_weights_only__: 
  - 如果 True，那么只有模型的权重会被保存 (`model.save_weights(filepath)`)，
  - 否则的话，整个模型会被保存 (`model.save(filepath)`)。
- __period__: 每个检查点之间的间隔（训练轮数）。



**Remark:**

- 保存模型需要系统支持安装h5py. You can learn more from the [h5py Installation documentation](http://docs.h5py.org/en/latest/build.html).

  ```python
  pip install h5py
  ```



### 1.1.3 动态调参工具

#### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

**作用：**学习速率定时器，在每一个epoch调用此函数用来更新学习速率，学习速率的更新方式在通过自定义一个schedule函数实现。

**源码：**

```python
class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
```



__参数__

- __schedule__: 指定一个学习率更新函数
  - 函数输入：老版本仅有一个(epoch)，新版本为 (epoch, lr)
  - 返回：一个学习速率作为输出（浮点数）
  - 需要注意的是，这个函数的输入参数是LearningRateScheduler自己传入的初始epoch=0与lr=当前模型的学习速率，我们无法自己传入，但可以进行一些修改来实现控制返回值
- __verbose__: 整数。 0：安静，1：更新信息。

**使用：**当前很多优化器自带学习速度调整，因此尽量要使用此来调整

```python
# 方式1 老版本keras：（效果不好，尽量不用）
from keras.callbacks import LearningRateScheduler
from keras import backend as K
def lr_sche(epoch):
    # 固定步长降低学习率
    if epoch%10==0:
        lr=0.1*float(K.get_value(model.optimizer.lr))
    else:
        # 必须每个epoch都返回学习率
        lr=float(K.get_value(model.optimizer.lr))
    return lr
learning_rate_scheduler = LearningRateScheduler(schedule=lr_sche, verbose=1)

# 方式2 （高级封装）
import numpy as np
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        # 以epoch和step_size作为参数
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)

model.fit(X_train, Y_train, callbacks=[lr_sched])
```



#### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
```

**作用：**

- 当监控指标停止提升时，降低学习速率
- 当学习停止时，模型总是会受益于降低 2-10 倍的学习速率。这个回调函数监测一个数据并且当这个数据在一定 `patience` 的训练轮之后还没有进步，那么学习速率就会被降低。

__参数__

- __monitor__: 被监测的数据。
- __factor__: 学习速率被降低的因数。新的学习速率 = 学习速率 * 因数
- __patience__: 没有进步的训练轮数，在这之后训练速率会被降低。
  - 这里的patience最好和earlystopping中的patience配合使用，实验表明，二者配合使用时，能进一步提升训练效果。
- __verbose__: 整数。0：安静，1：更新信息。
- __mode__: {auto, min, max} 其中之一。如果是 `min` 模式，学习速率会被降低如果被监测的数据已经停止下降；
在 `max` 模式，学习塑料会被降低如果被监测的数据已经停止上升；
在 `auto` 模式，方向会被从被监测的数据中自动推断出来。
- __min_delta__: 对于测量新的最优化的阀值，只关注巨大的改变。
- __cooldown__: 在学习速率被降低之后，重新恢复正常操作之前等待的训练轮数量。
- __min_lr__: 学习速率的下边界



### 1.1.4 记录与监控工具

#### BaseLogger

```python
keras.callbacks.BaseLogger(stateful_metrics=None)
```

**源码：(待研究)**

```python
class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """

    def __init__(self, stateful_metrics=None):
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.stateful_metrics:
                self.totals[k] = v
            else:
                if k in self.totals:
                    self.totals[k] += v * batch_size
                else:
                    self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    if k in self.stateful_metrics:
                        logs[k] = self.totals[k]
                    else:
                        logs[k] = self.totals[k] / self.seen

```



**作用：**

- Callback that accumulates epoch averages of metrics.

- ***This callback is automatically applied to every Keras model.***

__参数__

__stateful_metrics__: 

- Iterable of string names of metrics that should *not* be averaged over an epoch. 
- Metrics in this list will be logged as-is in `on_epoch_end`. All others will be averaged in `on_epoch_end`.

#### History

```python
keras.callbacks.History()
```

**作用：**

- 把所有事件都记录到 `History` 对象的回调函数
- 此回调函数被自动启用到每一个 Keras 模型。
- `History` 对象会被模型的 `fit` 方法返回，如`History=model.fit()`

**用法：**

- `History`对象的 `history` 属性是一个字典，`key`为损失函数和评价指标的字符串名称，同样分为训练集和验证集两种，value值为一个列表，里面存储了每个epoch对应参数的平均值
- `History` 对象的  `params` 也是一个字典，里面存储了训练模型相关的参数及对应的值



#### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
```

**作用：**

Callback that prints metrics to stdout.

**Arguments**

- **count_mode**: One of "steps" or "samples". Whether the progress bar should count samples seen or steps (batches) seen.
- **stateful_metrics**: Iterable of string names of metrics that should *not* be averaged over an epoch. Metrics in this list will be logged as-is. All others will be averaged over time (e.g. loss, etc).

**Raises**

- **ValueError**: In case of invalid `count_mode`.



#### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

**作用：**

- Callback that streams epoch results to a csv file.
- Supports all values that can be represented as a string, including 1D iterables such as np.ndarray.
- 相当于把`History.history`中存储的值直接存到文件中，方便后期查看或绘制训练过程图

__参数__

- __filename__: csv 文件的文件名，例如 'run/log.csv' 或者‘train.log’。
- __separator__: 用来隔离 csv 文件中元素的字符串。
- __append__: True：如果文件存在则增加（可以被用于继续训练）。False：覆盖存在的文件。



#### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
```

TensorBoard basic visualizations. [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) is a visualization tool provided with TensorFlow.

**作用：**

- This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
- as well as activation histograms for the different layers in your model.

__参数__

- **log_dir**: the path of the directory where to save the log files to be parsed by TensorBoard.

- **histogram_freq**: frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. 
  - If set to 0, histograms won't be computed. 
  - Validation data (or split) must be specified for histogram visualizations.

- **batch_size**: size of batch of inputs to feed to the network for histograms computation.

- **write_graph**: whether to visualize the graph in TensorBoard. 
  - The log file can become quite large when write_graph is set to True.

- **write_grads**: whether to visualize gradient histograms in TensorBoard.
  - `histogram_freq` must be greater than 0.

- **write_images**: whether to write model weights to visualize as image in TensorBoard.

- **embeddings_freq**: frequency (in epochs) at which selected embedding layers will be saved. 
  - If set to 0, embeddings won't be computed. 
  - Data to be visualized in TensorBoard's Embedding tab must be passed as `embeddings_data`.

- **embeddings_layer_names**: a list of names of layers to keep eye on. 
  - If None or empty list all the embedding layer will be watched.

- **embeddings_metadata**: a dictionary which maps layer name to a file name in which metadata for this embedding layer is saved. 
  - See the [details](https://www.tensorflow.org/guide/embedding#metadata) about metadata files format. In case if the same metadata file is used for all embedding layers, string can be passed.

- **embeddings_data**: data to be embedded at layers specified in `embeddings_layer_names`. 
  - Numpy array (if the model has a single input) or list of Numpy arrays (if the model has multiple inputs). Learn [more about embeddings](https://www.tensorflow.org/guide/embedding).

- **update_freq**: `'batch'` or `'epoch'` or integer. 
  - When using `'batch'`, writes the losses and metrics to TensorBoard after each batch. 
  - The same applies for `'epoch'`. 
  - If using an integer, let's say `10000`, the callback will write the metrics and losses to TensorBoard every 10000 samples. 
  - Note that writing too frequently to TensorBoard can slow down your training.

**使用：**

If you have installed TensorFlow with pip, you should be able to launch TensorBoard from the command line:

```sh
tensorboard --logdir=/full_path_to_your_logs_folder
```

然后右键在浏览器中打开相应的链接即可。



#### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
```

**作用：**

- 将事件数据流到服务器的回调函数。需要 `requests` 库。
- 事件被默认发送到 `root + '/publish/epoch/end/'`。、
- 采用 HTTP POST ，其中的 `data` 参数是以 JSON 编码的事件数据字典。
- 如果 send_as_json 设置为 True，请求的 content type 是 application/json。否则，将在表单中发送序列化的 JSON。

__参数__

- __root__: 字符串；目标服务器的根地址。
- __path__: 字符串；相对于 `root` 的路径，事件数据被送达的地址。
- __field__: 字符串；JSON ，数据被保存的领域。
- __headers__: 字典；可选自定义的 HTTP 的头字段。
- __send_as_json__: 布尔值；请求是否应该以 application/json 格式发送。\



#### LambdaCallback

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



## 1.2 自定义回调函数

使用者可以通过在keras.callbacks.LambdaCallback中指定操作或对keras.callbacks.Callback进行子类继承创建自定义的回馈器。对第1种选择，keras.callbacks.LambdaCallback对象的创建方法为 [13]  ：

```
`keras.callbacks.LambdaCallback(on_epoch_begin``=``None``, on_epoch_end``=``None``, ``                               ``on_batch_begin``=``None``, on_batch_end``=``None``, ``                               ``on_train_begin``=``None``, on_train_end``=``None``)`
```

这里keras.callbacks.LambdaCallback的使用方法类似于Python中的[匿名函数](https://baike.baidu.com/item/匿名函数/4337265)，格式中的6个参数表示在一个学习的不同阶段可以进行的操作，具体展开如下 [13]  ：

- on_epoch_begin和on_epoch_end表示在每个学习纪元开始和结束时的操作，该处接收使用epoch和logs定义的匿名函数。
- on_batch_begin和on_batch_end表示在每个样本批次开始和结束时的操作，该处接收使用batch和logs定义的匿名函数。
- on_train_begin和on_train_end表示在每个学习纪元开始和结束时的操作，该处接收使用logs定义的匿名函数。

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

你可以通过扩展 `keras.callbacks.Callback` 基类来创建一个自定义的回调函数。
通过类的属性 `self.model`，回调函数可以获得它所联系的模型。

下面是一个简单的例子，

[http://zhouchen.tech/2019/02/28/Keras%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B-%EF%BC%9A%E4%BD%BF%E7%94%A8%E8%BD%BB%E6%9D%BE%E7%9A%84%E6%96%B9%E5%BC%8F%E6%90%AD%E5%BB%BA%E7%BD%91%E7%BB%9C/](http://zhouchen.tech/2019/02/28/Keras使用教程-：使用轻松的方式搭建网络/)

在训练时，保存一个列表的批量损失值：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```



例: 记录损失历史

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

# 2. 相关理论

## 2.1 早停法

当我们训练深度学习神经网络的时候通常希望能获得最好的泛化性能（**generalization performance**，即可以很好地拟合数据）。

但是所有的标准深度学习神经网络结构如全连接多层感知机都很容易**过拟合**：

- 当网络在训练集上表现越来越好，错误率越来越低的时候，实际上在某一刻，它在测试集的表现已经开始变差。
- 模型的泛化能力通常使用模型在验证数据集（validation set）上的表现来评估。随着网络的优化，我们期望当模型在训练集上的误差降低的时候，其在验证集上的误差表现不会变差。反之，当模型在训练集上表现很好，在验证集上表现很差的时候，我们认为模型出现了过拟合（overfitting）的情况。

解决过拟合问题有两个方向：降低参数空间的维度或者降低每个维度上的有效规模（effective size）。

- 降低参数数量的方法包括greedy constructive learning、剪枝和权重共享等。
- 降低每个参数维度的有效规模的方法主要是正则化，如权重衰变（weight decay）和早停法（early stopping）等。



**相关论文：**

- <http://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf>

### 简介：

早停法是一种被广泛使用的方法，在很多案例上都比正则化的方法要好。其基本含义是在训练中计算模型在验证集上的表现，当模型在验证集上的表现开始下降的时候，停止训练，这样就能避免继续训练导致过拟合的问题。其主要步骤如下：

1. 将原始的训练数据集划分成训练集和验证集
2. 只在训练集上进行训练，并每个一个周期计算模型在验证集上的误差，例如，每15次epoch（mini batch训练中的一个周期）
3. 当模型在验证集上的误差比上一次训练结果差的时候停止训练
4. 使用上一次迭代结果中的参数作为模型的最终参数

然而，在现实中，模型在验证集上的误差不会像上图那样平滑，而是像下图一样：

<img src=https://www.researchgate.net/profile/Lutz_Prechelt/publication/2874749/figure/fig1/AS:645735506771969@1530966750067/A-real-validation-error-curve-Vertical-validation-set-error-horizontal-time-in.png height=250/>

<center> 真实的验证集误差变化曲线</center>

也就是说，模型在验证集上的表现可能在短暂的变差之后有可能继续变好:

- 上图在训练集迭代到400次的时候出现了16个局部最低。其中全局最优大约出现在第205次迭代中。首次出现最低点是第45次迭代。
- 相比较第45次迭代停止，到第400次迭代停止的时候找出的最低误差比第45次提高了1.1%，但是训练时间大约是前者的7倍。

但是，并不是所有的误差曲线都像上图一样，有可能在出现第一次最低点之后，后面再也没有比当前最低点更低的情况了。所以:

- **早停法主要是训练时间和泛化错误之间的权衡。**
- 尽管如此，也有某些停止标准也可以帮助我们寻找更好的权衡。



### 使用

我们需要一个停止的标准来实施早停法，因此，我们希望它可以产生最低的泛化错误，同时也可以有最好的性价比，即给定泛化错误下的最小训练时间

#### 停止标准

停止标准有很多，也很灵活，大约有三种。在给出早停法的具体标准之前，我们先确定一下符号。假设我们使用$E$作为训练算法的误差函数，那么$E_{tr}(t)$是训练数据上的误差，$E_{te}(t)$是测试集上的误差。实际情况下我们并不能知道泛化误差，因此我们使用验证集误差来估计它。

**第一类停止标准**

假设$E_{opt}(t)$是在迭代次数$t$时取得最好的验证集误差：
$$
E_{opt}(t) := \text{min}_{t'\leq t}E_{va}(t')
$$
我们定义一个新变量叫**泛化损失（generalization loss）**，它描述的是在当前迭代周期t中，泛化误差相比较目前的最低的误差的一个增长率：
$$
GL(t) = 100 \cdot \big( \frac{E_{va}(t)}{E_{opt}(t)} - 1 \big)
$$
较高的泛化损失显然是停止训练的一个候选标准，因为它直接表明了过拟合。这就是第一类的停止标准，即当泛化损失超过一定阈值的时候，停止训练。我们用$GL_{\alpha}$来定义
$$
GL_{\alpha} > \alpha
$$
**第二类停止标准**

然而，当训练的速度很快的时候，我们可能希望模型继续训练。因为如果训练错误依然下降很快，那么泛化损失有很大概率被修复。我们通常会**假设过拟合只会在训练错误降低很慢的时候出现**。在这里，我们定义一个$k$周期，以及基于周期的一个新变量**度量进展（measure progress）**：
$$
P_k(t) = 1000 \cdot \big( \frac{ \sum_{t' = t-k+1}^t E_{tr}(t') }{ k \cdot min_{t' = t-k+1}^t E_{tr}(t') } -1 \big)
$$
它表达的含义是，当前的指定迭代周期内的平均训练错误比该期间最小的训练错误大多少。

- 当训练过程不稳定的时候，这个measure progress结果可能很大，其中训练错误会变大而不是变小。实际中，很多算法都由于选择了不适当的较大的步长而导致这样的抖动。除非全局都不稳定，否则在较长的训练之后，measure progress结果趋向于0（其实这个就是度量训练集错误在某段时间内的平均下降情况）。

由此，我们引入了第二个停止标准，即泛化损失和进展的商$PQ_{\alpha}$大于指定值的时候停止，即
$$
\frac{GL(t)}{P_k(t)} \gt \alpha
$$
**第三类停止标准**
第三类停止标准则完全依赖于泛化错误的变化，即当泛化错误在连续s个周期内增长的时候停止（UP）。

当验证集错误在连续s个周期内出现增长的时候，我们假设这样的现象表明了过拟合，它与错误增长了多大独立。这个停止标准可以度量局部的变化，因此可以用在剪枝算法中，即在训练阶段，允许误差可以比前面最小值高很多时候保留。



#### 停止标准选择

一般情况下，“较慢”的标准会相对而言在平均水平上表现略好，可以提高泛化能力。然而，这些标准需要较长的训练时间。其实，总体而言，这些标准在系统性的区别很小。主要选择规则包括：

1. 除非较小的提升也有很大价值，负责选择较快的停止标准
2. 为了最大可能找到一个好的方案，使用GL标准
3. 为了最大化平均解决方案的质量，如果网络只是过拟合了一点点，可以使用PQ标准，否则使用UP标准

注意，目前并没有理论上可以证明那种停止标准较好，所以都是实验的数据。后续我们再介绍一下实验结果。

