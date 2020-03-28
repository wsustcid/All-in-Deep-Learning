[TOC]

# 1. Keras快速入门

![](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

## 1.1 简介

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 或者 [Theano](https://github.com/Theano/Theano) 作为后端运行。Keras 的开发重点是支持快速的实验。*能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。*

如果你在以下情况下需要深度学习库，请使用 Keras：

- 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
- 同时支持卷积神经网络和循环神经网络，以及两者的组合。
- 在 CPU 和 GPU 上无缝运行。

**Remark：**

- 查看文档，请访问 [Keras.io](https://keras-zh.readthedocs.io/)。

- Keras 兼容的 Python 版本: __Python 2.7-3.6__。

### 1.1.1 为什么使用Keras

- **Keras 优先考虑开发人员的经验**
  - Keras 是为人类而非机器设计的 API。[Keras 遵循减少认知困难的最佳实践](https://blog.keras.io/user-experience-design-for-apis.html): 它提供一致且简单的 API，它将常见用例所需的用户操作数量降至最低，并且在用户错误时提供清晰和可操作的反馈。
  - 这使 Keras 易于学习和使用。作为 Keras 用户，你的工作效率更高，能够比竞争对手更快地尝试更多创意，从而[帮助你赢得机器学习竞赛](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions)。
  - 这种易用性并不以降低灵活性为代价：因为 Keras 与底层深度学习语言（特别是 TensorFlow）集成在一起，所以它可以让你实现任何你可以用基础语言编写的东西。特别是，`tf.keras` 作为 Keras API 可以与 TensorFlow 工作流无缝集成。(并不会出现Tensorflow可以实现而Keras无法实现的情况)
- **Keras 被工业界和学术界广泛采用**
  - 截至 2018 年中期，Keras 拥有超过 250,000 名个人用户。与其他任何深度学习框架相比，Keras 在行业和研究领域的应用率更高（除 TensorFlow 之外，且 Keras API 是 TensorFlow 的官方前端，通过 `tf.keras` 模块使用）。

<img src='https://s3.amazonaws.com/keras.io/img/dl_frameworks_power_scores.png' style='width:500px; display: block; margin: 0 auto;'/>

<p style='font-style: italic; font-size: 10pt; text-align: center;'>
    Deep learning 框架排名，由 Jeff Hale 基于 7 个分类的 11 个数据源计算得出
</i>
</p>



- **Keras 可以轻松将模型转化为产品**，与任何其他深度学习框架相比，你的 Keras 模型可以在更广泛的平台上轻松部署：
  - 在 iOS 上，通过 [Apple’s CoreML](https://developer.apple.com/documentation/coreml)（苹果为 Keras 提供官方支持）。这里有一个[教程](https://www.pyimagesearch.com/2018/04/23/running-keras-models-on-ios-with-coreml/)。
  - 在 Android 上，通过 TensorFlow Android runtime，例如：[Not Hotdog app](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)。
  - 在浏览器中，通过 GPU 加速的 JavaScript 运行时，例如：[Keras.js](https://transcranial.github.io/keras-js/#/) 和 [WebDNN](https://mil-tokyo.github.io/webdnn/)。
  - 在 Google Cloud 上，通过 [TensorFlow-Serving](https://www.tensorflow.org/serving/)。
  - [在 Python webapp 后端（比如 Flask app）中](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)。
  - 在 JVM 上，通过 [SkyMind 提供的 DL4J 模型导入](https://deeplearning4j.org/model-import-keras)。
  - 在 Raspberry Pi 树莓派上。

- **Keras 支持多个后端引擎，不会将你锁定到一个生态系统中**

  你的 Keras 模型可以基于不同的[深度学习后端](https://keras.io/zh/backend/)开发。重要的是，任何仅利用内置层构建的 Keras 模型，都可以在所有这些后端中移植：你可以用一种后端训练模型，再将它载入另一种后端中（例如为了发布的需要）。支持的后端有：

  - 谷歌的 TensorFlow 后端
  - 微软的 CNTK 后端
  - Theano 后端

  亚马逊也正在为 Keras 开发 MXNet 后端。

  如此一来，你的 Keras 模型可以在 CPU 之外的不同硬件平台上训练：

  - [NVIDIA GPU](https://developer.nvidia.com/deep-learning)
  - [Google TPU](https://cloud.google.com/tpu/)，通过 TensorFlow 后端和 Google Cloud
  - OpenCL 支持的 GPU, 比如 AMD, 通过 [PlaidML Keras 后端](https://github.com/plaidml/plaidml)

- **Keras 拥有强大的多 GPU 和分布式训练支持**
  - Keras [内置对多 GPU 数据并行的支持](https://keras.io/zh/utils/#multi_gpu_model)。
  - 优步的 [Horovod](https://github.com/uber/horovod) 对 Keras 模型拥有一流的支持。
  - Keras 模型[可以被转换为 TensorFlow Estimators](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator) 并在 [Google Cloud 的 GPU 集群](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)上训练。
  - Keras 可以在 Spark（通过 CERN 的 [Dist-Keras](https://github.com/cerndb/dist-keras)）和 [Elephas](https://github.com/maxpumperla/elephas) 上运行。

- **Keras 的发展得到深度学习生态系统中的关键公司的支持**

  Keras 的开发主要由谷歌支持，Keras API 以 `tf.keras` 的形式包装在 TensorFlow 中。此外，微软维护着 Keras 的 CNTK 后端。亚马逊 AWS 正在开发 MXNet 支持。其他提供支持的公司包括 NVIDIA、优步、苹果（通过 CoreML）等。



### 1.1.2 安装指引

在安装 Keras 之前，请安装以下后端引擎之一：TensorFlow，Theano，或者 CNTK。我们推荐 TensorFlow 后端。

- [TensorFlow 安装指引](https://www.tensorflow.org/install/)。
- [Theano 安装指引](http://deeplearning.net/software/theano/install.html#install)。
- [CNTK 安装指引](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)。

你也可以考虑安装以下**可选依赖**：

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (如果你计划在 GPU 上运行 Keras，建议安装)。
- HDF5 和 [h5py](http://docs.h5py.org/en/latest/build.html) (如果你需要将 Keras 模型保存到磁盘，则需要这些)。
- [graphviz](https://graphviz.gitlab.io/download/) 和 [pydot](https://github.com/erocarrera/pydot) (用于[可视化工具](https://keras.io/zh/visualization/)绘制模型图)。

然后你就可以安装 Keras 本身了。有两种方法安装 Keras：

- **使用 PyPI 安装 Keras (推荐)：**

```sh
sudo pip install keras
```

如果你使用 virtualenv 虚拟环境, 你可以避免使用 sudo：

```sh
pip install keras
```

- **或者：使用 GitHub 源码安装 Keras：**

首先，使用 `git` 来克隆 Keras：

```sh
git clone https://github.com/keras-team/keras.git
```

然后，`cd` 到 Keras 目录并且运行安装命令：

```sh
cd keras
sudo python setup.py install
```



**配置你的 Keras 后端:**

默认情况下，Keras 将使用 TensorFlow 作为其张量操作库。请[跟随这些指引](https://keras.io/zh/backend/)来配置其他 Keras 后端。



**技术支持:**

你可以提出问题并参与开发讨论：

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)。
- [Keras Slack channel](https://kerasteam.slack.com)。 使用 [这个链接](https://keras-slack-autojoin.herokuapp.com/) 向该频道请求邀请函。
- 或者加入 Keras 深度学习交流群，协助文档的翻译工作，群号为 951623081。

你也可以在 [GitHub issues](https://github.com/keras-team/keras/issues) 中发布**漏洞报告和新功能请求**（仅限于此）。注意请先阅读[规范文档](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)。

------------------


## 1.2 快速开始

Keras 的核心数据结构是 __model__，一种组织网络层的方式。

- 最简单的模型是 [Sequential 顺序模型](/getting-started/sequential-model-guide)，它由多个网络层**线性堆叠**。
- 对于更复杂的结构，你应该使用 [Keras 函数式 API](/getting-started/functional-api-guide)，它允许构建任意的神经网络图。



1. 实例化`Sequential` 模型

```python
from keras.models import Sequential

model = Sequential()
```

2. 使用 `.add()` 来堆叠模型：

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

3. 完成了模型的构建后, 可以使用 `.compile()` 来配置学习过程：

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

3. 1 如果需要，你还可以进一步地配置你的优化器。Keras 的核心原则是使事情变得相当简单，同时又允许用户在需要的时候能够进行完全的控制（终极的控制是源代码的易扩展性）。

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

4. 模型拟合：批量地在训练数据上进行迭代：

```python
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

4. 1 或者，可以分批次的将数据提供给模型，防止内存溢出，（使用迭代器逐批次生成数据）

```python
model.fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

5. 评估模型性能：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

6. 对新的数据生成预测：

```python
classes = model.predict(x_test, batch_size=128)
```

有关 Keras 更深入的教程，请查看：

- [开始使用 Sequential 模型](/getting-started/sequential-model-guide)
- [开始使用函数式 API](/getting-started/functional-api-guide)
- 在代码仓库的 [examples 目录](https://github.com/keras-team/keras/tree/master/examples)中，你会找到更多高级模型：基于记忆网络的问答系统、基于栈式 LSTM 的文本生成等等。



## 1.3 Keras 常用技巧

- [如何引用 Keras?](#how-should-i-cite-keras)
- [如何在 GPU 上运行 Keras?](#how-can-i-run-keras-on-gpu)
- [如何在多 GPU 上运行 Keras 模型？](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["sample", "batch", "epoch" 分别是什么？](#what-does-sample-batch-epoch-mean)
- [如何保存 Keras 模型？](#how-can-i-save-a-keras-model)
- [为什么训练集误差比测试集的误差高很多？](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [如何获取中间层的输出？](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [如何用 Keras 处理超过内存的数据集？](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [在验证集的误差不再下降时，如何中断训练？](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [验证集划分是如何计算的？](#how-is-the-validation-split-computed)
- [在训练过程中数据是否会混洗？](#is-the-data-shuffled-during-training)
- [如何在每个 epoch 后记录训练集和验证集的误差和准确率？](#how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch)
- [如何「冻结」网络层？](#how-can-i-freeze-keras-layers)
- [如何使用状态 RNNs (stateful RNNs)?](#how-can-i-use-stateful-rnns)
- [如何从 Sequential 模型中移除一个层？](#how-can-i-remove-a-layer-from-a-sequential-model)
- [如何在 Keras 中使用预训练的模型？](#how-can-i-use-pre-trained-models-in-keras)
- [如何在 Keras 中使用 HDF5 输入？](#how-can-i-use-hdf5-inputs-with-keras)
- [Keras 配置文件保存在哪里？](#where-is-the-keras-configuration-file-stored)
- [如何在 Keras 开发过程中获取可复现的结果？](#how-can-i-obtain-reproducible-results-using-keras-during-development)
- [如何在 Keras 中安装 HDF5 或 h5py 来保存我的模型？](#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)

------

<span id="how-should-i-cite-keras"></span>

### 如何引用 Keras? 

如果 Keras 有助于您的研究，请在你的出版物中引用它。以下是 BibTeX 条目引用的示例：

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/keras-team/keras}},
}
```

------

<span id="how-can-i-run-keras-on-gpu"></span>

### 如何在 GPU 上运行 Keras?

如果你以 TensorFlow 或 CNTK 后端运行，只要检测到任何可用的 GPU，那么代码将自动在 GPU 上运行。

注意看终端输出，提示系统检测到了GPU 才说明GPU被使用，一般输出如下：

```python
2019-06-25 20:20:34.295275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8095
pciBusID: 0000:01:00.0
totalMemory: 7.92GiB freeMemory: 7.52GiB
2019-06-25 20:20:34.295289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-06-25 20:20:34.484338: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-06-25 20:20:34.484387: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-06-25 20:20:34.484393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-06-25 20:20:34.484550: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7253 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)

```

或者使用 `nvidia-smi`命令查看GPU内存占用，一般GPU被使用内存会几乎被占满。

如果GPU未被使用，则需要重启任务或重启电脑，或检查驱动或系统环境是否运行正确。



如果你以 Theano 后端运行，则可以使用以下方法之一：

**方法 1**: 使用 Theano flags。

```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

"gpu" 可能需要根据你的设备标识符（例如gpu0，gpu1等）进行更改。

**方法 2**: 创建 `.theanorc`: [指导教程](http://deeplearning.net/software/theano/library/config.html)

**方法 3**: 在代码的开头手动设置 `theano.config.device`, `theano.config.floatX`：

```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

------

<span id="how-can-i-run-a-keras-model-on-multiple-gpus"></span>

### 如何在多 GPU 上运行 Keras 模型?

我们建议使用 TensorFlow 后端来执行这项任务。有两种方法可在多个 GPU 上运行单个模型：**数据并行**和**设备并行**。

在大多数情况下，你最需要的是数据并行。

**数据并行**

数据并行包括在每个设备上复制一次目标模型，并使用每个模型副本处理不同部分的输入数据。Keras 有一个内置的实用函数 `keras.utils.multi_gpu_model`，它可以生成任何模型的数据并行版本，在多达 **8 个 GPU 上实现准线性加速**。

有关更多信息，请参阅 [multi_gpu_model](/utils/#multi_gpu_model) 的文档。这里是一个快速的例子：

```python
from keras.utils import multi_gpu_model

# 将 `model` 复制到 8 个 GPU 上。
# 假定你的机器有 8 个可用的 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 这个 `fit` 调用将分布在 8 个 GPU 上。
# 数据并行： batch size 为 256时，256个样本将分成8份，同时喂给8个GPU, 8个GPU上的模型参数同步更新？每个 GPU 将处理 32 个样本。
parallel_model.fit(x, y, epochs=20, batch_size=256)
```



**设备并行:**

设备并行性包括在不同设备上运行同一模型的不同部分。对于具有并行体系结构的模型，例如有两个分支的模型，这种方式很合适。

这种并行可以通过使用 TensorFlow device scopes 来实现。这里是一个简单的例子：

```python
# 模型中共享的 LSTM 用于并行编码两个不同的序列
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 在一个 GPU 上处理第一个序列
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 在另一个 GPU上 处理下一个序列
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# 在 CPU 上连接结果
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

------

<span id="what-does-sample-batch-epoch-mean"></span>

### "sample", "batch", "epoch" 分别是什么？

为了正确地使用 Keras，以下是必须了解和理解的一些常见定义：

- **Sample**: 样本，数据集中的一个元素，一条数据。
  - *例1:* 在卷积神经网络中，一张图像是一个样本。
  - *例2:* 在语音识别模型中，一段音频是一个样本。
- **Batch**: 批，含有 *N* 个样本的集合。每一个 batch 的样本都是独立并行处理的。在训练时，一个 batch 的结果只会用来更新一次模型。
  - 一个 **batch** 的样本通常比单个输入更接近于总体输入数据的分布，batch 越大就越近似。然而，每个 batch 将花费更长的时间来处理，并且仍然只更新模型一次。在推理（评估/预测）时，建议条件允许的情况下选择一个尽可能大的 batch，（因为较大的 batch 通常评估/预测的速度会更快）。 
- **Epoch**: 轮次，通常被定义为 「在整个数据集上的一轮迭代」，用于训练的不同的阶段，这有利于记录和定期评估。
  - 当在 Keras 模型的 `fit` 方法中使用 `validation_data` 或 `validation_split` 时，评估将在每个 **epoch** 结束时运行。
  - 在 Keras 中，可以添加专门的用于在 epoch 结束时运行的 [callbacks 回调](/callbacks/)。例如学习率变化和模型检查点（保存）。

------

<span id="how-can-i-save-a-keras-model"></span>

### 如何保存 Keras 模型？

#### 保存/加载整个模型（结构 + 权重 + 优化器状态）

*不建议使用 pickle 或 cPickle 来保存 Keras 模型。*

你可以使用 `model.save(filepath)` 将 Keras 模型保存到单个 HDF5 文件中，该文件将包含：

- 模型的结构，允许重新创建模型
- 模型的权重
- 训练配置项（损失函数，优化器）
- 优化器状态，允许准确地从你上次结束的地方继续训练。

你可以使用 `keras.models.load_model(filepath)` 重新实例化模型。`load_model` 还将负责使用保存的训练配置项来编译模型（除非模型从未编译过）。

例子：

```python
from keras.models import load_model

model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'
del model  # 删除现有模型

# 返回一个编译好的模型
# 与之前那个相同
model = load_model('my_model.h5')
```

另请参阅[如何安装 HDF5 或 h5py 以在 Keras 中保存我的模型？](#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)，查看有关如何安装 h5py 的说明。

#### 只保存/加载模型的结构

如果您只需要保存**模型的结构**，而非其权重或训练配置项，则可以执行以下操作：

```python
# 保存为 JSON
json_string = model.to_json()

# 保存为 YAML
yaml_string = model.to_yaml()
```

生成的 JSON/YAML 文件是人类可读的，如果需要还可以手动编辑。

你可以从这些数据建立一个新的模型：

```python
# 从 JSON 重建模型：
from keras.models import model_from_json
model = model_from_json(json_string)

# 从 YAML 重建模型：
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 只保存/加载模型的权重

如果您只需要 **模型的权重**，可以使用下面的代码以 HDF5 格式进行保存。

请注意，我们首先需要安装 HDF5 和 Python 库 h5py，它们不包含在 Keras 中。

```python
model.save_weights('my_model_weights.h5')

```

假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中：

```python
model.load_weights('my_model_weights.h5')

```

如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：

```python
model.load_weights('my_model_weights.h5', by_name=True)

```

例子：

```python
"""
假设原始模型如下所示：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 新模型
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 将被加载
model.add(Dense(10, name='new_dense'))  # 将不被加载

# 从第一个模型加载权重；只会影响第一层，dense_1
model.load_weights(fname, by_name=True)

```

#### 处理已保存模型中的自定义层（或其他自定义对象）

如果要加载的模型包含自定义层或其他自定义类或函数，则可以通过 `custom_objects` 参数将它们传递给加载机制：

```python
from keras.models import load_model
# 假设你的模型包含一个 AttentionLayer 类的实例
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})

```

或者，你可以使用 [自定义对象作用域](/utils/#customobjectscope)：

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')

```

自定义对象的处理与 `load_model`, `model_from_json`, `model_from_yaml` 的工作方式相同：

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})

```

------

<span id="why-is-the-training-loss-much-higher-than-the-testing-loss"></span>

### 为什么训练误差比测试误差高很多？

Keras 模型有两种模式：训练和测试。

- 正则化机制，如 Dropout 和 L1/L2 权重正则化，在测试时是关闭的。

- 此外，训练误差是每批训练数据的平均误差。由于你的模型是随着时间而变化的，一个 epoch 中的第一批数据的误差通常比最后一批的要高。另一方面，测试误差是模型在一个 epoch 训练完后计算的，因而误差较小。

------

<span id="how-can-i-obtain-the-output-of-an-intermediate-layer"></span>

### 如何获取中间层的输出？

一个简单的方法是创建一个新的 `Model` 来输出你所感兴趣的层：

```python
from keras.models import Model

model = ...  # 创建原始模型

layer_name = 'my_layer'
# 指定原始模型中的输入与中间层分别作为新模型的输入与输出
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)

intermediate_output = intermediate_layer_model.predict(data)

```

或者，你也可以构建一个 Keras 函数，该函数将在给定输入的情况下返回某个层的输出，例如：

```python
from keras import backend as K

# 以 Sequential 模型为例
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]

```

同样，你可以直接建立一个 Theano 或 TensorFlow 函数。

注意，如果你的模型在训练和测试阶段有不同的行为（例如，使用 `Dropout`, `BatchNormalization` 等），则需要将学习阶段标志传递给你的函数：

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 测试模式 = 0 时的输出
layer_output = get_3rd_layer_output([x, 0])[0]

# 测试模式 = 1 时的输出
layer_output = get_3rd_layer_output([x, 1])[0]

```

------

<span id="how-can-i-use-keras-with-datasets-that-dont-fit-in-memory"></span>

### 如何用 Keras 处理超过内存的数据集？

你可以使用 `model.train_on_batch(x，y)` 和 `model.test_on_batch(x，y)` 进行批量训练与测试。请参阅 [模型文档](/models/sequential)。

或者，你可以编写一个生成批处理训练数据的生成器，然后使用 `model.fit_generator(data_generator，steps_per_epoch，epochs)` 方法。

你可以在 [CIFAR10 example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) 中找到实践代码。`./examples/cifar10_cnn.md`

------

<span id="how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore"></span>

### 在验证集的误差不再下降时，如何中断训练？

你可以使用 `EarlyStopping` 回调：

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])

```

更多信息请查看 [callbacks 文档](/callbacks)。

------

<span id="how-is-the-validation-split-computed"></span>

### 验证集划分是如何计算的？

如果您将 `model.fit` 中的 `validation_split` 参数设置为 0.1，那么使用的验证数据将是最后 10％ 的数据。如果设置为 0.25，就是**最后 25% 的数据**。注意，在提取分割验证集之前，数据不会被混洗，因此验证集仅仅是传递的输入中最后一个 x％ 的样本。

 -- 所以要么提前混洗数据，要么自己手动划分训练集和验证集之后（手动划分也要包含混洗`shuffle`操作），再分别作为参数传入fit函数中。

所有 epoch 都使用相同的验证集（在同一个 `fit` 中调用）。

------

<span id="is-the-data-shuffled-during-training"></span>

### 在训练过程中数据是否会混洗？

是的，如果 `model.fit`中的 `shuffle`参数设置为 True（默认值），则训练数据将在每个 epoch 混洗。**验证集永远不会混洗**。

------

<span id="how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch"></span>

### 如何在每个 epoch 后记录训练集和验证集的误差和准确率？

`model.fit` 方法返回一个 `History` 回调，它具有包含连续误差的列表和其他度量`metrics`列表中指定的评价指标的 `history` 属性。

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)

```

------

<span id="how-can-i-freeze-keras-layers"></span>

### 如何「冻结」网络层？

「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。

您可以将 `trainable` 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

```python
frozen_layer = Dense(32, trainable=False)
```

另外，可以在实例化之后将**网络层的 `trainable` 属性**设置为 True 或 False。为了使之生效，在修改 `trainable` 属性之后，需要在模型上调用 `compile()`。这是一个例子：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 在下面的模型中，训练期间不会更新层的权重
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 使用这个模型，训练期间 `layer` 的权重将被更新
# (##这也会影响上面的模型，因为它使用了同一个网络层实例##)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # 这不会更新 `layer` 的权重
trainable_model.fit(data, labels)  # 这会更新 `layer` 的权重

```

------

<span id="how-can-i-use-stateful-rnns"></span>

### 如何使用有状态 RNN (stateful RNNs)?

使 RNN 具有状态意味着每批样品的状态将被重新用作下一批样品的初始状态。

当使用有状态 RNN 时，假定：

- 所有的批次都有相同数量的样本
- 如果 `x1` 和 `x2` 是连续批次的样本，则 `x2[i]` 是 `x1[i]` 的后续序列，对于每个 `i`。

要在 RNN 中使用状态，你需要:

- 通过将 `batch_size` 参数传递给模型的第一层来显式指定你正在使用的批大小。例如，对于 10 个时间步长的 32 样本的 batch，每个时间步长具有 16 个特征，`batch_size = 32`。 
- 在 RNN 层中设置 `stateful = True`。
- 在调用 `fit()` 时指定 `shuffle = False`。

重置累积状态：

- 使用 `model.reset_states()` 来重置模型中所有层的状态
- 使用 `layer.reset_states()` 来重置指定有状态 RNN 层的状态

例子：

```python
x  # 输入数据，尺寸为 (32, 21, 16)
# 将步长为 10 的序列输送到模型中

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 训练网络，根据给定的前 10 个时间步，来预测第 11 个时间步：
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 网络的状态已经改变。我们可以提供后续序列：
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# 重置 LSTM 层的状态：
model.reset_states()

# 另一种重置方法：
model.layers[0].reset_states()

```

请注意，`predict`, `fit`, `train_on_batch`, `predict_classes` 等方法*全部*都会更新模型中有状态层的状态。这使你不仅可以进行有状态的训练，还可以进行有状态的预测。

------

<span id="how-can-i-remove-a-layer-from-a-sequential-model"></span>

### 如何从 Sequential 模型中移除一个层？

你可以通过调用 `.pop()` 来**删除 `Sequential` 模型中最后添加的层**：

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"

```

------

<span id="how-can-i-use-pre-trained-models-in-keras"></span>

### 如何在 Keras 中使用预训练的模型？

我们提供了以下图像分类模型的代码和预训练的权重：

- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1
- DenseNet
- NASNet
- MobileNet v2

它们可以使用 `keras.applications` 模块进行导入：

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.mobilenet_v2 import MobileNetV2

model = VGG16(weights='imagenet', include_top=True)

```

**有关一些简单的用法示例，请参阅 [Applications 模块的文档](/applications)。**

有关**如何使用此类预训练的模型进行特征提取或微调的详细示例**，请参阅 [此博客文章](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)。

VGG16 模型也是以下几个 Keras 示例脚本的基础：

- [Style transfer](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

------

<span id="how-can-i-use-hdf5-inputs-with-keras"></span>

### 如何在 Keras 中使用 HDF5 输入？

你可以使用 `keras.utils.io_utils` 中的 `HDF5Matrix` 类。有关详细信息，请参阅 [HDF5Matrix文档](/utils/#hdf5matrix)。

你也可以直接使用 **HDF5 数据集**：

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)

```

------

<span id="where-is-the-keras-configuration-file-stored"></span>

### Keras 配置文件保存在哪里？

所有 Keras 数据存储的默认目录是：

```bash
$HOME/.keras/

```

注意，Windows 用户应该将 `$HOME` 替换为 `％USERPROFILE％`。如果 Keras 无法创建上述目录（例如，由于权限问题），则使用 `/tmp/.keras/` 作为备份。

Keras配置文件是存储在 `$HOME/.keras/keras.json` 中的 JSON 文件。默认的配置文件如下所示：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

```

它包含以下字段：

- 图像处理层和实用程序所使用的默认值图像数据格式（`channels_last` 或 `channels_first`）。
- 用于防止在某些操作中被零除的 `epsilon` 模糊因子。
- 默认浮点数据类型。
- 默认后端。详见 [backend 文档](/backend)。

同样，缓存的数据集文件（如使用 `get_file()` 下载的文件）默认存储在 `$HOME/.keras/datasets/` 中。

------

<span id="how-can-i-obtain-reproducible-results-using-keras-during-development"></span>

### 如何在 Keras 开发过程中获取可复现的结果？

在模型的开发过程中，能够在一次次的运行中获得可复现的结果，以确定性能的变化是来自模型还是数据集的变化，或者仅仅是一些新的随机样本点带来的结果，有时候是很有用处的。

1. 首先，你需要在程序启动之前将 `PYTHONHASHSEED` 环境变量设置为 0（不在程序本身内）。对于 Python 3.2.3 以上版本，它对于某些基于散列的操作具有可重现的行为是必要的（例如，集合和字典的 item 顺序，请参阅 [Python 文档](https://docs.python.org/3.7/using/cmdline.html#envvar-PYTHONHASHSEED)和 [issue #2280](https://github.com/keras-team/keras/issues/2280#issuecomment-306959926) 获取更多详细信息）。设置环境变量的一种方法是，在这样启动 python 时：

```bash
$ cat test_hash.py
print(hash("keras"))
$ python3 test_hash.py                  # 无法复现的 hash (Python 3.2.3+)
-8127205062320133199
$ python3 test_hash.py                  # 无法复现的 hash (Python 3.2.3+)
3204480642156461591
$ PYTHONHASHSEED=0 python3 test_hash.py # 可复现的 hash
4883664951434749476
$ PYTHONHASHSEED=0 python3 test_hash.py # 可复现的 hash
4883664951434749476

```

2. 此外，当使用 TensorFlow 后端并在 GPU 上运行时，某些操作具有非确定性输出，特别是 `tf.reduce_sum()`。这是因为 GPU 并行运行许多操作，因此并不总能保证执行顺序。由于浮点数的精度有限，即使添加几个数字，也可能会产生略有不同的结果，具体取决于添加它们的顺序。你可以尝试避免某些非确定性操作，但有些操作可能是由 TensorFlow 在计算梯度时自动创建的，因此在 CPU 上运行代码要简单得多。为此，你可以将 `CUDA_VISIBLE_DEVICES` 环境变量设置为空字符串，例如：

```bash
$ CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 python your_program.py

```

下面的代码片段提供了一个如何获得可复现结果的例子 - 针对 Python 3 环境的 TensorFlow 后端。

```python
import numpy as np
import tensorflow as tf
import random as rn

# 以下是 Numpy 在一个明确的初始状态生成固定随机数字所必需的。

np.random.seed(42)

# 以下是 Python 在一个明确的初始状态生成固定随机数字所必需的。

rn.seed(12345)

# 强制 TensorFlow 使用单线程。
# 多线程是结果不可复现的一个潜在因素。
# 更多详情，见: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
                              
from keras import backend as K

# `tf.set_random_seed()` 将会以 TensorFlow 为后端，
# 在一个明确的初始状态下生成固定随机数字。
# 更多详情，见: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 剩余代码 ...

```

------

<span id="how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras"></span>

### 如何在 Keras 中安装 HDF5 或 h5py 来保存我的模型？

为了将你的 Keras 模型保存为 HDF5 文件，例如通过 `keras.callbacks.ModelCheckpoint`，Keras 使用了 h5py Python 包。h5py 是 Keras 的依赖项，应默认被安装。在基于 Debian 的发行版本上，你需要再额外安装 `libhdf5`：

```
sudo apt-get install libhdf5-serial-dev
```

如果你不确定是否安装了 h5py，则可以打开 Python shell 并通过下面的命令加载模块

```
import h5py

```

如果模块导入没有错误，那么说明模块已经安装成功，否则你可以在 [http://docs.h5py.org/en/latest/build.html](http://docs.h5py.org/en/latest/build.html) 中找到详细的安装说明。

