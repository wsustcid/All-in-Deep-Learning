# sequence

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L16)</span>

### TimeseriesGenerator

```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

用于生成批量时序数据的实用工具类。

这个类以一系列由相等间隔以及一些时间序列参数（例如步长、历史长度等）汇集的数据点作为输入，以生成用于训练/验证的批次数据。

__参数__

- __data__: 可索引的生成器（例如列表或 Numpy 数组），包含连续数据点（时间步）。数据应该是 2D 的，且第 0 个轴为时间维度。
- __targets__: 对应于 `data` 的时间步的目标值。它应该与 `data` 的长度相同。
- __length__: 输出序列的长度（以时间步数表示）。
- __sampling_rate__: 序列内连续各个时间步之间的周期。对于周期 `r`, 时间步 `data[i]`, `data[i-r]`, ... `data[i - length]` 被用于生成样本序列。
- __stride__: 连续输出序列之间的周期. 对于周期 `s`, 连续输出样本将为 `data[i]`, `data[i+s]`, `data[i+2*s]` 等。
- __start_index__: 在 `start_index` 之前的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
- __end_index__: 在 `end_index` 之后的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
- __shuffle__: 是否打乱输出样本，还是按照时间顺序绘制它们。
- __reverse__: 布尔值: 如果 `true`, 每个输出样本中的时间步将按照时间倒序排列。
- __batch_size__: 每个批次中的时间序列样本数（可能除最后一个外）。

__返回__

一个 [Sequence](https://keras.io/zh/utils/#sequence) 实例。

__例子__

```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```

---

### pad_sequences


```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```

将多个序列截断或补齐为相同长度。

该函数将一个 `num_samples` 的序列（整数列表）转化为一个 2D Numpy 矩阵，其尺寸为 `(num_samples, num_timesteps)`。 `num_timesteps` 要么是给定的 `maxlen` 参数，要么是最长序列的长度。

比 `num_timesteps` 短的序列将在末端以 `value` 值补齐。

比 `num_timesteps` 长的序列将会被截断以满足所需要的长度。补齐或截断发生的位置分别由参数 `pading` 和 `truncating` 决定。

向前补齐为默认操作。

__参数__

- __sequences__: 列表的列表，每一个元素是一个序列。
- __maxlen__: 整数，所有序列的最大长度。
- __dtype__: 输出序列的类型。
要使用可变长度字符串填充序列，可以使用 `object`。
- __padding__: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
- __truncating__: 字符串，'pre' 或 'post' ，移除长度大于 `maxlen` 的序列的值，要么在序列前端截断，要么在后端。
- __value__: 浮点数，表示用来补齐的值。


__返回__

- __x__: Numpy 矩阵，尺寸为 `(len(sequences), maxlen)`。

__异常__

- ValueError: 如果截断或补齐的值无效，或者序列条目的形状无效。

---

### skipgrams


```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```

生成 skipgram 词对。

该函数将一个单词索引序列（整数列表）转化为以下形式的单词元组：

- （单词, 同窗口的单词），标签为 1（正样本）。
- （单词, 来自词汇表的随机单词），标签为 0（负样本）。

若要了解更多和 Skipgram 有关的知识，请参阅这份由 Mikolov 等人发表的经典论文： [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

__参数__

- __sequence__: 一个编码为单词索引（整数）列表的词序列（句子）。如果使用一个 `sampling_table`，词索引应该以一个相关数据集的词的排名匹配（例如，10 将会编码为第 10 个最长出现的词）。注意词汇表中的索引 0 是非单词，将被跳过。
- __vocabulary_size__: 整数，最大可能词索引 + 1
- __window_size__: 整数，采样窗口大小（技术上是半个窗口）。词 `w_i` 的窗口是 `[i - window_size, i + window_size+1]`。
- __negative_samples__: 大于等于 0 的浮点数。0 表示非负（即随机）采样。1 表示与正样本数相同。
- __shuffle__: 是否在返回之前将这些词语打乱。
- __categorical__: 布尔值。如果 False，标签将为整数（例如 `[0, 1, 1 .. ]`），如果 True，标签将为分类，例如 `[[1,0],[0,1],[0,1] .. ]`。
- __sampling_table__: 尺寸为 `vocabulary_size` 的 1D 数组，其中第 i 项编码了排名为 i 的词的采样概率。
- __seed__: 随机种子。
  

__返回__

couples, labels: 其中 `couples` 是整数对，`labels` 是 0 或 1。

__注意__

按照惯例，词汇表中的索引 0 是非单词，将被跳过。

---

### make_sampling_table


```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```


生成一个基于单词的概率采样表。

用来生成 `skipgrams` 的 `sampling_table` 参数。`sampling_table[i]` 是数据集中第 i 个最常见词的采样概率（出于平衡考虑，出现更频繁的词应该被更少地采样）。

采样概率根据 word2vec 中使用的采样分布生成：

```python
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

我们假设单词频率遵循 Zipf 定律（s=1），来导出 frequency(rank) 的数值近似：

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`，其中 `gamma` 为 Euler-Mascheroni 常量。

__参数__

- __size__: 整数，可能采样的单词数量。
- __sampling_factor__: word2vec 公式中的采样因子。

__返回__

一个长度为 `size` 大小的 1D Numpy 数组，其中第 i 项是排名为 i 的单词的采样概率。



# text

### Text Preprocessing

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/text.py#L138)</span>

### Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None, 
                                   filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	', 
                                   lower=True, 
                                   split=' ', 
                                   char_level=False, 
                                   oov_token=None, 
                                   document_count=0)
```

文本标记实用类。

该类允许使用两种方法向量化一个文本语料库：
将每个文本转化为一个整数序列（每个整数都是词典中标记的索引）；
或者将其转化为一个向量，其中每个标记的系数可以是二进制值、词频、TF-IDF权重等。

__参数__

- __num_words__: 需要保留的最大词数，基于词频。只有最常出现的 `num_words` 词会被保留。
- __filters__: 一个字符串，其中每个元素是一个将从文本中过滤掉的字符。默认值是所有标点符号，加上制表符和换行符，减去 `'` 字符。
- __lower__: 布尔值。是否将文本转换为小写。
- __split__: 字符串。按该字符串切割文本。
- __char_level__: 如果为 True，则每个字符都将被视为标记。
- __oov_token__: 如果给出，它将被添加到 word_index 中，并用于在 `text_to_sequence` 调用期间替换词汇表外的单词。

默认情况下，删除所有标点符号，将文本转换为空格分隔的单词序列（单词可能包含 `'` 字符）。
这些序列然后被分割成标记列表。然后它们将被索引或向量化。

`0` 是不会被分配给任何单词的保留索引。

------

### hashing_trick

```python
keras.preprocessing.text.hashing_trick(text, n,
                                       hash_function=None, 
                                       filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	', lower=True, 
                                       split=' ')
```

将文本转换为固定大小散列空间中的索引序列。

__参数__

- __text__: 输入文本（字符串）。
- __n__: 散列空间维度。
- __hash_function__: 默认为 python 散列函数，可以是 'md5' 或任意接受输入字符串并返回整数的函数。注意 'hash' 不是稳定的散列函数，所以它在不同的运行中不一致，而 'md5' 是一个稳定的散列函数。
- __filters__: 要过滤的字符列表（或连接），如标点符号。默认：`!"#$%&()*+,-./:;<=>?@[\]^_{|}~`，包含基本标点符号，制表符和换行符。
- __lower__: 布尔值。是否将文本转换为小写。
- __split__: 字符串。按该字符串切割文本。

__返回__

整数词索引列表（唯一性无法保证）。

`0` 是不会被分配给任何单词的保留索引。

由于哈希函数可能发生冲突，可能会将两个或更多字分配给同一索引。
碰撞的[概率](https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)与散列空间的维度和不同对象的数量有关。

------

### one_hot

```python
keras.preprocessing.text.one_hot(text, n, 
                                 filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
                                 lower=True, 
                                 split=' ')
```

One-hot 将文本编码为大小为 n 的单词索引列表。

这是 `hashing_trick` 函数的一个封装，
使用 `hash` 作为散列函数；单词索引映射无保证唯一性。

__参数__

- __text__: 输入文本（字符串）。
- __n__: 整数。词汇表尺寸。
- __filters__: 要过滤的字符列表（或连接），如标点符号。默认：`!"#$%&()*+,-./:;<=>?@[\]^_{|}~`，包含基本标点符号，制表符和换行符。
- __lower__: 布尔值。是否将文本转换为小写。
- __split__: 字符串。按该字符串切割文本。

__返回__

[1, n] 之间的整数列表。每个整数编码一个词（唯一性无法保证）。

------

### text_to_word_sequence

```python
keras.preprocessing.text.text_to_word_sequence(text, 
                                               filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~	', 
                                               lower=True, 
                                               split=' ')
```

将文本转换为单词（或标记）的序列。

__参数__

- __text__: 输入文本（字符串）。
- __filters__: 要过滤的字符列表（或连接），如标点符号。默认：`!"#$%&()*+,-./:;<=>?@[\]^_{|}~`，包含基本标点符号，制表符和换行符。
- __lower__: 布尔值。是否将文本转换为小写。
- __split__: 字符串。按该字符串切割文本。

__返回__

词或标记的列表。



# image

# 图像预处理

## ImageDataGenerator 类

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,  
                                             samplewise_center=False, 
                                             featurewise_std_normalization=False, 
                                             samplewise_std_normalization=False, 
                                             zca_whitening=False, 
                                             zca_epsilon=1e-06, 
                                             rotation_range=0, 
                                             width_shift_range=0.0, 
                                             height_shift_range=0.0, 
                                             brightness_range=None, 
                                             shear_range=0.0, 
                                             zoom_range=0.0, 
                                             channel_shift_range=0.0, 
                                             fill_mode='nearest', 
                                             cval=0.0, 
                                             horizontal_flip=False, 
                                             vertical_flip=False, 
                                             rescale=None, 
                                             preprocessing_function=None, 
                                             data_format=None, 
                                             validation_split=0.0, 
                                             dtype=None)
```

通过实时数据增强按批次循环生成张量图像数据。

__参数__

- __featurewise_center__: 布尔值。将输入数据的均值设置为 0，逐特征进行。
- __samplewise_center__: 布尔值。将每个样本的均值设置为 0。
- __featurewise_std_normalization__: Boolean. 布尔值。将输入除以数据标准差，逐特征进行。
- __samplewise_std_normalization__: 布尔值。将每个输入除以其标准差。
- __zca_epsilon__: ZCA 白化的 epsilon 值，默认为 1e-6。
- __zca_whitening__: 布尔值。是否应用 ZCA 白化。
- __rotation_range__: 整数。随机旋转的度数范围。
- __width_shift_range__: 浮点数、一维数组或整数
  - float: 
    - 如果 <1，则是除以总宽度的值（按比例移动），
    - 或者如果 >=1，则为像素值。
  - 1-D 数组: 随机选取此数组中的元素。
  - int: 随机选取 `(-width_shift_range, +width_shift_range)`区间的整数
    - `width_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `width_shift_range=[-1, 0, +1]` 相同；
    - 而 `width_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
- __height_shift_range__: 浮点数、一维数组或整数
  - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
  - 1-D array-like: 数组中的随机元素。
  - int: 来自间隔 `(-height_shift_range, +height_shift_range)` 之间的整数个像素。
  - `height_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `height_shift_range=[-1, 0, +1]` 相同；而 `height_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
- __shear_range__: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
- __zoom_range__: 浮点数 或 `[lower, upper]`。随机缩放范围。
  - 如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
- __channel_shift_range__: 浮点数。随机通道转换的范围。
- __fill_mode__: 输入边界以外的点根据给定的模式填充，{"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。
  - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
  - 'nearest': aaaaaaaa|abcd|dddddddd
  - 'reflect': abcddcba|abcd|dcbaabcd
  - 'wrap': abcdabcd|abcd|abcdabcd
- __cval__: 浮点数或整数。当 `fill_mode = "constant"` 时，用于边界之外的点的值。
- __horizontal_flip__: 布尔值。随机水平翻转。
- __vertical_flip__: 布尔值。随机垂直翻转。
- __rescale__: 重缩放因子。默认为 None。
  - 如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
- __preprocessing_function__: 应用于每个输入的函数。这个函数会在任何其他改变之前运行。
  - 这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。
- __data_format__: 图像数据格式，{"channels_first", "channels_last"} 之一。
  - "channels_last" 模式表示图像输入尺寸应该为 `(samples, height, width, channels)`，
  - "channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。
  - 默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
- __validation_split__: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。
- __dtype__: 生成数组使用的数据类型。

### ImageDataGenerator 类方法

#### apply_transform

```python
apply_transform(x, transform_parameters)
```

根据给定的参数将变换应用于图像。

__参数__

- __x__: 3D 张量，单张图像。
- __transform_parameters__: 字符串 - 参数 对表示的字典，用于描述转换。目前，使用字典中的以下参数：
  - 'theta': 浮点数。旋转角度（度）。
  - 'tx': 浮点数。在 x 方向上移动。
  - 'ty': 浮点数。在 y 方向上移动。
  - shear': 浮点数。剪切角度（度）。
  - 'zx': 浮点数。放大 x 方向。
  - 'zy': 浮点数。放大 y 方向。
  - 'flip_horizontal': 布尔 值。水平翻转。
  - 'flip_vertical': 布尔值。垂直翻转。
  - 'channel_shift_intencity': 浮点数。频道转换强度。
  - 'brightness': 浮点数。亮度转换强度。

__返回__

输入的转换后版本（相同尺寸）。

------

#### fit

```python
fit(x, augment=False, rounds=1, seed=None)
```

将数据生成器用于某些示例数据。

它基于一组样本数据，计算与数据转换相关的内部数据统计。

当且仅当 `featurewise_center` 或 `featurewise_std_normalization` 或 `zca_whitening` 设置为 True 时才需要。

__参数__

- __x__: 样本数据。秩应该为 4。对于灰度数据，通道轴的值应该为 1；对于 RGB 数据，值应该为 3。
- __augment__: 布尔值（默认为 False）。是否使用随机样本扩张。
- __rounds__: 整数（默认为 1）。如果数据数据增强（augment=True），表明在数据上进行多少次增强。
- __seed__: 整数（默认 None）。随机种子。

------

#### flow

```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```

采集数据和标签数组，生成批量增强数据。

__参数__

- __x__: 输入数据。秩为 4 的 Numpy 矩阵或元组。如果是元组，第一个元素应该包含图像，第二个元素是另一个 Numpy 数组或一列 Numpy 数组，它们不经过任何修改就传递给输出。可用于将模型杂项数据与图像一起输入。对于灰度数据，图像数组的通道轴的值应该为 1，而对于 RGB 数据，其值应该为 3。
- __y__: 标签。
- __batch_size__: 整数 (默认为 32)。
- __shuffle__: 布尔值 (默认为 True)。
- __sample_weight__: 样本权重。
- __seed__: 整数（默认为 None）。
- __save_to_dir__: None 或 字符串（默认为 None）。这使您可以选择指定要保存的正在生成的增强图片的目录（用于可视化您正在执行的操作）。
- __save_prefix__: 字符串（默认 `''`）。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __subset__: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。

__返回__

一个生成元组 `(x, y)` 的 `Iterator`，其中 `x` 是图像数据的 Numpy 数组（在单张图像输入时），或 Numpy 数组列表（在额外多个输入时），`y` 是对应的标签的 Numpy 数组。如果 'sample_weight' 不是 None，生成的元组形式为 `(x, y, sample_weight)`。如果 `y` 是 None, 只有 Numpy 数组 `x` 被返回。

__例子__

使用 `.flow(x, y)` 的例子：

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 计算特征归一化所需的数量
# （如果应用 ZCA 白化，将计算标准差，均值，主成分）
datagen.fit(x_train)

# 使用实时数据增益的批数据对模型进行拟合：
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 这里有一个更 「手动」的例子
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # 我们需要手动打破循环，
            # 因为生成器会无限循环
            break
```

#### flow_from_dataframe

```python
flow_from_dataframe(dataframe, directory, x_col='filename', y_col='class', has_ext=True, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest')
```

输入 dataframe 和目录的路径，并生成批量的增强/标准化的数据。

这里有一个简单的教程： [http://bit.ly/keras_flow_from_dataframe](http://bit.ly/keras_flow_from_dataframe)

__参数__

- __dataframe__: Pandas dataframe，一列为图像的文件名，另一列为图像的类别，
  或者是可以作为原始目标数据多个列。
- __directory__: 字符串，目标目录的路径，其中包含在 dataframe 中映射的所有图像。
- __x_col__: 字符串，dataframe 中包含目标图像文件夹的目录的列。
- __y_col__: 字符串或字符串列表，dataframe 中将作为目标数据的列。
- __has_ext__: 布尔值，如果 dataframe[x_col] 中的文件名具有扩展名则为 True，否则为 False。
- __target_size__: 整数元组 `(height, width)`，默认为 `(256, 256)`。
      ​    所有找到的图都会调整到这个维度。
- __color_mode__: "grayscale", "rbg" 之一。默认："rgb"。
      ​    图像是否转换为 1 个或 3 个颜色通道。
- __classes__: 可选的类别列表
  (例如， `['dogs', 'cats']`)。默认：None。
   如未提供，类比列表将自动从 y_col 中推理出来，y_col 将会被映射为类别索引）。
   包含从类名到类索引的映射的字典可以通过属性 `class_indices` 获得。
- __class_mode__: "categorical", "binary", "sparse", "input", "other" or None 之一。
  默认："categorical"。决定返回标签数组的类型：
  - `"categorical"` 将是 2D one-hot 编码标签，
  - `"binary"` 将是 1D 二进制标签，
  - `"sparse"` 将是 1D 整数标签，
  - `"input"` 将是与输入图像相同的图像（主要用于与自动编码器一起使用），
  - `"other"` 将是 y_col 数据的 numpy 数组，
  - None, 不返回任何标签（生成器只会产生批量的图像数据，这对使用 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。
- __batch_size__: 批量数据的尺寸（默认：32）。
- __shuffle__: 是否混洗数据（默认：True）
- __seed__: 可选的混洗和转换的随即种子。
- __save_to_dir__: None 或 str (默认: None).
      ​    这允许你可选地指定要保存正在生成的增强图片的目录（用于可视化您正在执行的操作）。
- __save_prefix__: 字符串。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __follow_links__: 是否跟随类子目录中的符号链接（默认：False）。
- __subset__: 数据子集 (`"training"` 或 `"validation"`)，如果在 `ImageDataGenerator` 中设置了 `validation_split`。
- __interpolation__: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。
  支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。
  如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。
  如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。
  默认情况下，使用 `"nearest"`。

__Returns__

一个生成 `(x, y)` 元组的 DataFrameIterator，
其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)` 
的图像样本的 numpy 数组，`y` 是对应的标签的 numpy 数组。

------

#### flow_from_directory

```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```

__参数__

- __directory__: 目标目录的路径。
  - 每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。更多细节，详见 [此脚本](https://gist.github.com/fchollet/%20%20%20%20%20%20%20%200830affa1f7f19fd47b06d4cf89ed44d)。
- __target_size__: 整数元组 `(height, width)`，默认：`(256, 256)`。所有的图像将被调整到的尺寸。
- __color_mode__: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
- __classes__: 可选的类的子目录列表（例如 `['dogs', 'cats']`）。默认：None。
  - 如果未提供，类的列表将自动从 `directory` 下的 子目录名称/结构 中推断出来，其中每个子目录都将被作为不同的类（类名将按字典序映射到标签的索引）。
  - 包含从类名到类索引的映射的字典可以通过 `class_indices` 属性获得。
- __class_mode__:  "categorical", "binary", "sparse", "input" 或 None 之一。默认："categorical"。决定返回的标签数组的类型：
  - "categorical" 将是 2D one-hot 编码标签，
  - "binary" 将是 1D 二进制标签，"sparse" 将是 1D 整数标签，
  - "input" 将是与输入图像相同的图像（主要用于自动编码器）。
  - 如果为 None，不返回标签（生成器将只产生批量的图像数据，对于 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。请注意，如果 `class_mode` 为 None，那么数据仍然需要驻留在 `directory` 的子目录中才能正常工作。
- __batch_size__: 一批数据的大小（默认 32）。
- __shuffle__: 是否混洗数据（默认 True）。
- __seed__: 可选随机种子，用于混洗和转换。
- __save_to_dir__: None 或 字符串（默认 None）。这使你可以最佳地指定正在生成的增强图片要保存的目录（用于可视化你在做什么）。
- __save_prefix__: 字符串。 保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __follow_links__: 是否跟踪类子目录中的符号链接（默认为 False）。
- __subset__: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。
- __interpolation__: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。
  - 如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。
  - 如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。
  - 默认情况下，使用 `"nearest"`。

__返回__

一个生成 `(x, y)` 元组的 `DirectoryIterator`，其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)`的图像的 Numpy 数组，`y` 是对应标签的 Numpy 数组。

**例子：**

```python
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory('data/validation',
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

model.fit_generator(train_generator,
                    steps_per_epoch=2000,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=800)
```

**例子: 同时转换图像和蒙版 (mask)** 

```python
# 创建两个相同参数的实例
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 为 fit 和 flow 函数提供相同的种子和关键字参数
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# 将生成器组合成一个产生图像和蒙版（mask）的生成器
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)

```

------

#### get_random_transform

```python
get_random_transform(img_shape, seed=None)

```

为转换生成随机参数。

__参数__

- __seed__: 随机种子
- __img_shape__: 整数元组。被转换的图像的尺寸。

__返回__

包含随机选择的描述变换的参数的字典。

------

#### random_transform

```python
random_transform(x, seed=None)

```

将随机变换应用于图像。

__参数__

- __x__: 3D 张量，单张图像。
- __seed__: 随机种子。

__返回__

输入的随机转换版本（相同形状）。

------

#### standardize

```python
standardize(x)

```

将标准化配置应用于一批输入。

__参数__

- __x__: 需要标准化的一批输入。

__返回__

标准化后的输入。



# 附录

## 白化

白化的目的是去除输入数据的冗余信息。假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性，所以用于训练时输入是冗余的；白化的目的就是降低输入的冗余性。

输入数据集X，经过白化处理后，新的数据X'满足两个性质：

- 特征之间相关性较低；
- 所有特征具有相同的方差。
      

PCA给我们的印象是一般用于降维操作。然而其实PCA如果不降维，而是仅仅使用PCA求出特征向量，然后把数据X映射到新的特征空间，这样的一个映射过程，其实就是满足了我们白化的第一个性质：除去特征之间的相关性。因此白化算法的实现过程，第一步操作就是PCA，求出新特征空间中X的新坐标，然后再对新的坐标进行方差归一化操作。

PCA的数学原理

<https://zhuanlan.zhihu.com/p/21580949>

**算法概述：**
白化分为PCA白化、ZCA白化，下面主要讲解算法实现。这部分主要是学了UFLDL的深度学习《白化》教程：http://ufldl.stanford.edu/wiki/index.php/%E7%99%BD%E5%8C%96，算法实现步骤如下：

1. 首先是PCA预处理

   ​                   <img src=https://img-blog.csdn.net/20160312120157759 > <img src= https://img-blog.csdn.net/20160312120205309 >    
   上面图片，左图表示原始数据X，然后我们通过协方差矩阵可以求得特征向量u1、u2，然后把每个数据点，投影到这两个新的特征向量，得到进行坐标如下：

   ![img](https://img-blog.csdn.net/20160312120214088)

   这就是所谓的pca处理。

2. PCA白化
   所谓的pca白化是指对上面的pca的新坐标X’,每一维的特征做一个标准差归一化处理。因为从上面我们看到在新的坐标空间中，(x1,x2)两个坐标轴方向的数据明显标准差不同，因此我们接着要对新的每一维坐标做一个标注差归一化处理：
   $$
   X''_{PCAWhite} = \frac{X'}{std(X')}
   $$
   也可以采用下面的公式：
   $$
   X''_{PCAWhite} = \frac{X'}{\sqrt{\lambda_i + \varepsilon}}
   $$
   其中`X'`为经过PCA处理的新PCA坐标空间,然后λi就是第i维特征对应的特征值（前面pca得到的特征值），ε是为了避免除数为0。



3. ZCA白化
   ZCA白化是在PCA白化的基础上，又进行处理的一个操作。具体的实现是把上面PCA白化的结果，又变换到原来坐标系下的坐标：
   $$
   Y_{ZCAWhite} = U * Y_{PCAwhite}
   $$

具体源码实现如下：

```python
def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
    U,S,V = np.linalg.svd(sigma) #奇异分解
    epsilon = 0.1                #白化的时候，防止除数为0
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #计算zca白化矩阵
    return np.dot(ZCAMatrix, inputs)   #白化变换



```



参考文献：
1、http://ufldl.stanford.edu/wiki/index.php/%E7%99%BD%E5%8C%96