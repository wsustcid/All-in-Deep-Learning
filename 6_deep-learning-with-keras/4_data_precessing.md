# 1. 数据生成器

## Image

### ImageDataGenerator 类

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

<https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/>

<https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad>

brightness_range=[0.5,1.5]

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
  - 每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。更多细节，详见 [此脚本](<https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d>)。
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





## sequence

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



## text

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







------

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302)</span>

## utils.Sequence

`Sequence` 是进行多进程处理的更安全的方法。这种结构保证网络在每个时期每个样本只训练一次，这与生成器不同。

<https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>

<https://github.com/shervinea/enzynet/blob/master/enzynet/volume.py#L24>

<https://github.com/shervinea/enzynet/blob/master/scripts/architecture/enzynet_adapted.py>

```python
keras.utils.Sequence()
```

用于拟合数据序列的基对象，例如一个数据集。

每一个 `Sequence` 必须实现 `__getitem__` 和 `__len__` 方法。
如果你想在迭代之间修改你的数据集，你可以实现 `on_epoch_end`。
`__getitem__` 方法应该范围一个完整的批次。

__注意__

`Sequence` 是进行多进程处理的更安全的方法。这种结构保证网络在每个时期每个样本只训练一次，这与生成器不同。

<https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly>

__例子__

```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# 这里，`x_set` 是图像的路径列表
# 以及 `y_set` 是对应的类别

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```



### Example

In this blog post, we are going to show you how to **generate your dataset on multiple cores in real time** and **feed it right away** to your **deep learning model**.

The framework used in this tutorial is the one provided by Python's high-level package *Keras*, which can be used on top of a GPU installation of either *TensorFlow* or *Theano*.

**Notations**

Before getting started, let's go through a few organizational tips that are particularly useful when dealing with large datasets.

Let `ID` be the Python string that identifies a given sample of the dataset. A good way to keep track of samples and their labels is to adopt the following framework:

1. Create a dictionary called `partition` where you gather:
   - in `partition['train']` a list of training IDs
   - in `partition['validation']` a list of validation IDs
2. Create a dictionary called `labels` where for each `ID` of the dataset, the associated label is given by `labels[ID]`

For example, let's say that our training set contains `id-1`, `id-2` and `id-3` with respective labels `0`, `1` and `2`, with a validation set containing `id-4` with label `1`. In that case, the Python variables `partition` and `labels` look like

```python
>>> partition
{'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
```

and

```python
>>> labels
{'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}
```

Also, for the sake of **modularity**, we will write Keras code and customized classes in separate files, so that your folder looks like

```python
folder/
├── my_classes.py
├── keras_script.py
└── data/
```

where `data/` is assumed to be the folder containing your dataset.

Finally, it is good to note that the code in this tutorial is aimed at being **general** and **minimal**, so that you can easily adapt it for your own dataset.

**Data generator**

Now, let's go through the details of how to set the Python class `DataGenerator`, which will be used for real-time data feeding to your Keras model.

First, let's write the initialization function of the class. We make the latter inherit the properties of `keras.utils.Sequence` so that we can leverage nice functionalities such as *multiprocessing*.

```python
def __init__(self, list_IDs, labels, batch_size=32, 
             dim=(32,32,32), n_channels=1,
             n_classes=10, shuffle=True):
    'Initialization'
    self.dim = dim
    self.batch_size = batch_size
    self.labels = labels
    self.list_IDs = list_IDs
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()
```

We put as arguments relevant information about the data, such as dimension sizes (e.g. a volume of length 32 will have `dim=(32,32,32)`), number of channels, number of classes, batch size, or decide whether we want to shuffle our data at generation. We also store important information such as labels and the list of IDs that we wish to generate at each pass.

**Here, the method `on_epoch_end` is triggered once at the very beginning as well as at the end of each epoch.** If the `shuffle` parameter is set to `True`, we will get a new order of exploration at each pass (or just keep a linear exploration scheme otherwise).

```python
def on_epoch_end(self):
  'Updates indexes after each epoch'
  self.indexes = np.arange(len(self.list_IDs))
  if self.shuffle == True:
      np.random.shuffle(self.indexes)
```

Shuffling the order in which examples are fed to the classifier is helpful so that batches between epochs do not look alike. Doing so will eventually make our model more robust.

Another method that is core to the generation process is the one that achieves the most crucial job: producing batches of data. **The private method** in charge of this task is called `__data_generation` and takes as argument the list of IDs of the target batch.

```python
def __data_generation(self, list_IDs_temp):
  'Generates data containing batch_size samples' 
  # X : (n_samples, *dim, n_channels)
  # Initialization
  X = np.empty((self.batch_size, *self.dim, self.n_channels))
  y = np.empty((self.batch_size), dtype=int)

  # Generate data
  for i, ID in enumerate(list_IDs_temp):
      # Store sample
      X[i,] = np.load('data/' + ID + '.npy')

      # Store class
      y[i] = self.labels[ID]

  return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
```

During data generation, this code reads the NumPy array of each example from its corresponding file `ID.npy`. Since our code is multicore-friendly, note that you can do more complex operations instead (e.g. computations from source files) without worrying that data generation becomes a bottleneck in the training process.

Also, please note that we used Keras' `keras.utils.to_categorical` function to convert our numerical labels stored in `y` to a binary form (e.g. in a 6-class problem, the third label corresponds to `[0 0 1 0 0 0]`) suited for classification.

Now comes the part where we build up all these components together. Each call requests a batch index between 0 and the total number of batches, where the latter is specified in the `__len__` method.

```python
def __len__(self):
  'Denotes the number of batches per epoch'
  return int(np.floor(len(self.list_IDs) / self.batch_size))
```

so that the model sees the training samples at most once per epoch.

Now, when the batch corresponding to a given index is called, the generator executes the `__getitem__` method to generate it.

```python
def __getitem__(self, index):
  'Generate one batch of data'
  # Generate indexes of the batch
  indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

  # Find list of IDs
  list_IDs_temp = [self.list_IDs[k] for k in indexes]

  # Generate data
  X, y = self.__data_generation(list_IDs_temp)

  return X, y
```

The complete code corresponding to the steps that we described in this section is shown below.

```python
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """ Generates data for Keras
    Base object for fitting to a sequence of data, such as a dataset.
    Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement
    `on_epoch_end`. The method `__getitem__` should return a complete batch.
    """
    def __init__(self, list_IDs, labels, batch_size=32, 
                 dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # 实现了每个epoch都打乱一次，这里就需要ID的存在，而不能直接打乱数据
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
```

**Keras script**

Now, we have to modify our Keras script accordingly so that it accepts the generator that we just created.

```python
import numpy as np

from keras.models import Sequential
from my_classes import DataGenerator

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = # IDs # 字典：通过‘train’ 和 ‘validation’ 关键词索引ID列表
labels = # Labels # 字典：通过ID索引label

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)
    
    
    import multiprocessing
    model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs=FLAGS.max_epochs,
        verbose=1,
        callbacks=[history, tensorBoard, earlystopping, reducelr, modelcheckpoint],
        validation_data=validation_sequence,
        validation_steps = len(validation_sequence),
        max_queue_size=10,
        workers=int(multiprocessing.cpu_count() * 0.7),
        use_multiprocessing=True,
        shuffle=True
    )
```

As you can see, we called from `model` the `fit_generator` method instead of `fit`, where we just had to give our training generator as one of the arguments. Keras takes care of the rest!

Note that our implementation enables the use of the `multiprocessing` argument of `fit_generator`, where the number of threads specified in `n_workers` are those that generate batches in parallel. A high enough number of workers assures that CPU computations are efficiently managed, *i.e.* that the bottleneck is indeed the neural network's forward and backward operations on the GPU (and not data generation).

**Conclusion:**

This is it! You can now run your Keras script with the command

```
python3 keras_script.py
```

and you will see that during the training phase, **data** is **generated in parallel by the CPU** and then **directly fed to the GPU**.

You can find a complete example of this strategy on applied on a specific example on [GitHub](https://github.com/shervinea/enzynet) where codes of [data generation](https://github.com/shervinea/enzynet/blob/master/enzynet/volume.py#L24) as well as the [Keras script](https://github.com/shervinea/enzynet/blob/master/scripts/architecture/enzynet_adapted.py) are available.





# 2. 常用数据类型

## raster data

- [Why store data as a raster?](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/what-is-raster-data.htm#GUID-CBED6408-2437-4554-A3E1-F0FDC4AFBD63)
- [General characteristics of raster data](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/what-is-raster-data.htm#GUID-40BD8AAA-E6A8-4263-9086-20647BDDC00F)

In its simplest form, a raster consists of a matrix of cells (or pixels) organized into rows and columns (or a grid) where each cell contains a value representing information, such as temperature. Rasters are digital aerial photographs, imagery from satellites, digital pictures, or even scanned maps.

![The cells in a raster](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-6754AF39-CDE9-4F9D-8C3A-D59D93059BDD-web.png)

Data stored in a raster format represents real-world phenomena:

- Thematic data (also known as discrete) represents features such as land-use or soils data.
- Continuous data represents phenomena such as temperature, elevation, or spectral data such as satellite images and aerial photographs.
- Pictures include scanned maps or drawings and building photographs.

Thematic and continuous rasters may be displayed as data layers along with other geographic data on your map but are often used as the source data for spatial analysis with the ArcGIS Spatial Analyst extension. Picture rasters are often used as attributes in tables—they can be displayed with your geographic data and are used to convey additional information about map features.

[Learn more about thematic and continuous data](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/discrete-and-continuous-data.htm)

While the structure of raster data is simple, it is exceptionally useful for a wide range of applications. Within a GIS, the uses of raster data fall under four main categories:

- Rasters as basemaps

  A common use of raster data in a GIS is as a background display for other feature layers. For example, orthophotographs displayed underneath other layers provide the map user with confidence that map layers are spatially aligned and represent real objects, as well as additional information. Three main sources of raster basemaps are orthophotos from aerial photography, satellite imagery, and scanned maps. Below is a raster used as a basemap for road data.

  ![raster as basemap](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-327EC884-91B2-4EAF-8BC2-FE2F79B26DA8-web.gif)

  

- Rasters as surface maps

  Rasters are well suited for representing data that changes continuously across a landscape (surface). They provide an effective method of storing the continuity as a surface. They also provide a regularly spaced representation of surfaces. Elevation values measured from the earth's surface are the most common application of surface maps, but other values, such as rainfall, temperature, concentration, and population density, can also define surfaces that can be spatially analyzed. The raster below displays elevation—using green to show lower elevation and red, pink, and white cells to show higher elevations.

  ![Terrain analysis example showing elevation](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-8A8B9A2A-268C-4AE6-9036-80C82A3F7B78-web.gif)

  

- Rasters as thematic maps

  Rasters representing thematic data can be derived from analyzing other data. A common analysis application is classifying a satellite image by land-cover categories. Basically, this activity groups the values of multispectral data into classes (such as vegetation type) and assigns a categorical value. Thematic maps can also result from geoprocessing operations that combine data from various sources, such as vector, raster, and terrain data. For example, you can process data through a geoprocessing model to create a raster dataset that maps suitability for a specific activity. Below is an example of a classified raster dataset showing land use.

  ![thematic raster example](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-00D97209-C0FF-4976-9D43-129A2FC30874-web.gif)

  

- Rasters as attributes of a feature

  Rasters used as attributes of a feature may be digital photographs, scanned documents, or scanned drawings related to a geographic object or location. A parcel layer may have scanned legal documents identifying the latest transaction for that parcel, or a layer representing cave openings may have pictures of the actual cave openings associated with the point features. Below is a digital picture of a large, old tree that could be used as an attribute to a landscape layer that a city may maintain.

  ![Fig Tree](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-7232ED35-0509-49D4-BE1F-CF899144657F-web.gif)

  

## Why store data as a raster?

Sometimes you don't have the choice of storing your data as a raster; for example, imagery is only available as a raster. However, there are many other features (such as points) and measurements (such as rainfall) that could be stored as either a raster or a feature (vector) data type.

The advantages of storing your data as a raster are as follows:

- A simple data structure—A matrix of cells with values representing a coordinate and sometimes linked to an attribute table
- A powerful format for advanced spatial and statistical analysis
- The ability to represent continuous surfaces and perform surface analysis
- The ability to uniformly store points, lines, polygons, and surfaces
- The ability to perform fast overlays with complex datasets

There are other considerations for storing your data as a raster that may convince you to use a vector-based storage option. For example:

- There can be spatial inaccuracies due to the limits imposed by the raster dataset cell dimensions.

- Raster datasets are potentially very large. Resolution increases as the size of the cell decreases; however, normally cost also increases in both disk space and processing speeds. For a given area, changing cells to one-half the current size requires as much as four times the storage space, depending on the type of data and storage techniques used.

  [Learn more about cell size](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/cell-size-of-raster-data.htm)

- There is also a loss of precision that accompanies restructuring data to a regularly spaced raster-cell boundary.

[Learn more about representing features in a raster dataset](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/how-raster-data-is-stored-and-managed.htm)

## General characteristics of raster data

In raster datasets, each cell (which is also known as a pixel) has a value. The cell values represent the phenomenon portrayed by the raster dataset such as a category, magnitude, height, or spectral value. The category could be a land-use class such as grassland, forest, or road. A magnitude might represent gravity, noise pollution, or percent rainfall. Height (distance) could represent surface elevation above mean sea level, which can be used to derive slope, aspect, and watershed properties. Spectral values are used in satellite imagery and aerial photography to represent light reflectance and color.

Cell values can be either positive or negative, integer, or floating point. Integer values are best used to represent categorical (discrete) data and floating-point values to represent continuous surfaces. For additional information on discrete and continuous data, see [Discrete and continuous data](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/discrete-and-continuous-data.htm). Cells can also have a NoData value to represent the absence of data. For information on NoData, see [NoData in raster datasets](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/nodata-in-raster-datasets.htm).

![Cell values are applied to the center point or whole area of a cell](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-665BD874-9198-4488-9B19-DB33A3639F6D-web.gif)

Rasters are stored as an ordered list of cell values, for example, 80, 74, 62, 45, 45, 34, and so on.

![rasters are stored as an ordered list](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-B5896AC5-755E-4ADB-9AEA-455B6EE929F7-web.gif)

The area (or surface) represented by each cell consists of the same width and height and is an equal portion of the entire surface represented by the raster. For example, a raster representing elevation (that is, digital elevation model) may cover an area of 100 square kilometers. If there were 100 cells in this raster, each cell would represent 1 square kilometer of equal width and height (that is, 1 km x 1 km).

![cell width and height](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-D353AA60-32C2-4168-970A-E774B3E9A613-web.gif)

The dimension of the cells can be as large or as small as needed to represent the surface conveyed by the raster dataset and the features within the surface, such as a square kilometer, square foot, or even square centimeter. The cell size determines how coarse or fine the patterns or features in the raster will appear. The smaller the cell size, the smoother or more detailed the raster will be. However, the greater the number of cells, the longer it will take to process, and it will increase the demand for storage space. If a cell size is too large, information may be lost or subtle patterns may be obscured. For example, if the cell size is larger than the width of a road, the road may not exist within the raster dataset. In the diagram below, you can see how this simple polygon feature will be represented by a raster dataset at various cell sizes.

![raster feature cell size](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-F1FD32DF-3924-4EC6-907E-D0B8BFB8DBEF-web.gif)

The location of each cell is defined by the row or column where it is located within the raster matrix. Essentially, the matrix is represented by a Cartesian coordinate system, in which the rows of the matrix are parallel to the x-axis and the columns to the y-axis of the Cartesian plane. Row and column values begin with 0. In the example below, if the raster is in a Universal Transverse Mercator (UTM) projected coordinate system and has a cell size of 100, the cell location at 5,1 would be 300,500 East, 5,900,600 North.

![Coordinate location](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-F04B5976-9501-4546-A2E9-CBD0ACA6B970-web.gif)

[Learn about transforming the raster dataset](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/georeferencing-a-raster-to-a-vector.htm)

Often you need to specify the extent of a raster. The extent is defined by the top, bottom, left, and right coordinates of the rectangular area covered by a raster, as shown below.

![raster extents](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/GUID-B59895EA-0676-488C-AC9F-70C586AEEA50-web.gif)

## Related Topics

- [Raster bands](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/raster-bands.htm)
- [Cell size of raster data](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/cell-size-of-raster-data.htm)
- [How features are represented in a raster](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/how-features-are-represented-in-a-raster.htm)
- [Supported raster dataset file formats](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/supported-raster-dataset-file-formats.htm)

Feedback on this topic?

ArcGIS for Desktop[Home](http://desktop.arcgis.com/en/)[ArcGIS Pro](http://pro.arcgis.com/en/pro-app/)[ArcMap](http://desktop.arcgis.com/en/arcmap/)[Documentation](http://desktop.arcgis.com/en/documentation/)[Pricing](http://desktop.arcgis.com/en/pricing/)[Support](http://desktop.arcgis.com/en/support/)ArcGIS Platform[ArcGIS Online](http://www.arcgis.com/)[ArcGIS for Desktop](http://desktop.arcgis.com/)[ArcGIS for Server](http://server.arcgis.com/)[ArcGIS for Developers](https://developers.arcgis.com/)[ArcGIS Solutions](http://solutions.arcgis.com/)[ArcGIS Marketplace](http://marketplace.arcgis.com/)About Esri[About Us](http://www.esri.com/about-esri/)[Careers](http://www.esri.com/careers/)[Insiders Blog](http://blogs.esri.com/esri/esri-insider/)[User Conference](http://www.esri.com/events/user-conference/index.html)[Developer Summit](http://www.esri.com/events/devsummit/index.html)[Esri](http://esri.com/)   © Copyright 2016 Environmental Systems Research Institute, Inc. | [Privacy](http://www.esri.com/legal/privacy) | [Legal](http://www.esri.com/legal/software-license)



# 3. 数据处理

## 数据增广



```python
# -*- coding:utf-8 -*-

"""数据增强
   1. 翻转变换 flip
   2. 随机修剪 random crop
   3. 色彩抖动 color jittering
   4. 平移变换 shift
   5. 尺度变换 scale
   6. 对比度变换 contrast
   7. 噪声扰动 noise
   8. 旋转变换/反射变换 Rotation/reflection
"""

from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import threading, os, time
import logging
import math
import shutil

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:
    """
    包含数据增强的八种方式
    """

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomFlip(image, mode=Image.FLIP_LEFT_RIGHT):
        """
        对图像进行上下左右四个方面的随机翻转
        :param image: PIL的图像image
        :param model: 水平或者垂直方向的随机翻转模式,默认右向翻转
        :return: 翻转之后的图像
        """
        #random_model = np.random.randint(0, 2)
        #flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
        #return image.transpose(flip_model[random_model])
        return image.transpose(mode)

    @staticmethod
    def randomShift(image):
    #def randomShift(image, xoffset, yoffset=None):
        """
        对图像进行平移操作
        :param image: PIL的图像image
        :param xoffset: x方向向右平移
        :param yoffset: y方向向下平移
        :return: 翻转之后的图像
        """
        random_xoffset = np.random.randint(0, math.ceil(image.size[0]*0.2))
        random_yoffset = np.random.randint(0, math.ceil(image.size[1]*0.2))
        #return image.offset(xoffset = random_xoffset, yoffset = random_yoffset)
        return image.offset(random_xoffset)

    @staticmethod
    def randomRotation(image, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode)

    @staticmethod
    def randomCrop(image):
        """
        对图像随意剪切,裁剪图像大小宽和高的2/3
        :param image: PIL的图像image
        :return: 剪切之后的图像

        """
        image_width = image.size[0]
        image_height = image.size[1]
        crop_image_width = math.ceil(image_width*2/3)
        crop_image_height = math.ceil(image_height*2/3)
        x = np.random.randint(0, image_width - crop_image_width)
        y = np.random.randint(0, image_height - crop_image_height) 
        random_region = (x, y, x + crop_image_width, y + crop_image_height)
        return image.crop(random_region)

    @staticmethod
    def randomColor(image):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    @staticmethod
    def randomGaussian(image, mean=0.2, sigma=0.3):
        """
         对图像进行高斯噪声处理
        :param image:
        :return:
        """

        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
            :param sigma: 标准差
            :return:
            """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        # 将图像转化成数组
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        try:
            img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
            img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
            img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
        except:
            img = img
        return Image.fromarray(np.uint8(img))

    @staticmethod
    def saveImage(image, path):
        try:
            image.save(path)
        except:
            print('not save img: ', path)
            pass

files = []
def get_files(dir_path):
    global files
    if os.path.exists(dir_path):
        parents = os.listdir(dir_path)
        for parent in parents:
            child = os.path.join(dir_path, parent)
            if os.path.exists(child) and os.path.isfile(child):
               #child = child.split('/')[4:]
               #str_child = '/'.join(child)
               files.append(child)
            elif os.path.isdir(child):
                get_files(child)
        return files
    else:
        return None

if __name__ == '__main__':
    times = 2  #重复次数
    imgs_dir = '/opt/sda/imgData20190322/train'
    new_imgs_dir = '/opt/sda/imgData20190322/train_data_augment'
    #if os.path.exists(new_imgs_dir):
    #    shutil.rmtree(new_imgs_dir)
    funcMap = {"flip": DataAugmentation.randomFlip,
               "rotation": DataAugmentation.randomRotation,
               "crop": DataAugmentation.randomCrop,
               "color": DataAugmentation.randomColor,
               "gaussian": DataAugmentation.randomGaussian
               }
    #funcLists = {"flip", "rotation", "crop", "color", "gaussian"}
    funcLists = {"flip", "rotation", "crop", "gaussian"}
    
    global _index
    imgs_list = get_files(imgs_dir)
    for index_img, img in enumerate(imgs_list):
        if index_img != 0 and index_img % 50 == 0:
            print('now is dealing %d image' % (index_img) )
        tmp_img_dir_list = img.split('/')[:-1]
        tmp_img_dir_list[0:len(new_imgs_dir.split('/'))] = new_imgs_dir.split('/')
        new_img_dir = '/'.join(tmp_img_dir_list)

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        try:
            shutil.copy(img, os.path.join(new_img_dir, img.split('/')[-1]))
        except:
            pass

        img_name = img.split('/')[-1].split('.')[0]
        postfix = img.split('.')[1]   #后缀 
        if postfix.lower() in ['jpg', 'jpeg', 'png', 'bmp']:
            image = DataAugmentation.openImage(img)
            _index = 1
            for func in funcLists:
                if func == 'flip':
                    flip_model = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
                    for model_index in range(len(flip_model)):
                        new_image = DataAugmentation.randomFlip(image, flip_model[model_index])
                        img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                        DataAugmentation.saveImage(new_image, img_path)
                        _index += 1 
                elif func == 'gaussian':
                   new_image = DataAugmentation.randomGaussian(image)
                   img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                   DataAugmentation.saveImage(new_image, img_path)
                   _index += 1 
                else:
                    for _i in range(0, times, 1):
                        new_image = funcMap[func](image)
                        img_path = os.path.join(new_img_dir, img_name + '_' + str(_index) + '.' + postfix)
                        DataAugmentation.saveImage(new_image, img_path)
                        _index += 1
                        
```





## 样本均衡

#### 简单的过采样和欠抽样

过采样：将稀有类别的样本进行复制，通过增加此稀有类样本的数量来平衡数据集。该方法适用于数据量较小的情况。

欠抽样：从丰富类别的样本中**随机选取**和稀有类别相同数目的样本，通过减少丰富类的样本来平衡数据集。该方法适用于数据量较大的情况。

也可以将过采样和欠采样结合在一起使用。

**存在问题**

过采样：可能会存在过拟合问题。（可以使用SMOTE算法，增加随机的噪声的方式来改善这个问题）
欠采样：可能会存在信息减少的问题。因为只是利用了一部分数据，所以模型只是学习到了一部分模型。

#### 过采样改进（smote算法）

这种方法属于过采样的一种，主要是将样本较少的类别进行重新组合构造新的样本。

SMOTE算法是一种过采样的算法。这个算法不是简单的复制已有的数据，而是在原有数据基础上，通过算法产生新生数据。

SMOTE全称是Synthetic Minority Oversampling Technique即合成少数类过采样技术，它是基于随机过采样算法的一种改进方案，由于随机过采样采取简单复制样本的策略来增加少数类样本，这样容易产生模型过拟合的问题，即使得模型学习到的信息过于特别(Specific)而不够泛化(General)，***SMOTE算法的基本思想是对少数类样本进行分析并根据少数类样本人工合成新样本添加到数据集中，算法流程如下：***

- 对于少数类中每一个样本x，以欧氏距离为标准计算它到少数类样本集$S_{min}$ 中所有样本的距离，得到其k近邻。

- 根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本xx，从其k近邻中随机选择若干个样本，假设选择的近邻为xn。

- 对于每一个随机选出的近邻xn，分别与原样本按照如下的公式构建新的样本
  $$
  x_{new}=x+rand(0,1) * | x - x_n |
  $$



#### 欠采样改进

**方法一：模型融合 （bagging的思想 ）**

- 从丰富类样本中随机的选取（**有放回的选取**）和稀有类等量样本的数据。和稀有类样本组合成新的训练集。这样我们就产生了多个训练集，并且是互相独立的，然后训练得到多个分类器。
- 若是分类问题，就把多个分类器投票的结果（少数服从多数）作为分类结果。若是回归问题，就将均值作为最后结果。

**方法二：增量模型 （boosting的思想）**

- 使用全部的样本作为训练集，得到分类器L1
- 从L1正确分类的样本中和错误分类的样本中各抽取50%的数据，即循环的一边采样一个。此时训练样本是平衡的。训练得到的分类器作为L2.
- 从L1和L2分类结果中，选取结果不一致的样本作为训练集得到分类器L3.
- 最后投票L1,L2,L3结果得到最后的分类结果。

#### 样本权重（sample weight）

交叉熵误差
交叉熵公式如下

![](http://latex.codecogs.com/gif.latex?E=-\sum_{k}t_{k}log&space;y_{k})



这里，log表示以e为底数的自然对数。y_k是神经网络的输出，t_k是正确的标签。并且，t_k中只有正确的标签的索引为1，其他均为0（one-hot表示）。用mnist数据举例，如果是3，那么标签是[0，0，0，1，0，0，0，0，0，0]，除了第4个值为1，其他全为0。

因此，上公式实际上只计算对应正确的标签的输出的自然对数，交叉熵误差的值是由正确的标签所对应的输出结果决定的。

在tensorflow里面，是用

```
tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)

```



来计算交叉熵的。predicttion就是神经网络最后一层的输出，y就是实际的标签。

带权重的交叉熵loss
那么现在我们修改交叉熵loss的权重，要增加少数类的分错的权重，就是在交叉熵的公式里面对应的少数类乘上一个系数：

![](http://latex.codecogs.com/gif.latex?E=-\sum_{k}\alpha\cdot&space;t_{k}\cdot&space;log&space;y_{k}\left&space;\begin{pmatrix}&space;\alpha&space;=n(n%3E1)&if&k=i&space;\\&space;\alpha&space;=1&if&k\neq&space;i&space;\end{pmatrix})

假设k类里面，第i类是少数类，为了加大分错第类的成本，在交叉熵上给第i类乘以一个大于1的系数，这样如果分错第i类的话，交叉熵loss就会增加。

下面，就用一个二分类的例子的代码来说明一下，以下是用tensorflow来举例。

在硬盘故障预测里面，由于损坏的硬盘数量远远小于健康硬盘的数量，样本极其不均衡，所以通过修稿交叉熵的loss视图体改算法的准确率。

假设健康硬盘样本的标签为[1,0]，损坏硬盘样本的标签为[0,1]，即在标签中第一个索引为1的是健康硬盘的样本，第二个索引为1的是损坏的样本。

假设在网络中，最后一层输出的结果是prediction，我们要先对prediction做一个softmax，求取输出属于某一类的概率：

```
yprediction = tf.nn.softmax(prediction)
```



然后给实际标签乘上一个系数，健康的样本保持系数1不变，损坏样本乘上系数10，这样健康样本的标签就变为[1,0]，损坏样本的标签就变为[0,10]，再计算交叉熵loss：

```
coe = tf.constant([1.0,10.0])
y_coe = y*coe
loss = -tf.reduce_mean(y_coe*tf.log(yprediction))

```



这样就达到了修改交叉熵loss增大分错损坏样本的成本的目的。这时候公式就变为：

![](http://latex.codecogs.com/gif.latex?E=-\sum_{k}\alpha\cdot&space;t_{k}\cdot&space;log&space;y_{k}\left&space;\begin{pmatrix}&space;\alpha&space;=1&if&k=healthDisk&space;\\&space;\alpha&space;=10&if&k=failureDisk&space;\end{pmatrix})
$$
E=-\sum_{k}\alpha\cdot t_{k}\cdot log y_{k}\left \begin{pmatrix} \alpha =1&if&k=healthDisk \\ \alpha =10&if&k=failureDisk \end{pmatrix}
$$


tensorflow官方的权重交叉熵

```
tf.nn.weighted_cross_entropy_with_logits(labels,logits, pos_weight, name=None)
```



参考文献

https://blog.csdn.net/mao_xiao_feng/article/details/53382790
https://www.svds.com/learning-imbalanced-classes/
http://scikitlearn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
深度学习入门-基于python的与实现 by 斋藤康毅



将二分类问题转换成其他问题
对于正负样本极不平衡的场景，我们可以换一个完全不同的角度来看待问题：把它看做一分类（One Class Learning）或异常检测（Novelty Detection）问题。这类方法的重点不在于捕捉类间的差别，而是为其中一类进行建模，经典的工作包括One-class SVM等。

说明：对于正负样本极不均匀的问题，使用异常检测，或者一分类问题，也是一个思路。

使用其他评价指标
在准确率不行的情况下，使用召回率或者精确率试试。

准确度这个评价指标在类别不均衡的分类任务中并不能work。几个比传统的准确度更有效的评价指标：

混淆矩阵(Confusion Matrix)：使用一个表格对分类器所预测的类别与其真实的类别的样本统计，分别为：TP、FN、FP与TN。
　　精确度(Precision)
　　召回率(Recall)
　　F1得分(F1 Score)：精确度与找召回率的加权平均。

特别是：
Kappa (Cohen kappa)
ROC曲线(ROC Curves)：见Assessing and Comparing Classifier Performance with ROC Curves

如何选择
解决数据不平衡问题的方法有很多，上面只是一些最常用的方法，而最常用的方法也有这么多种，如何根据实际问题选择合适的方法呢？接下来谈谈一些我的经验。

1、在正负样本都非常之少的情况下，应该采用数据合成的方式；

2、在负样本足够多，正样本非常之少且比例及其悬殊的情况下，应该考虑一分类方法；

3、在正负样本都足够多且比例不是特别悬殊的情况下，应该考虑采样或者加权的方法。

4、采样和加权在数学上是等价的，但实际应用中效果却有差别。尤其是采样了诸如Random Forest等分类方法，训练过程会对训练集进行随机采样。在这种情况下，如果计算资源允许上采样往往要比加权好一些。

5、另外，虽然上采样和下采样都可以使数据集变得平衡，并且在数据足够多的情况下等价，但两者也是有区别的。实际应用中，我的经验是如果计算资源足够且小众类样本足够多的情况下使用上采样，否则使用下采样，因为上采样会增加训练集的大小进而增加训练时间，同时小的训练集非常容易产生过拟合。

6、对于下采样，如果计算资源相对较多且有良好的并行环境，应该选择Ensemble方法。

参考文献
如何解决样本不均衡问题
SMOTE算法(人工合成数据)<https://blog.csdn.net/jiede1/article/details/70215477>
如何解决机器学习中数据不平衡问题







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




