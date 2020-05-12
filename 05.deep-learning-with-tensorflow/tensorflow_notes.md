# tf
Resources:
- [TensorFlow 1.15 api_docs](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf)

- [TensorFlow newest api_docs](https://www.tensorflow.org/api_docs/python/tf)

## Variable
### tf.Variable 
Variable 是tensorflow的一个类，里面封装了很多operations,简称ops,所以它是大写的。创建Variable对象后，必须经过初始化才可以使用(类的实例化)

**关于Variable的eval()方法：**
```python
  # W is a random 700 x 100 variable object 
  W = tf.Variable(tf.truncated_normal([700, 10])) 
  with tf.Session() as sess: 
      sess.run(W.initializer)
  print W 
  >> Tensor("Variable/read:0", shape=(700, 10), dtype=float32)
  ## 
  W = tf.Variable(tf.truncated_normal([700, 10])) 
  with tf.Session() as sess: 
      sess.run(W.initializer)
  print W.eval()
  >> [[-0.76781619 -0.67020458 1.15333688 ..., -0.98434633 -1.25692499  -0.90904623]
```
**Variable的assign()方法：**
  ```python
    W = tf.Variable(10)
    W.assign(100) 
    with tf.Session() as sess: 
        sess.run(W.initializer)
    print W.eval() 
    # 打印的结果，是10，Why?

    #一条tensorflow的规则： W.assign(100) 并不会给W赋值，assign()是一个op，所以它返回一个op object，需要在Session中run这个op object，才会赋值给W.
    W = tf.Variable(10)
    assign_op = W.assign(100) 
    with tf.Session() as sess:
      sess.run(W.initializer) # 此句可以省略，因为assign_op可以完成赋初始值操作。事实上， initializer op 是一个特殊的assign op.
        sess.run(assign_op) 
        print W.eval() # >> 100

  ```

### tf.variable_scope
使用varibale_scope主要有两个目的：
1. 可以使用同一个变量名创建不同的变量（貌似直接使用的话还是默认后面的transform），并多次调用同一个模块（虽然函数名不同，但其内部都是使用的同样的卷积模块，同样的，各卷积模块来自同一个卷积函数并使用scope区分，但transform函数中用的卷积模块scope是相同的，这样就需要再在外部套一个scope,形成transform_net1/conv1/）
```python
with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)

with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
```
2. 创建变量时可以使用同一变量名，不同变量用scope区分，使用get_variable()创建变量，如果相同已经存在，不用再多次创建
```python
with tf.variable_scope('transform_XYZ') as sc:
        assert(K==3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
```
## tf.Group
#### tf.Graph()
Graphs are used by tf.functions to represent the function's computations. Each graph contains a set of tf.Operation objects, which represent units of computation; and tf.Tensor objects, which represent the units of data that flow between operations.
#### Methods
**as_default()**
This method should be used if you want to create multiple graphs in the same process. For convenience, a global default graph is provided, and all ops will be added to this graph if you do not create a new graph explicitly.

  - Use this method with the with keyword to specify that ops created within the scope of a block should be added to this graph. 
  - The default graph is a property of the current thread. If you create a new thread, and wish to use the default graph in that thread, you must explicitly add a with g.as_default(): in that thread's function.

```python
#The following code examples are equivalent:
# 1. Using Graph.as_default():
g = tf.Graph()
with g.as_default():
  c = tf.constant(5.0)
  assert c.graph is g

# 2. Constructing and making default:
with tf.Graph().as_default() as g:
  c = tf.constant(5.0)
  assert c.graph is g
```
  - If eager execution is enabled ops created under this context manager will be added to the graph instead of executed eagerly.

Returns:
A context manager for using this graph as the default graph.



# tf.data
xxx
# tf.keras
xxx
# tf.layers
xxx

# tf.losses
主要分为两类损失(loss)，一种是针对拟合任务的MSE损失，另一种是针对分类任务的Cross Entropy损失，这两种损失根据具体的应用场景具有不同的变种。

## MSE
$$
MSE = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i-y_i)^2
$$

## Cross Entropy
## 定义：
$$
CE = -\frac{1}{m} \sum_{i=1}^m y_i * log_2 \hat{y}_i
$$
- 这里要求label与输出值在0~1之间，将值限制在0~1之间可以通过softmax函数和sigmoid函数实现，因此对应的产生了两种损失函数；
**交叉熵与KL散度**
KL散度用来衡量两个分布p和q的相似度：
$$
D_{KL}(p||q) = \sum_{i=1}^m p(x_i) * log_2 \frac{p(x_i)}{q(x_i)} \\
             = \sum_{i=1}^m p(x_i) * log_2 p(x_i) - \sum_{i=1}^m p(x_i) * log_2 q(x_i) \\
             = -H(p) + H(p,q)
$$
- 第一项称为信息熵， 第二项便是交叉熵

### tf.nn.softmax_cross_entropy_with_logits
对于独立**互斥**离散的多分类任务(multi-class), 我们使用此类损失。即类与类之间是独立的，且每个样本只对应一个类(在某一类的上概率为1，其余为0)
 - logits先经过softmax层转化为概率分布，然后计算交叉熵(只计算类别对应为1的损失)
 - 因此，该函数要求每个样本的label都是一个关于类别的有效的概率分布，即使用one-hot编码；
 - 计算损失之后输出尺寸为(batch, 1)的loss向量，因此后续还需要计算均值得到本批样本的L;(函数计算的是单个样本的交叉熵l)

### tf.nn.sigmoid_cross_entropy_with_logits
应该是使用的BCE公式：
$$
BCE = -\sum_{i=0}^1 y_i * log_2 \hat{y}_i = y_i * log_2 \hat{y}_i + (1-y_i) * log_2 (1-\hat{y}_i) 
$$
此损失函数用于衡量独立**不互斥**离散分类任务的误差(multi-label)。如多分类任务中的多目标检测任务，一个样本的标签内部可以包含多个1，代表包含多个类别被检测到。而使用sigmoid函数恰好满足此要求。
- 该函数的输入为网络的输出logits；标签不需要进行one-hot处理；
- 该函数的输出为 (batch, n_class) 的loss,即单独对每一个类计算CE (BCE?);最后我们再进行求均值即可；

### tf.nn.sparse_softmax_cross_entropy_with_logits
此函数是tf.nn.softmax_cross_entropy_with_logits易用版本，不再需要手动对标签进行one-hot编码：
- 该函数的输入logits仍然是(batch, n_class)的网络输出，但label的形状是(batch,)，即每个样本的标签直接为取值为(0,n_class-1)的类别值

### 区别与联系
- 从函数表达的形式，sigmoid函数是softmax函数的一种特殊情况，对于二分类，二者计算得到的概率是相同的；因此对于二分类问题，两种损失没有区别（标签相同，概率分布相同）；计算公式不同但结果相同：logits=[0.8, 0.2], label=[1,0], sigmoid_CE=[ln0.8, ln0.8], softmax_CE=[ln0.8],计算均值后二者相同； 但对于multi-label的分类问题，sigmoid把每一个类别都当做二分类来处理；
- tf.nn.sigmoid_cross_entropy_with_logits solves N binary classifications at once.
-tf.losses.sigmoid_cross_entropy in addition allows to set the in-batch weights, i.e. make some examples more important than others. 
- tf.nn.weighted_cross_entropy_with_logits allows to set class weights (remember, the classification is binary), i.e. make positive errors larger than negative errors. This is useful when the training data is unbalanced.

- Just like in sigmoid family, tf.losses.softmax_cross_entropy allows to set the in-batch weights, i.e. make some examples more important than others. As far as I know, as of tensorflow 1.3, there's no built-in way to set class weights.

- For sparse: Like above, tf.losses version has a weights argument which allows to set the in-batch weights.

# tf.metrics

# tf.saved_model

# tf.train

### Learning Rate
#### 指数衰减学习率
见pointNet代码

#### 固定步数变化学习率
```python
def piecewise_constant(x, boundaries, values, name=None)

#Example: use a learning rate that's 1.0 for the first 100001 steps, 
# 0.5 for the next 10000 steps, and 0.1 for any additional steps.
  global_step = tf.Variable(0, trainable=False)
  boundaries = [100000, 110000]
  values = [1.0, 0.5, 0.1]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
```

# tf.summary
关于如何想keras一样输出model.summary()
https://stackoverflow.com/questions/46560313/is-there-an-easy-way-to-get-something-like-keras-model-summary-in-tensorflow
## Overview
The tf.summary module provides APIs for writing summary data. This data can be visualized in TensorBoard, the visualization toolkit that comes with TensorFlow. See the [TensorBoard website](https://www.tensorflow.org/tensorboard) for more detailed tutorials about how to use these APIs, or some quick examples below.

Example usage with eager execution, the default in TF 2.0:
```python
writer = tf.summary.create_file_writer("/tmp/mylogs")
with writer.as_default():
  for step in range(100):
    # other model code would go here
    tf.summary.scalar("my_metric", 0.5, step=step)
    writer.flush()
```

```python
## Example usage with tf.function graph execution:
writer = tf.summary.create_file_writer("/tmp/mylogs")

@tf.function
def my_func(step):
  # other model code would go here
  with writer.as_default():
    tf.summary.scalar("my_metric", 0.5, step=step)

for step in range(100):
  my_func(step)
  writer.flush()

##Example usage with legacy TF 1.x graph execution:
with tf.compat.v1.Graph().as_default():
  step = tf.Variable(0, dtype=tf.int64)
  step_update = step.assign_add(1)
  writer = tf.summary.create_file_writer("/tmp/mylogs")
  with writer.as_default():
    tf.summary.scalar("my_metric", 0.5, step=step)
  all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
  writer_flush = writer.flush()

  sess = tf.compat.v1.Session()
  sess.run([writer.init(), step.initializer])
  for i in range(100):
    sess.run(all_summary_ops)
    sess.run(step_update)
    sess.run(writer_flush)
```
### Functions
  - audio(...): Write an audio summary.
  - create_file_writer(...): Creates a summary file writer for the given log directory.
  - create_noop_writer(...): Returns a summary writer that does nothing.
  - flush(...): Forces summary writer to send any buffered data to storage.
  - histogram(...): Write a histogram summary.
  - image(...): Write an image summary.
  - record_if(...): Sets summary recording on or off per the provided boolean value.
  - scalar(...): Write a scalar summary.
  - text(...): Write a text summary.
  - trace_export(...): Stops and exports the active trace as a Summary and/or profile file.
  - trace_off(...): Stops the current trace and discards any collected information.
  - trace_on(...): Starts a trace to record computation graphs and profiling information.
  - write(...): Writes a generic summary to the default SummaryWriter if one exists.

-----------------------------------------------
# tf.scan



```python
tf.scan(
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    reverse=False,
    name=None
)
```

Defined in [`tensorflow/python/ops/functional_ops.py`](https://www.tensorflow.org/code/stable/tensorflow/python/ops/functional_ops.py).

scan on the list of tensors unpacked from `elems` on dimension 0.

The simplest version of `scan` repeatedly applies the callable `fn` to a sequence of elements from first to last. The elements are made of the tensors unpacked from `elems` on dimension 0. The callable fn takes two tensors as arguments. The first argument is the accumulated value computed from the preceding invocation of fn. If `initializer`is None, `elems` must contain at least one element, and its first element is used as the initializer.

Suppose that `elems` is unpacked into `values`, a list of tensors. The shape of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`. If reverse=True, it's fn(initializer, values[-1]).shape.

This method also allows multi-arity `elems` and accumulator. If `elems` is a (possibly nested) list or tuple of tensors, then each of these tensors must have a matching first (unpack) dimension. The second argument of `fn` must match the structure of `elems`.

If no `initializer` is provided, the output structure and dtypes of `fn` are assumed to be the same as its input; and in this case, the first argument of `fn` must match the structure of `elems`.

If an `initializer` is provided, then the output of `fn` must have the same structure as `initializer`; and the first argument of `fn` must match this structure.

For example, if `elems` is `(t1, [t2, t3])` and `initializer` is `[i1, i2]` then an appropriate signature for `fn` in `python2` is: `fn = lambda (acc_p1, acc_p2), (t1, [t2, t3]):` and `fn` must return a list, `[acc_n1, acc_n2]`. An alternative correct signature for `fn`, and the one that works in `python3`, is: `fn = lambda a, t:`, where `a` and `t`correspond to the input tuples.

#### Args:

- **fn**: The callable to be performed. It accepts two arguments. The first will have the same structure as `initializer` if one is provided, otherwise it will have the same structure as `elems`. The second will have the same (possibly nested) structure as `elems`. Its output must have the same structure as `initializer` if one is provided, otherwise it must have the same structure as `elems`.
- **elems**: A tensor or (possibly nested) sequence of tensors, each of which will be unpacked along their first dimension. The nested sequence of the resulting slices will be the first argument to `fn`.
- **initializer**: (optional) A tensor or (possibly nested) sequence of tensors, initial value for the accumulator, and the expected output type of `fn`.
- **parallel_iterations**: (optional) The number of iterations allowed to run in parallel.
- **back_prop**: (optional) True enables support for back propagation.
- **swap_memory**: (optional) True enables GPU-CPU memory swapping.
- **infer_shape**: (optional) False disables tests for consistent output shapes.
- **reverse**: (optional) True scans the tensor last to first (instead of first to last).
- **name**: (optional) Name prefix for the returned tensors.

#### Returns:

A tensor or (possibly nested) sequence of tensors. Each tensor packs the results of applying `fn` to tensors unpacked from `elems` along the first dimension, and the previous accumulator value(s), from first to last (or last to first, if `reverse=True`).

#### Raises:

- **TypeError**: if `fn` is not callable or the structure of the output of `fn` and `initializer` do not match.
- **ValueError**: if the lengths of the output of `fn` and `initializer` do not match.

Examples:

> ```
> elems = np.array([1, 2, 3, 4, 5, 6])
> sum = scan(lambda a, x: a + x, elems)
> # sum == [1, 3, 6, 10, 15, 21]
> sum = scan(lambda a, x: a + x, elems, reverse=True)
> # sum == [22, 21, 18, 15, 11, 6]
> ```

> ```
> elems = np.array([1, 2, 3, 4, 5, 6])
> initializer = np.array(0)
> sum_one = scan(
>     lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
> # sum_one == [1, 2, 3, 4, 5, 6]
> ```

> ```
> elems = np.array([1, 0, 0, 0, 0, 0])
> initializer = (np.array(0), np.array(1))
> fibonaccis = scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
> # fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])
> ```





tf.scan 函数定义如下

scan(
​    fn,
​    elems,
​    initializer=None,
​    parallel_iterations=10,
​    back_prop=True,
​    swap_memory=False,
​    infer_shape=True,
​    name=None
)

该函数就是一个递归函数。
我们先来看几个简单的例子：
例子一： 
elems = np.array([1, 2, 3, 4, 5, 6])
sum = scan(lambda a, x: a + x, elems)

sum == [1, 3, 6, 10, 15, 21]

在这里的scan中，由于没有initializer这个初始化参数，所以 ，在运行过程中，elems的第一个元素会被当作是初始化的值赋给a,
index     a      x     sum
1            1     #       1
2            1      2      3
3            3      3      6
4            6      4      10
5           10     5      15
6           15     6      21
故a中存储的是上一时刻的值，而x中存储的是输入的值的依次遍历

例子二：
import tensorflow as tf
import numpy as np 

elems = np.array([1, 2, 3, 4, 5, 6])
initializer = np.array(0)
sum_one = tf.scan(lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)

sum_one == [1, 2, 3, 4, 5, 6]

观察这个函数，首先，它传入的值是两个list(elems+1,elems),
其中elems+1为[2,3,4,5,6,7],elems为[1,2,3,4,5,6]
a被初始化为0
故在函数执行过程中
index   a       x        sum_one
1          0    [2,1]     2-1+0=1
2          1    [3,2]     3-2+1=2
3          2    [4,3]     4-3+2=3
4          3    [5,4]           4
5          4    [6,5]           5
6          5    [7,6]           6
由于**当传入多个值时，是依次取对应索引上的值**，故而在elems中传入的值必须要shape相同，也很容易注意到，a保存的是前一时刻的计算结果，x中保存的是当前的输入值。

例子三：
import tensorflow as tf
import numpy as np
elems = np.array([1, 0, 0, 0, 0, 0])
initializer = (np.array(0), np.array(1))
fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
sess = tf.Session()
print initializer
print sess.run(fibonaccis)

fibonaccis == ([1, 1, 2, 3, 5, 8], [1, 2, 3, 5, 8, 13])

index       a          x      fibonaccis
1            [0,1]      1          [1],[1]
2            [1,1]      0          [1,1],[1,2]
3            [1,2]      0          [1,1,2],[1,2,3]
4            [2,3]      0          [1,1,2,3],[1,2,3,5]
.....依次类推
在初始化值有多个的情况下，是这样调用的。

到此我们分别总结了初始化为一个列表的情况，和输入为列表的情况。现在就要来自己实现一个，初始化为多值的列表，输入同样是为一个列表的情况。（大部分代码中应该都有这样的需求）
前面的列子均来自于官网tf.scan函数下面的函数。下面的这个例子属于笔者胡编乱造，为了看下如何实现两个输入均为列表的情况

import tensorflow as tf
import numpy as np
def fn(xs,ys):
​    (x1,x2) = xs
​    (y1,y2) = ys
​    return (y1+y2+x1,y2*y1+x2)

elems = np.array([1,2,3,4,5,6])
initializer = (np.array(0),np.array(1))
outcome = tf.scan(fn,(elems+1,elems),initializer)

sess = tf.Session()
print "outcome",sess.run(outcome)

#outcome (array([ 3,  8, 15, 24, 35, 48]), array([  3,   9,  21,  41,  71, 113]))

让我们来看看它是如何生成序列的
首先，我们在scan中设置了elmes和initializer,
在调用函数时，xs中为[[2,3,4,5,6,7],[1,2,3,4,5,6]]
​                      ys中为[0,1]
下面的xs和ys仅表示在某一时刻下的值
index       xs        ys        outcome
1            [0,1][2,1]      [3],[3]
2            [3,3][3,2]      [3,8],[3,9]
3            [8,9][4,3]      [3,8,15],[3,9,21]
...
依次类推

故而，对于tf.scan函数，可以总结为以下几点
1.当没有赋初始值时，tf.scan会把elems的第一个元素作为初始化值，故，elems在initializer没有值时一定要有值。
2.initializer参数和elems参数都可以有若干个。
3.initializer中的值只在第一轮中用到，然后它就用来存储每次计算出来的值，所以，初始化值和输出的中间结果个数一定要想等。

到这里，tf.scan常用参数大家应该都已经很清楚了，笔者反正感觉应该可以开始写代码了。

加油↖(^ω^)↗
于7月19日两点，笔者成功的把tf.scan用在了自己的程序上，并且完美的实现了功能，点个赞。

补充：在实践过程中 ，你会遇到一个问题。就是有些参数被用来初始化参数，但是却并不被返回。而有些值则是在调用函数中被生成，并且会被返回，遇到这样的情况，我们应该怎么做呢？

通过探索，在tf.scan中并没有这个功能，对于那种用作初始化并且不会在循环中改变的值，我们只能用self.* = *的形式传入。我在tf的相关函数中，均没有看到对这种问题的解决方法。
但在这里我不能妄下定论，只是笔者没有找到，因为看到这篇文章并且知道的大神可以给予指正 



tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)

fn：计算函数
 elems：以elems的第一维度的变量list作函数计算直到遍历完整个elems
 initializer：fn计算的初始值，替代elems做第一次计算

举个好理解的例子：

```
x = [1,2,3]
z = 10

x = tf.convert_to_tensor(x)
z = tf.convert_to_tensor(z)

def f(x,y):
    return x+y

g = tf.scan(fn=f,elems = x,initializer=z)

sess = tf.Session()
sess.run(tf.global_variables_initializer)

sess.run(g)
```

会得到：

```
In [97]: sess.run(g)
Out[97]: array([11, 13, 16], dtype=int32)
```

详细的计算逻辑如下：
 11 = 10(初始值initializer)+ 1(x[0])
 13 = 11(上次的计算结果)+2(x[1])
 16 = 13(上次的计算结果)+3(x[2])

作者：slade_sal

链接：https://www.jianshu.com/p/3ea4429593a6

來源：简书

简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。



tf.scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True, swap_memory=False, infer_shape=True, name=None)

具体参数定义可以参见官网，这里捡最常见的形式来说，常用函数形式为tf.scan(fn, elems, initializer=None)
 函数说明：scan on the list of tensors unpacked from elems on dimension 0.
 f(n)以elems的第一维度的变量list作函数计算直到遍历完整个elems
 关于 initializer的说明为：
 If no initializer is provided, the output structure and dtypes of fn are assumed to be the same as its input; and in this case, the first argument of fn must match the structure of elems.
 If an initializer is provided, then the output of fn must have the same structure as initializer; and the first argument of fn must match this structure.
 也就是说当initializer给定的时候，fn的输出结构必须和initializer保持一致，且fn的第一个参变量也必须和该结构一致。而如果该参数没有给定的时候初始化默认和x[0]的维度保持一致。

```
  设函数为f，
  x = [u(0),u(1),...,u(n)]
  y = tf.scan(f,x,initializer=v(0))
  此时f的参数类型必须是(v(0),x)，f的输出必须和v(0)保持一致，整个计算过程如下：
  v(1)=f(v(0),u(0))
  v(2)=f(v(1),u(1))
  ....
  v(n+1)=f(v(n),u(n))
  y=v(n+1)
```

作者：ClarenceHoo

链接：https://www.jianshu.com/p/7776f771435e

來源：简书

简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。







# tf.train.batch和tf.train.shuffle_batch的理解

2017年06月28日 21:51:30

 

美利坚节度使

 

阅读数：26149

更多

个人分类： [Python](https://blog.csdn.net/ying86615791/article/category/6641267)[Tensorflow](https://blog.csdn.net/ying86615791/article/category/6830117)



版权声明：本文为博主原创文章，未经博主允许不得转载。	https://blog.csdn.net/ying86615791/article/details/73864381

capacity是队列的长度

min_after_dequeue是出队后，队列至少剩下min_after_dequeue个数据

假设现在有个test.tfrecord文件，里面按从小到大顺序存放整数0~100

\1. tf.train.batch是按顺序读取数据，队列中的数据始终是一个有序的队列，

比如队列的capacity=20，开始队列内容为0,1，..,19=>读取10条记录后，队列剩下10,11，..,19，然后又补充10条变成=>10,11,...,29,

队头一直按顺序补充，队尾一直按顺序出队,到了第100条记录后，又重头开始补充0,1,2...

\2. tf.train.shuffle_batch是将队列中数据打乱后，再读取出来，因此队列中剩下的数据也是乱序的，队头也是一直在补充（我猜也是按顺序补充），

比如batch_size=5,capacity=10,min_after_dequeue=5,

初始是有序的0,1，..,9(10条记录)，

然后打乱8,2,6,4,3,7,9,2,0,1(10条记录),

队尾取出5条，剩下7,9,2,0,1(5条记录),

然后又按顺序补充进来，变成7,9,2,0,1,10,11,12,13,14(10条记录)，

再打乱13,10,2,7,0,12...1(10条记录)，

再出队...



capacity可以看成是局部数据的范围，读取的数据是基于这个范围的，

在这个范围内，min_after_dequeue越大，数据越乱

这样按batch读取的话，最后会自动在前面添加一个维度，比如数据的维度是[1],batch_size是10，那么读取出来的shape就是[10,1]





sparse_softmax_cross_entropy_with_logits VS softmax_cross_entropy_with_logits
这两者都是计算分类问题的softmax loss的，所以两者的输出应该是一样的，唯一区别是两者的labels输入形似不一样。

Difference
在tensorflow中使用softmax loss的时候，会发现有两个softmax cross entropy。刚开始很难看出什么差别，结合程序看的时候，就很容易能看出两者差异。总的来说两者都是计算分类问题的softmax交叉熵损失，而两者使用的标签真值的形式不同。

sparse_softmax_cross_entropy_with_logits: 
使用的是实数来表示类别，数据类型为int16，int32，或者 int64，标签大小范围为[0，num_classes-1]，标签的维度为[batch_size]大小。

softmax_cross_entropy_with_logits： 
使用的是one-hot二进制码来表示类别，数据类型为float16，float32，或者float64，维度为[batch_size, num_classes]。这里需要说明一下的时，标签数据类型并不是Bool型的。这是因为实际上在tensorflow中，softmax_cross_entropy_with_logits中的每一个类别是一个概率分布，tensorflow中对该模块的说明中明确指出了这一点，Each row labels[i] must be a valid probability distribution。很显然，one-hot的二进码也可以看是一个有效的概率分布。

另外stackoverflow上面对两者的区别有一个总结说得很清楚，可以参考一下。

Common
有一点需要注意的是，softmax_cross_entropy_with_logits和sparse_softmax_cross_entropy_with_logits中的输入都需要unscaled logits，因为tensorflow内部机制会将其进行归一化操作以提高效率，什么意思呢？就是说计算loss的时候，不要将输出的类别值进行softmax归一化操作，输入就是wT∗X+bwT∗X+b的结果。

tensorflow的说明是这样的： 
Warning: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency. Do not call this op with the output of softmax, as it will produce incorrect results.

至于为什么这样可以提高效率，简单地说就是把unscaled digits输入到softmax loss中在反向传播计算倒数时计算量更少，感兴趣的可以参考pluskid大神的博客Softmax vs. Softmax-Loss: Numerical Stability，博文里面讲得非常清楚了。另外说一下，看了大神的博文，不得不说大神思考问题和解决问题的能力真的很强！

Example
import tensorflow as tf
#batch_size = 2
labels = tf.constant([[0, 0, 0, 1],[0, 1, 0, 0]])
logits = tf.constant([[-3.4, 2.5, -1.2, 5.5],[-3.4, 2.5, -1.2, 5.5]])

loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(labels,1), logits=logits)

with tf.Session() as sess:  
​    print "softmax loss:", sess.run(loss)
​    print "sparse softmax loss:", sess.run(loss_s)
1
2
3
4
5
6
7
8
9
10
11
12
Output: 
softmax loss: [ 0.04988896 3.04988885] 
sparse softmax loss: [ 0.04988896 3.04988885]

Reference
tensorflow:softmax_cross_entropy_with_logits 
tensorflow:sparse_softmax_cross_entropy_with_logits 
stackoverflow 
Softmax vs. Softmax-Loss: Numerical Stability
--------------------- 
作者：蜗牛一步一步往上爬 
来源：CSDN 
原文：https://blog.csdn.net/yc461515457/article/details/77861695 
版权声明：本文为博主原创文章，转载请附上博文链接！





### states = tf.concat(values=states, axis=1)

