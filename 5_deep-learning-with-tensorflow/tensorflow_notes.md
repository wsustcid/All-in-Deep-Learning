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

