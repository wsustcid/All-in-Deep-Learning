在机器学习中，常常会出现overfitting，网络权值越大往往overfitting的程度越高，因此，为了避免出现overfitting,会给误差函数添加一个惩罚项，常用的惩罚项是所有权重的平方乘以一个衰减常量之和。

![img](https://img-blog.csdn.net/20170322092158987?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)



右边项即用来惩罚大权值。权值衰减惩罚项使得权值收敛到较小的绝对值，而惩罚大的权值。从而避免overfitting的出现。



假设我们原来的损失函数没有weight decay项，设为E(w)，这种情况下的权值更新如下：

![img](https://img-blog.csdn.net/20170322093341992?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

带有weight decay项后，损失函数变为：

![img](https://img-blog.csdn.net/20170322093439165?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

此时的更新函数为：

![img](https://img-blog.csdn.net/20170322093534730?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

这样则会使权重衰减。



![img](https://img-blog.csdn.net/20170322093804916?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![img](https://img-blog.csdn.net/20170322093826593?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

![img](https://img-blog.csdn.net/20170322093849198?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMTk5MTgzNzM=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

# L2正则=Weight Decay？并不是这样

文章链接是[https://arxiv.org/pdf/1711.05101.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1711.05101.pdf)。

在训练神经网络的时候，由于Adam有着收敛快的特点被广泛使用。但是在很多数据集上的最好效果还是用SGD with Momentum细调出来的。可见Adam的泛化性并不如SGD with Momentum。在这篇文章中指出了Adam泛化性能差的一个重要原因就是Adam中L2正则项并不像在SGD中那么有效，并且通过Weight Decay的原始定义去修正了这个问题。文章表达了几个观点比较有意思。

一、L2正则和Weight Decay并不等价。这两者常常被大家混为一谈。首先两者的目的都是想是使得模型权重接近于0。

- L2正则是在损失函数的基础上增加L2 norm， 即为![f_{t}^{reg}(x_{t})=f_{t}(x_t)+\frac{w}{2}||x||_{2}^{2}](https://www.zhihu.com/equation?tex=f_%7Bt%7D%5E%7Breg%7D%28x_%7Bt%7D%29%3Df_%7Bt%7D%28x_t%29%2B%5Cfrac%7Bw%7D%7B2%7D%7C%7Cx%7C%7C_%7B2%7D%5E%7B2%7D) 。
- 而权重衰减则是在梯度更新时直接增加一项， ![x_{t+1}=(1-w)x_t-\alpha \nabla f_t(x_t) ](https://www.zhihu.com/equation?tex=x_%7Bt%2B1%7D%3D%281-w%29x_t-%5Calpha+%5Cnabla+f_t%28x_t%29+) 。
- 在标准SGD的情况下，通过对衰减系数做变换，可以将L2正则和Weight Decay看做一样。但是在Adam这种自适应学习率算法中两者并不等价。

二、使用Adam优化带L2正则的损失并不有效。如果引入L2正则项，在计算梯度的时候会加上对正则项求梯度的结果。那么如果本身比较大的一些权重对应的梯度也会比较大，由于Adam计算步骤中减去项会有除以梯度平方的累积，使得减去项偏小。按常理说，越大的权重应该惩罚越大，但是在Adam并不是这样。而权重衰减对所有的权重都是采用相同的系数进行更新，越大的权重显然惩罚越大。在常见的深度学习库中只提供了L2正则，并没有提供权重衰减的实现。这可能就是导致Adam跑出来的很多效果相对SGD with Momentum偏差的一个原因。

三、下图中的绿色部分就是在Adam中正确引入Weight Decay的方式，称作AdamW。

![img](https://pic1.zhimg.com/80/v2-411fcb8b515ba506e1e8abd0c0bee134_hd.jpg)



我们可以自己实现AdamW，在Adam更新后的参数基础上继续做一次更新。

```python
weights_var = tf.trainable_variables()
gradients = tf.gradients(loss, weights_var)
optimizer = tf.train.AdamOptimizer(learning_rate=deep_learning_rate)
train_op = optimizer.apply_gradients(zip(gradients, weights_var))
# weight decay operation
with tf.control_dependencies([train_op]):
  l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in weights_var])
  sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
  decay_op = sgd.minimize(l2_loss)
```

不过tensorflow上已有AdamW修正，在tensorflow[1.10.0-rc0](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/tensorflow/releases/tag/v1.10.0-rc0)中也包含了这个feature，但还没正式release，按照tensorflow的更新速度，应该很快了。可以像下面直接使用。

```python
#optimizer = tf.train.AdamOptimizer(learning_rate=deep_learning_rate)
AdamWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
optimizer = AdamWOptimizer(weight_decay=weight_decay, learning_rate=deep_learning_rate)
```

具体实现的细节可以参考一下两个函数。

```python
  def _decay_weights_op(self, var):
    if not self._decay_var_list or var in self._decay_var_list:
      return var.assign_sub(self._weight_decay * var, self._use_locking)
    return control_flow_ops.no_op()
  def _apply_dense(self, grad, var):
    with ops.control_dependencies([self._decay_weights_op(var)]):
      return super(DecoupledWeightDecayExtension, self)._apply_dense(grad, var)
```



发布于 2018-07-29