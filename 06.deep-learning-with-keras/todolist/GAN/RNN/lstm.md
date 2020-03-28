## 一、介绍

## 1.1 LSTM介绍

LSTM全称*Long Short-Term Memory*，是1997年就被发明出来的算法，作者是谁说了你们也记不住干脆就不说了（主要是我记不住，逃...）

经过这么多年的发展，基本上没有什么理论创新，唯一值得说的一点也就是加入了Dropout来对抗过拟合。真的是应了那句话呀：

> Deep learning is an art more than a science.

怪不得学数学的一直看不起搞算法的，不怪人家，整天拿着个梯度下降搞来搞去，确实有点low。。。 即使这样，LSTM的应用依旧非常的广泛，而且效果还不错。但是，LSTM的原理稍显复杂，苦于没有找到非常好的资料，小编之前也是被各种博客绕的团团转，今天重新梳理了一次，发现并没有那么难，这里把总结的资料分享给大家。

认真阅读本文，你将学到：

> \1. RNN原理、应用背景、缺点
> \2. LSTM产生原因、原理，以及关于LSTM各种“门”的一些intuition（哲学解释） （别怕，包教包会）
> \3. 如何利用Keras使用LSTM来解决实际问题
> \4. 关于Recurrent Network的一些常用技巧，包括：过拟合，stack rnn



## 1.2 应用背景

Recurrent network的应用主要如下两部分：

1. 文本相关。主要应用于自然语言处理（NLP）、对话系统、情感分析、机器翻译等等领域，Google翻译用的就是一个7-8层的LSTM模型。
2. 时序相关。就是时序预测问题（timeseries），诸如预测天气、温度、包括个人认为根本不可行的但是很多人依旧在做的预测股票价格问题

这些问题都有一个共同点，就是**有先后顺序的概念**的。举个例子： 根据前5天每个小时的温度，来预测接下来1个小时的温度。典型的时序问题，温度是从5天前，一小时一小时的记录到现在的，它们的顺序不能改变，否则含义就发生了变化；再比如情感分析中，判断一个人写的一篇文章或者说的一句话，它是积极地（positive），还是消极的（negative），这个人说的话写的文章，里面每个字都是有顺序的，不能随意改变，否则含义就不同了。

全连接网络Fully-Connected Network，或者卷积神经网络Convnet，他们在处理一个sequence（比如一个人写的一条影评），或者一个timeseries of data points（比如连续1个月记录的温度）的时候，他们**缺乏记忆**。一条影评里的每一个字经过word embedding后，被当成了一个独立的个体输入到网络中；网络不清楚之前的，或者之后的文字是什么。这样的网络，我们称为**feedforward network**。

但是实际情况，我们理解一段文字的信息的时候，每个文字并不是独立的，我们的脑海里也有它的上下文。比如当你看到这段文字的时候，你还记得这篇文章开头表达过一些关于LSTM的信息；

所以，我们在脑海里维护一些信息，这些信息随着我们的阅读不断的更新，帮助我们来理解我们所看到的每一个字，每一句话。这就是RNN的做法：**维护一些中间状态信息**。

## 二、SimpleRNN

## 2.1 原理

RNN是**Recurrent Neural Network**的缩写，它就是实现了我们来维护中间信息，记录之前看到信息这样最简单的一个概念的模型。

关于名称，你可以这样理解：`Recurrent Neural Network = A network with a loop`. 如图：

![img](https://pic2.zhimg.com/80/v2-710ee6443547973d5dce1921547ebb29_hd.jpg)



为了更清楚的说明*loop*和*state*，我们来实现一个简单的toy-rnn。输入是2维的`(timesteps, input_features)`. 这里的*loop*就是在*timesteps*上的*loop*：每一个时刻t，RNN会考虑当前时刻t 的状态*state*，以及当前时刻t 的输入(维度是`(input_features,)`),然后总和得到在时刻t的输出。并且为当前时刻t的输出去更新状态*state*。但是最初的时刻，没有上一个时刻的输出，所以state会被全初始化为0，叫做*initial state of the network.*

代码如下：

```python
state_t = 0 #时刻t的状态
for input_t in input_sequence: # 在timesteps上loop
    output_t = f(input_t, state_t) # input_t state_t得到时刻t输出
    state_t = output_t # 用当前输出去更新内部状态
```

`f`是一个函数，它完成从input和state到output的转换，通常包含两个矩阵`W, U`和一个偏置向量`b`，然后再经过激活函数激活。形式如下：

```
f = activation(dot(W, input) + dot(U, state) + b)
```

非常类似DNN中的全连接层。

还不明白看代码：

```python
# SimpleRNN in numpy
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random(shape=(timesteps, input_features))

state_t = np.zeros(shape=(output_features,)) # init state

W = np.random.random(shape=(output_features, input_features))
U = np.random.random(shape=(output_features, output_features))
b = np.random.random(shape=(output_features,))

successive_outputs = []

for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) #input_t state_t => output_t

    successive_outputs.append(output_t)

    state_t = output_t  # update state_t using output_t

final_outputs = np.concatenate(successive_outputs, axis=0) #get the final_output with shape=(timesteps, output_features)
```

所以，RNN其实就是在时间上的一个循环，每次循环都会用到上一次计算的结果，就这么简单。在时间上，把RNN展开如下图：

![img](https://pic4.zhimg.com/80/v2-7a334d42ef9a1b89b3b505d6983e751b_hd.jpg)



关于输出，虽然RNN每个时刻t都会有输出，但是最后时刻的输出实际上已经包含了之前所有时刻的信息，所以一般我们只保留最后一个时刻的输出就够了。

## 2.2 优缺点

1. 优点。处理a sequence或者a timeseries of data points效果比普通的DNN要好。中间状态理论上维护了从开头到现在的所有信息；
2. 缺点。不能处理long sequence/timeseries问题。原因是**梯度消失**，网络几乎不可训练。所以也只是理论上可以记忆任意长的序列。

## 三、LSTM

> LSTM就是用来解决RNN中梯度消失问题的，从而可以处理long-term sequences。

## 3.1 原理

LSTM是SimpleRNN的变体，它解决了梯度消失的问题。怎么解决的那？

LSTM增加了一个可以相隔多个timesteps来传递信息的方法。想想有一个传送带在你处理sequences时一起运转。每个时间节点的信息都可以放到传送带上，或者从传送带上拿下来，当然你也可以更新传送带上的信息。这样就保存了很久之前的信息，防止了信息的丢失。我们把SimpleRNN中的矩阵记为`Wo Uo bo`，LSTM的结构图如下：

![img](https://pic4.zhimg.com/80/v2-31da92629c2ddbb0a3971d18f1592b03_hd.jpg)



我们在SimpleRNN基础上，增加一条传送带（adding a carry track）用来传递信息。传送带上每个时刻的状态我们记为：`c t` c是carry的意思。

显然，当前时刻的输出就应该收到三个信息的影响：当前时刻的输入、当前时刻的状态、传送带上带来的很久以前的信息。如下：

```
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + dot(C_t, Vo) + bo)
```

这里的处理方式和SimpleRNN是一样的，都是矩阵相乘，矩阵相加，在经过激活函数的操作。

其实当前时刻t的输出就解释清楚了。还有一个问题就是两个状态怎么更新那：state_t, C_t.

1. RNN内部的状态state_t还是跟之前一样：用上一个时刻的输出来更新。
2. 传送带上的状态更新就是LSTM的重点了，也是复杂的地方

根据`input_t, state_t`以及三套**不同的**`W U b`，来计算出三个值：

```text
i_t = activation(dot(state_t, Ui) + dot(input_t, Wi)+ bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)
```

然后组合这三个值来更新`C_t`： `c_t+1 = i_t * k_t + c_t * f_t`

用图表示如下：

![img](https://pic2.zhimg.com/80/v2-4dba3f7fc26068d9089b16c1fbfc6295_hd.jpg)



也不是那么复杂嘛，对吧？还不理解看看下节的*Intuition*来帮助你更好的理解，为什么要这样更新c_t.

## 3.2 Intuition

这一节解释下为什么要这样更新c_t, 帮你建立一些*Intuition*，或者说一些哲学的解释，为什么要这样做。

还记得开篇说的么？

> Deep Learning is an art more than a science. 建立关于DL模型的一些*Intuition*对于算法工程师是非常重要的。打球还有球感那，搞DL没有点intuition都不敢说是这一行的。

你可以这样理解上面的操作： 1. `c_t * f_t` 是为了让模型忘记一些不相关的信息，在carry dataflow的时候。即时是很久之前的信息，模型也有不用他的选择权利，所以模型要有忘记不相关信息的能力。 这也就是常说的遗忘门（我觉得翻译成中文真的很没意思，因为中文的“门”意思是在是太多了，你懂得）。 2. `i_t * k_t` 为模型提供关于当前时刻的信息，给carry track增加一些新的信息。

所以，一个忘记不相关信息，一个增加新的信息，然后再carry track向下传递这个c_t, LSTM真的没那么复杂，如果你觉得很复杂，就是资料没找对。

> 补充一点，上面这样解释只是为了帮助大家理解，基本上所有的LSTM资料也都是这样解释的。但是，当模型真的训练完成后，所谓的模型就是存储下来的W U b矩阵里面的系数。这些系数到底是不是跟我们想的一样的那？没有人知道，也许一样，也许不一样，也许某些问题是这样的，也许某些问题不是这样的。 要不说DL是一门艺术那，没有严谨的科学证明，很多时候只是实际应用后发现效果好，大家就觉得是在朝着正确的方向发展。仔细想想，有点可怕，万一方向错了那？就拿BP算法来说，人类大脑学习记忆，并没有什么反向传播吧。。。

## 3.3 优缺点

1. 优点。解决了SimpleRNN梯度消失的问题，可以处理long-term sequence
2. 缺点。计算复杂度高，想想谷歌翻译也只是7-8层LSTM就知道了；自己跑代码也有明显的感觉，比较慢。

## 四、最佳实践指南

## 4.1 RNN表达能力

有的时候RNN的表达能力有限，为了增加RNN的表达能力，我们可以**stack rnn layers**来增加其表达能力。希望大家了解这是一种常用的做法。 当然了，中间的RNN layer必须把每个时刻的输出都记录下来，作为后面RNN层的输入。实践环节我们会给出例子。

## 4.2 过拟合

RNN LSTM同样会过拟合。这个问题直到2015年，在博士Yarin Gal在他的博士论文里给出了解决办法：类似dropout。但是在整个timesteps上使用同一个固定的drop mask。

博士大佬发了论文，还帮助keras实现了这一举动，我们只需要设置参数`dropout, recurrent_dropout`就可以了，前者是对输入的drop_rate,后者是对recurrent connection的drop_rate。recurrent_connection就是stata_t输入到SimpleRNN中的部分。

话说，为啥人家都这么优秀那，嗑盐厉害，工程也这么厉害，是真的秀

## 4.3 GRU

LSTM的计算比较慢，所有有了Gated Recurrent Unit（GRU），你可以认为他是经过特殊优化提速的LSTM，但是他的表达能力也是受到限制的。

实际使用的时候，从LSTM和GRU中选一个就行了，SimpleRNN太简单了，一般不会使用。

## 4.4 1D-Convnet

另外一种处理sequence或者timeseries问题的方法就是使用1维的卷积网络，并且跟上1维度的池化层。卷积或者池化的维度就是timestep的维度。它可以学习到一些local pattern，视它window大小而定。

优点就是简单，计算相比于LSTM要快很多，所以一种常用的做法就是：

1. 用1D-Convnet来处理简单的文本问题。
2. 把它和LSTM融合，利用1D-Conv轻量级，计算快的优点来得到低维度特征，然后再用LSTM进行学习。这对于处理long sequence非常有用，值得尝试。

## 五、SimpleRNN与LSTM实践

之前写代码一直在用Tensorflow，但是tf的API设计是真的即反人类又反智，要多难用有多难用。针对这个问题，**keras**应运而生，API十分的舒服，省去了很多不必要的代码，谷歌也一是到了这个问题，所以在最新的tensorflow中已经提供了keras的高阶API，主要的应用是快速实验模型。 这里我们采用keras来实战，有兴趣的可以和tensorflow进行对比下。

我们用imdb的数据来进行实践，这是一个二分类问题，判断review是positive还是negtive的。输出文本分类或者情感预测的范畴。实验分为三部分：

1. SimpleRNN
2. LSTM

另外，给两个示例代码： 1. Stack of SimpleRNN 2. Dropout for RNN

> 完整代码参考我的github： [https://github.com/gutouyu/ML_CIA/tree/master/LSTM](https://link.zhihu.com/?target=https%3A//github.com/gutouyu/ML_CIA/tree/master/LSTM) **看不看不关键，关键是记得star （手动抱拳）**

## 5.1 SimpleRNN

代码真的超级简单没有没。完整代码参考上面github。

```python
5.1 SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras import datasets
from keras.preprocessing import sequence


max_features = 10000 # 我们只考虑最常用的10k词汇
maxlen = 500 # 每个评论我们只考虑100个单词

(x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=max_features)
print(len(x_train), len(x_test))

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen) #长了就截断，短了就补0


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

plot_acc_and_loss(history)
rets = model.evaluate(x_test, y_test)
print(rets)
```

结果截图： validation的acc大约能到85%左右。我们只用了500个word并没有使用全部的word，而且SimpleRNN并不太适合处理long sequences。期待LSTM能有更好的表现。

![img](https://pic2.zhimg.com/80/v2-f74a8ee9c24ad3c804cd80d9c3eb44a1_hd.jpg)



## 5.2 LSTM

只用把模型这一部分换了就行了：

```python
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

运行结果： 验证集结果在89%，相比之前的SimpleRNN的85%提升效果显著。

![img](https://pic2.zhimg.com/80/v2-3a1880457f1c81622d5f17a73fb895b5_hd.jpg)



## 5.3 Stack RNN

上面的模型已经是过拟合了，所以模型的表达能力是够的，这里只是给大家参考下如何stack RNN

```python
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(1, activation='sigmoid'))
```

## 5.4 Dropout for RNN

上面的模型已经过拟合了，大家可以参考下面的代码增加Dropout来调整；需要注意点的是，dropout会降低模型的表达能力，所以可以尝试再stack几层rnn。 dropout同样适用于lstm layer，留给大家自己去尝试吧。

```python
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(64, dropout=0.1, recurrent_constraint=0.5, return_sequences=True))
model.add(SimpleRNN(32, dropout=0.1, recurrent_constraint=0.5))
model.add(Dense(1, activation='sigmoid'))
```

## 六、总结

1. LSTM关键在于增加了carry track，稍微复杂一点的在于carry track上c_t信息的更新
2. Recurrent Neural Network适合sequence或timeseries问题
3. keras的API非常的人性化，如果是学习或者做实验建议使用keras，而且tf现在也已经内置keras api 可以通过`from tensorflow import keras`来使用
4. keras内置SimpleRNN, LSTM, GRU，同时还可以使用1D-Conv来处理sequence或timeseries问题
5. 可以给stack RNN来增加模型的表达能力
6. 可以使用dropout来对抗RNN的过拟合

## Reference

1. Deep Learning with python