<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-06-16 09:50:25
 * @LastEditTime: 2020-06-16 11:38:00
 * @Description:  
--> 
# Building sequential model using tensorflow

## Demo Code
**vanilla_rnn_mnist.py**
1. Show how to implement RNN models from scratch (use MNIST images as sequences)
2. The updata step for vanilla RNN: ht = (Wt*Xt + W_h*h_t-1 + b)

Final results: test acc: 95.3125

**build_in_rnn_mnist.py**
1. Show how to implement RNN models using build in rnn cell (MNIST images as sequences)

Final results: test acc: 97.65625

**lstm_mnist.py**
1. Show how to implement RNN models using lstm (MNIST images as sequences)

Final results: test acc: [99.21875]

Hint:
 - cell output equals to the hidden state
 - So the state is a convenient tensor that holds the last actual RNN state, ignoring the zeros. The output tensor holds the outputs of all cells, so it doesn't ignore the zeros. 

**stack_lstm_mnist.py**
1. Show how to implement RNN models using multiple lstm (MNIST images as sequences)

Final results: test acc:[96.875] 过拟合

**rnn_text_sequences.py**
1. Explore how to use an RNN in a supervised text classification problem with word-embedding training. 
2. Show how to build a more advanced RNN model with long short-term memory (LSTM) networks and how to handle sequences of variable length.

```python
print(np.array(output_example).shape)
print("--------------------")
print(np.array(states_example).shape)

(1, 128, 6, 32)  (num_lstm, B,T,D)
--------------------
(1, 128, 32)     for h(output): (num_lstm, B, D)
```

**return_test.py**
通过实验说明lstm中 state与output之间的关系
```python
    0. 问题描述：比方说我们训练语料一共有3句话，每句话有4个词语，每个词语ebedding为5个维度，所以输入数据的 shape=［3，4，5］(B,T,D)；然后，经过一个神经元为10的 cell得到 outputs 和 state

    1. output shape = ［3，4，10］； 使用output[:, -1, :] 取每句话中最后一个时刻（词语）的输出作为下一步的输入，这样，就得到了 3 x 10 的矩阵。

    2. state 是个tuple(c, h): state = LSTMStateTuple(c=array([3,10], dtype=float32),  h=array([3,10], dtype=float32)）; 其中，c(t)是当前更新后的记忆；h(t)当前输出
      - 每句话经过cell后会得到一个最后时刻的state，状态的维度就是隐藏神经元的个数，此时与每句话中包含的词语个数无关，这样，state就只跟训练数据中包含多少句话(batch_size) 和 隐藏神经元个数(hidden size)有关了。
      - 其中 c =[batch_size, hidden_size], h = [batch_size, hidden_size]
      - 我们一般使用h即最后时刻的输出来处理
    3. state 中的 h 跟output 的最后一个时刻的输出是一样的，即：
       output[:, -1, :] = state[1]
```

## Q & A
**Q1: How to build Time Distributed Convolutional Layers in Tensorflow?**
A1: You could try a basic reshape:
1. Take a tensor of shape (?, timeSteps, ...other_dimensions...)
2. Reshape it to (-1,  ...other_dimensions...)
3. Apply the layer/operation
4. Reshape it back to (-1, timeSteps, ...other_dimensions...)

Note: Keras layers are now supported in recent versions of tensor flow. You can use tf.keras.layers.TimeDistributed to accomplish your task.

**训练时注意使用梯度截断：**
```python
# Grad clipping
train_op = tf.train.AdamOptimizer(learning_rate_)
gradients = train_op.compute_gradients(cost) 
capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients] optimizer = train_op.apply_gradients(capped_gradients)
```

## How to work with Time Distributed data in a neural network (Keras)
Finding a jumping cat now needs:
- find the cat on each frame
- then check if the cat movement corresponds to the “jumping” action
So we need to repeat several “cat” detection, and only after that we need to check the “action”, the “movements”.

Time Distributed layer will do that job, it can apply the same transformation for a list of input data. That can work with several inputs type, including images. All convolutions, Pooling, Dense…

As we are working on frames that are chronologically ordered, we want to be able to detect relation from frame to frame in a given time. Because we’ve got inputs that are ordered in time, LSTM is perfect to filter useful values from those inputs. There are commonly 2 possibilities:
 - make convolution or other neural computation before LSTM
 - make the same kind of work after LSTM

To decide which order to choose, you need to think about what you want to filter.
 - For our example, we need to check an object in motion, so we need to search the object before detecting the movement. So, here, we need to make convolutions before LSTM.
 - For another example, as cryptocurrency evolution, you can filter input values in time with LSTM, then make some manipulation on the output to find evidence. In this case, you can usefully connect layers after LSTM block.

### Time Distributed before LSTM
For the previous example, we worked with images that are “inputs”, so that’s obvious to make Time Distributed layers before the LSTM layer because we want to let LSTM work with convoluted images.

The LSTM output should not be a “sequence” (you can, but it’s useless here) — we only need to make some fully connected layers to find predicted “action” that is present on our input frames. So we set return_sequence to False

For images, the following python source is a pseudo example with Time Distributed layers before the LSTM layer:

```python
model = Sequential()
# after having Conv2D...
model.add(
    TimeDistributed(
        Conv2D(64, (3,3), activation='relu'), 
        input_shape=(5, 224, 224, 3) # 5 images...
    )
)
model.add(
    TimeDistributed(
        Conv2D(64, (3,3), activation='relu')
    )
)
# We need to have only one dimension per output
# to insert them to the LSTM layer - Flatten or use Pooling
model.add(
    TimeDistributed(
        GlobalAveragePooling2D() # Or Flatten()
    )
)
# previous layer gives 5 outputs, Keras will make the job
# to configure LSTM inputs shape (5, ...)
model.add(
    LSTM(1024, activation='relu', return_sequences=False)
)
# and then, common Dense layers... Dropout...
# up to you
model.add(Dense(1024, activation='relu'))
model.add(Dropout(.5))
# For example, for 3 outputs classes 
model.add(Dense(3, activation='sigmoid'))
model.compile('adam', loss='categorical_crossentropy')
```
***The Flatten layer is only needed because LSTM shape should have one dimension per input.***

### Time Distributed after LSTM
Goal: manipulate data after having managed the time.

This time, we need to set up LSTM to **produce a sequence**. We can imagine that we want to inject 5 items of 10 values and make some transformation on the sequence that LSTM can produce.
But this time, we want to make a fully connected computation on each filtered element. LSTM is not only a “filter”, but it also keeps each input computation in memory, so we can retrieve them to make our manipulation.
That’s what the return_sequences attribute provides. LSTM will now produce 5 outputs that can be time distributed.
```python
model = Sequential()
# a model with LSTM layers, we are using 5 frames of 
# shape (10, 20)
model.add(
    LSTM(1024, 
        activation='relu',
        return_sequences=True,
        input_shape=(5, 10)
    )
)
# LSTM outputs 5 items 
# that is the correct shape to continue to work.
# We need to get several outputs and make the same
# process on each sequence item:
model.add(TimeDistributed(
    Dense(128, activation='relu')
))
model.add(TimeDistributed(
    Dense(64, activation='relu')
))
# Flatten, then Dense... Dropout...
# note: Flatten should not be time distributed because here,
# we want to have only one dimension for the next layers
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(.5))
# use N outputs
model.add(Dense(N, activation='softmax'))
```

### Transfer Learning in Time Distributed block
```python
import keras
from keras.layers import Dense, LSTM, \
    Flatten, TimeDistributed, Conv2D, Dropout
from keras import Sequential
from keras.applications.vgg16 import VGG16
# create a VGG16 "model", we will use
# image with shape (224, 224, 3)
vgg = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
# do not train first layers, I want to only train
# the 4 last layers (my own choice, up to you)
for layer in vgg.layers[:-4]:
    layer.trainable = False
# create a Sequential model
model = Sequential()
# add vgg model for 5 input images (keeping the right shape
model.add(
    TimeDistributed(vgg, input_shape=(5, 224, 224, 3))
)
# now, flatten on each output to send 5 
# outputs with one dimension to LSTM
model.add(
    TimeDistributed(
        Flatten()
    )
)
model.add(LSTM(256, activation='relu', return_sequences=False))
# finalize with standard Dense, Dropout...
model.add(Dense(64, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(3, activation='softmax'))
model.compile('adam', loss='categorical_crossentropy')

```

Demo: https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
