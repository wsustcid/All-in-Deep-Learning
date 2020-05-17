'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 1970-01-01 08:00:00
@LastEditTime: 2020-05-15 23:47:03
'''

import tensorflow as tf
import numpy as np

def lstm_return():
    """ 本文通过实验说明lstm中 state与output之间的关系
    0. 问题描述：比方说我们训练语料一共有3句话，每句话有4个词语，每个词语ebedding为5个维度，所以输入数据的 shape=［3，4，5］(B,T,D)；然后，经过一个神经元为10的 cell得到 outputs 和 state

    1. output shape = ［3，4，10］； 使用output[:, -1, :] 取每句话中最后一个时刻（词语）的输出作为下一步的输入，这样，就得到了 4 x 10 的矩阵。

    2. state 是个tuple(c, h): state = LSTMStateTuple(c=array([3,10], dtype=float32),  h=array([3,10], dtype=float32)）; 其中，c(t)是当前更新后的记忆；h(t)当前输出
      - 每句话经过cell后会得到一个最后时刻的state，状态的维度就是隐藏神经元的个数，此时与每句话中包含的词语个数无关，这样，state就只跟训练数据中包含多少句话(batch_size) 和 隐藏神经元个数(hidden size)有关了。
      - 其中 c =[batch_size, hidden_size], h = [batch_size, hidden_size]
      - 我们一般使用h即最后时刻的输出来处理
    3. state 中的 h 跟output 的最后一个时刻的输出是一样的，即：
       output[:, -1, :] = state[1]

    """
    batch_size = 3 # 训练语料中一共有4句话
    time_steps = 4 # 每句话只有5个词语
    element_size = 5 # 每个词语的词向量维度为 6
    hidden_size = 10 # 神经元个数为10


    X = np.array([[[1,2,3,4,5],
                   [0,1,2,3,8],
                   [3,6,8,1,2],
                   [2,3,6,4,1]],
            
                  [[2,3,5,6,8],
                   [3,4,5,1,7],
                   [6,5,9,0,2],
                   [2,3,4,6,1]],
            
                  [[2,3,5,1,6],
                  [3,5,2,4,7],
                  [4,5,2,4,1],
                  [3,4,3,2,6]]
                ])
    X = tf.to_float(X)
    
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, X, time_major=False, dtype=tf.float32)
    
    last_output = outputs[:, -1, :]  # 取最后一个时序输出作为结果
    # fc_dense = tf.layers.dense(last, 10, name='fc1')
    # fc_drop = tf.contrib.layers.dropout(fc_dense, 0.8)
    # fc1 = tf.nn.relu(fc_drop)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(last_output))
        print('-------------------------\n')
        print(sess.run(state[1])) # h
        print('-------------------------\n')
        print(sess.run(state[0]))

def get_lstm_cell():
    return tf.nn.rnn_cell.BasicLSTMCell(num_units=10, forget_bias=1.0)

def stack_lstm_return():
    """ 本文通过实验说明多个stacked lstm中 state与output之间的关系
    0. 问题描述：比方说我们训练语料一共有3句话，每句话有4个词语，每个词语ebedding为5个维度，所以输入数据的 shape=［3，4，5］(B,T,D)；然后，经过两个神经元为10的 cell得到 outputs 和 states (多个cell是串联的，所以最后结果也只有一份output和states)

    1. output shape = ［3，4，10］； 使用output[:, -1, :] 取每句话中最后一个时刻（词语）的输出作为下一步的输入，这样，就得到了 4 x 10 的矩阵。

    2. state 是个tuple(c, h): state = LSTMStateTuple(c=array([3,10], dtype=float32),  h=array([3,10], dtype=float32)）; 其中，c(t)是当前更新后的记忆；h(t)当前输出
      - 每句话经过当前cell后会得到一个最后时刻的state，状态的维度就是隐藏神经元的个数，此时与每句话中包含的词语个数无关，这样，state就只跟训练数据中包含多少句话(batch_size) 和 隐藏神经元个数(hidden size)有关了。
      - 其中 c =[batch_size, hidden_size], h = [batch_size, hidden_size]
      - 我们一般使用h即最后时刻的输出来处理
    3. 经过多少个cell,就会有多少个LSTMstatTuple,即每个cell都会输出一个tuple(c,h)
    3. state最后一个tuple 中的 h 跟output 的最后一个时刻的输出是一样的，即：
       output[:, -1, :] = state[-1][1]

    """
    batch_size = 3 # 训练语料中一共有4句话
    time_steps = 4 # 每句话只有5个词语
    element_size = 5 # 每个词语的词向量维度为 6
    hidden_size = 10 # 神经元个数为10
    num_lstm = 2


    X = np.array([[[1,2,3,4,5],
                   [0,1,2,3,8],
                   [3,6,8,1,2],
                   [2,3,6,4,1]],
            
                  [[2,3,5,6,8],
                   [3,4,5,1,7],
                   [6,5,9,0,2],
                   [2,3,4,6,1]],
            
                  [[2,3,5,1,6],
                  [3,5,2,4,7],
                  [4,5,2,4,1],
                  [3,4,3,2,6]]
                ])
    X = tf.to_float(X)
    
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([get_lstm_cell() for _ in range(num_lstm)], state_is_tuple=True)
    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, X, time_major=False, dtype=tf.float32)
    
    last_output = outputs[:, -1, :]  # 取最后一个时序输出作为结果
    # fc_dense = tf.layers.dense(last, 10, name='fc1')
    # fc_drop = tf.contrib.layers.dropout(fc_dense, 0.8)
    # fc1 = tf.nn.relu(fc_drop)
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(last_output))
        print('-------------------------\n')
        print(sess.run(state[-1][1])) # h
        print('-------------------------\n')
        print(sess.run(state[-1][0]))



if __name__ == '__main__':
    #lstm_return()
    stack_lstm_return()