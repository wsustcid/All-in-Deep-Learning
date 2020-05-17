'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:15:36
@LastEditTime: 2020-05-15 20:39:37
'''

"""
1. Show how to implement RNN models from scratch (MNIST images as sequences)
2. Visualize the model with the interactive TensorBoard

Final results: test acc: 95.3125
Hint:
1. The updata step for vanilla RNN: ht = (Wt*Xt + W_h*h_t-1 + b)
"""
import os
import tensorflow as tf
import numpy as np

# You have to download mnist manually from http://yann.lecun.com/exdb/mnist/
# and put it in the data_dir folder
from tensorflow.examples.tutorials.mnist import input_data

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "datasets/MNIST")
log_dir = os.path.join(base_dir, "logs/RNN_with_summaries")

element_size = 28 # D dims of each vector in our sequence
time_steps = 28 # T   number of such elemets in a sequence
num_classes = 10 #    one sequence corresponding to one output
batch_size = 128 # N
hidden_layer_size = 128 # H Control the size of hidden RNN state vector

def variable_summaries(var):
    """ Add some ops that take care of logging summaries
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        
        tf.summary.histogram('histogram', var)

def get_pl(time_steps, element_size, num_classes):
    """creat placeholders for inputs and labels"""
    
    inputs = tf.placeholder(tf.float32, 
                            shape=[None,time_steps, element_size],
                            name='inputs')

    y = tf.placeholder(tf.float32, shape=[None,num_classes], name='labels')

    return inputs, y

def _get_zero_variable(name, shape):
    with tf.name_scope(name):
        var = tf.Variable(tf.zeros(shape))
        variable_summaries(var)
    return var

def _get_variable(name, shape, mean=0, stddev=0.01):
    with tf.name_scope(name):
        var = tf.Variable(tf.truncated_normal(shape, mean, stddev))
        variable_summaries(var)
    return var

# 以下两个函数不涉及时间维度,可使用正常FC或Conv操作
# 不能在rnn_step 中创建
# ValueError: Initializer for variable vanilla_rnn/scan/while/rnn_hidden/Wx/Variable/ is from inside a control-flow construct, such as a loop or conditional. When creating a variable inside a loop or conditional, use a lambda as the initializer.

def rnn_step(previous_hidden_state, x):
    """ vanilla RNN step
    """
    # (B,H) * (H,H) + (B,D) * (D,H)
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
    
    return current_hidden_state

def get_linear_layer(hidden_state):
    """ apply linear layer to state vector
    """
    #(B,H) * (H,C) + (C,) = (B,C)
    current_output = tf.matmul(hidden_state, Wl) + bl
    
    return current_output



def get_model(inputs):    
    with tf.name_scope("vanilla_rnn") as scope:
        # (batch_size, time_steps, element_size) -> (time_steps, batch_size, element_size)
        processed_inputs = tf.transpose(inputs, perm=[1,0,2])
        
        initial_hidden = tf.zeros([batch_size, hidden_layer_size])
        global Wx, Wh, b_rnn, Wl, bl
        with tf.name_scope("rnn_hidden") as scope:
            Wx = _get_zero_variable('Wx', shape=[element_size, hidden_layer_size]) # (D,H)
            Wh = _get_zero_variable('Wh', shape=[hidden_layer_size,hidden_layer_size]) # (H,H)
            b_rnn = _get_zero_variable('bias', shape=[hidden_layer_size]) # (H,)

        with tf.name_scope('rnn_linear') as scope:
            Wl = _get_variable('Wl', shape=[hidden_layer_size, num_classes])
            bl = _get_variable('bl', shape=[num_classes])

        # repeatedly applies a rnn_step to a sequence of elements in order
        all_hidden_states = tf.scan(rnn_step, processed_inputs, initializer = initial_hidden, name="states") # (T, B,H)

        # iterate across time, apple linear layer to all RNN outputs
        # 得到所有时刻的输出
        all_outputs = tf.map_fn(get_linear_layer, all_hidden_states) # (T,B,C)
            
        # get last time output
        output = all_outputs[-1] # (B,C)
            
        tf.summary.histogram('outputs', output)
    return output

def get_loss(output,y):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
        tf.summary.scalar('cross_entropy', cross_entropy)
        
    return cross_entropy





def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            mnist = input_data.read_data_sets(data_dir, one_hot=True)
            
            # get a small test set
            test_data = mnist.test.images[:batch_size].reshape((-1,time_steps, element_size))
            test_label = mnist.test.labels[:batch_size]

            inputs, y = get_pl(time_steps,element_size,num_classes)

            output = get_model(inputs)
            
            cross_entropy = get_loss(output,y)

            train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
            accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
            tf.summary.scalar('accuracy', accuracy)


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        merged = tf.summary.merge_all()
        # write summaries to log_dir -- used by tensorboard
        train_writer  = tf.summary.FileWriter(log_dir + '/train',
                                            graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(log_dir + '/test',
                                        graph=tf.get_default_graph())
        
        sess.run(tf.global_variables_initializer())

        for i in range(10000):
            # data comes in unrolled form: a vector of 784 pixels.
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # reshape data to get 28 squences of 28 pixels (N, T ,D)
            batch_x = batch_x.reshape((batch_size, time_steps, element_size))
            
            # train
            summary_train, _ = sess.run([merged, train_step], 
                                        feed_dict={inputs:batch_x, y:batch_y})
            # add to summaries
            train_writer.add_summary(summary_train,i)
            
            if i%1000 == 0:
                acc, loss = sess.run([accuracy, cross_entropy], feed_dict=
                                    {inputs:batch_x, y:batch_y})
                print("iter: {}  loss: {:.6f}  acc: {:.5f}%".format(i, loss, acc))
            
            if i%10 == 0:
                # calculate accuracy for 128 MNIST test images and add to summaries
                summary_test, acc = sess.run([merged, accuracy], feed_dict=
                                    {inputs: test_data, y:test_label})
                test_writer.add_summary(summary_test, i)
            
        test_acc = sess.run([accuracy], feed_dict={inputs:test_data, y:test_label})
        print("test acc: {}".format(test_acc))






def scan_demo():
    """tf.scan example """

    elems = np.array(["T","e","n","s","o","r", " ", "F","l","o","w"])
    scan_sum = tf.scan(lambda a, x: a + x, elems)   
    sess=tf.InteractiveSession()
    sess.run(scan_sum)

if __name__ == '__main__':
    train()

