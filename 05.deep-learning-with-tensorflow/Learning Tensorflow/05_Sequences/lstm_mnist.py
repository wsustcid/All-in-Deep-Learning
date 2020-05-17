'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:15:36
@LastEditTime: 2020-05-15 23:17:42
'''

"""
1. Show how to implement RNN models using lstm (MNIST images as sequences)
2. Visualize the model with the interactive TensorBoard

Final results: test acc: [99.21875]
Hint:
1. cell output equals to the hidden state
2. So the state is a convenient tensor that holds the last actual RNN state, ignoring the zeros. The output tensor holds the outputs of all cells, so it doesn't ignore the zeros. 
"""
import os
import tensorflow as tf
import numpy as np

# You have to download mnist manually from http://yann.lecun.com/exdb/mnist/
# and put it in the data_dir folder
from tensorflow.examples.tutorials.mnist import input_data


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "datasets/MNIST")
log_dir = os.path.join(base_dir, "logs/lstm")

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

def _get_variable(name, shape, mean=0, stddev=0.01):
    with tf.name_scope(name):
        var = tf.Variable(tf.truncated_normal(shape, mean, stddev))
        variable_summaries(var)
    return var


def get_linear_layer(hidden_state):
    """ apply linear layer to state vector
    """
    #(B,H) * (H,C) + (C,) = (B,C)
    current_output = tf.matmul(hidden_state, Wl) + bl
    
    return current_output



def get_model(inputs):    
    with tf.name_scope("lstm") as scope:
        
        lstm_cell = tf.contrib.rnn.LSTMCell(hidden_layer_size, forget_bias=1.0) # like rnn_step()
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)  

        global Wl, bl
        with tf.name_scope('rnn_linear') as scope:
            Wl = _get_variable('Wl', shape=[hidden_layer_size, num_classes])
            bl = _get_variable('bl', shape=[num_classes])

        # just get final time output
        final_output = get_linear_layer(state[1]) # 等价于 outputs[:,-1,:]
                
        tf.summary.histogram('outputs', final_output)
    
    return final_output

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

        for i in range(5000):
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

