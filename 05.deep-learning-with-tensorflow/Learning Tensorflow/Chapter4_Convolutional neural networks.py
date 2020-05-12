'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:14:27
@LastEditTime: 2020-05-06 16:14:27
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 1. MNIST
# %% [markdown]
# ### Dropout
# The final element we will need for our model is dropout. This is a regularization trick
# used in order to force the network to distribute the learned representation across all
# the neurons. Dropout “turns off ” a random preset fraction of the units in a layer, by
# setting their values to zero during training. These dropped-out neurons are random
# —different for each computation—forcing the network to learn a representation that
# will work even after the dropout. This process is often thought of as training an
# “ensemble” of multiple networks, thereby increasing generalization. When using the
# network as a classi

# %%
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# %%
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x): 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]]) # number of filters
    return tf.nn.relu(conv2d(input,W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,W) + b


# %%
x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.float32, shape=[None,10])

x_image = tf.reshape(x, [-1,28,28,1])

conv1 = conv_layer(x_image, shape=[5,5,1,32]) # [N,28,28,32]
conv1_pool = max_pool_2x2(conv1) # [N,14,14,32]

conv2 = conv_layer(conv1_pool, shape=[5,5,32,64]) # [N,14,14,64]
conv2_pool = max_pool_2x2(conv2) # [N,7,7,64]

conv2_flat = tf.reshape(conv2_pool,[-1,7*7*64]) # [N,7*7*64]
full_1 = tf.nn.relu(full_layer(conv2_flat, size=1024)) # N*1024

keep_prob = tf.placeholder(tf.float32)
full_1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

yout = full_layer(full_1_drop,size=10) # N*10


# %%
data_dir = 'datasets/MNIST'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yout,labels=y))
learning_rate = 1e-4
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(yout,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# %%
steps = 5000 # 5 epoches with mini-batches of size 50 (50000 train images)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(steps):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y:batch[1], keep_prob:0.5})
        
        if i%100 ==0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y:batch[1], keep_prob:1})
            print("step {}, training accuracy {}".format(i,train_accuracy))
        
        
    X = mnist.test.images.reshape(10,1000,784)
    Y = mnist.test.labels.reshape(10,1000,10)
    test_accuracy = np.mean([sess.run(accuracy, feed_dict={x: X[i], y:Y[i], keep_prob:1.0}) for i in range(10)])

print("test accuracy:{}".format(test_accuracy))

        
        
    

# %% [markdown]
# ## 2. CIFAR10 Dataset

# %%
import tensorflow as tf
import numpy as np
import pickle
import os 
import matplotlib.pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x): 
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]]) # number of filters
    return tf.nn.relu(conv2d(input,W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,W) + b



def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    
    return out

def unpickle(file):
    with open(os.path.join(data_path, file), 'rb') as fo:
        u = pickle._Unpickler(fo)
        u.encoding = 'latin1'
        dict = u.load()
    
    return dict

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)]) 
                    for i in range(size)]
                  )
    
    plt.imshow(im)
    plt.show
    


# We load it all into memory

class CifarLoader(object):
    def __init__(self, source_files):
        self.source = source_files # a list of filenames
        self.i = 0
        self.images = None
        self.labels = None
    
    def load(self):
        data = [unpickle(f) for f in self.source]
        images = np.vstack(d["data"] for d in data)
        labels = np.hstack(d["labels"] for d in data)
        
        n = len(images)
        self.images = images.reshape(n,3,32,32).transpose(0,2,3,1).astype(float) / 255 ## (n,3,32,32)->(n,32,32,3)
        self.labels = one_hot(labels,10)
        
        return self
    
    def next_batch(self, batch_size):
        x, y = self.images[self.i:self.i+batch_size], self.labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.images)
        
        return x, y
    
    def random_batch(self, batch_size):
        n = len(self.images)
        ix = np.random.choice(n,batch_size)
        
        return self.images[ix], self.labels[ix]

class CifarDataManger(object):
    def __init__(self):
        self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1, 6)]).load()
        self.test = CifarLoader(["test_batch"]).load()
    


# %%
def create_cifar_image():
    d = CifarDataManger()
    
    print("Number of train images: {}".format(len(d.train.images)))
    print("Number of train labels: {}".format(len(d.train.labels)))
    
    print("Number of test images: {}".format(len(d.test.images)))
    print("Number of test labels: {}".format(len(d.test.labels)))
    
    images = d.train.images
    display_cifar(images,10)


# %%
def run_simple_net():
    cifar = CifarDataManger()
    
    x = tf.placeholder(tf.float32, shape=[None,32,32,3])
    y = tf.placeholder(tf.float32, shape=[None,10])
    keep_prob = tf.placeholder(tf.float32)
    
    conv1 = conv_layer(x, shape=[5,5,3,32]) # N,32,32,32
    conv1_pool = max_pool_2x2(conv1)
    
    conv2 = conv_layer(conv1_pool,shape=[5,5,32,64]) # N,16,16,64
    conv2_pool = max_pool_2x2(conv2)
    
    conv3 = conv_layer(conv2_pool, shape=[5,5,64,128]) # N,8,8,128
    conv3_pool = max_pool_2x2(conv3)
    
    conv3_flat = tf.reshape(conv3_pool, [-1,4*4*128])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)
    
    full_1 = tf.nn.relu(full_layer(conv3_drop, 512))
    full_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)
    
    yout = full_layer(full_drop,10)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yout,labels=y))
    
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(yout,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    def test(sess):
        X = cifar.test.images.reshape(10,1000,32,32,3)
        Y = cifar.test.labels.reshape(10,1000,10)
        
        acc = np.mean([sess.run(accuracy, feed_dict={x:X[i], y:Y[i], keep_prob:1.0}) for i in range(10)])
        
        print("Accuracy: {:.4}%".format(acc*100))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(steps):
            batch = cifar.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch[0],y:batch[1], keep_prob:0.5})
            
            if i%500 == 0:
                test(sess)
        
        test(sess)

        
    


# %%
def run_smaller_net():
    
    cifar = CifarDataManger()
    
    x = tf.placeholder(tf.float32, shape=[None,32,32,3])
    y = tf.placeholder(tf.float32, shape=[None,10])
    keep_prob = tf.placeholder(tf.float32)
    
    conv1_1 = conv_layer(x,shape=[3,3,3,32])
    conv1_2 = conv_layer(conv1_1,shape=[3,3,32,32])
    conv1_3 = conv_layer(conv1_2,shape=[3,3,32,32])
    conv1_pool = max_pool_2x2(conv1_3) # 16,32
    conv1_drop = tf.nn.dropout(conv1_pool,keep_prob=keep_prob)
    
    conv2_1 = conv_layer(conv1_drop,shape=[3,3,32,64])
    conv2_2 = conv_layer(conv2_1,shape=[3,3,64,64])
    conv2_3 = conv_layer(conv2_2,shape=[3,3,64,64])
    conv2_pool = max_pool_2x2(conv2_3) # 8,64
    conv2_drop = tf.nn.dropout(conv2_pool,keep_prob=keep_prob)
    
    conv3_1 = conv_layer(conv2_drop,shape=[3,3,64,128])
    conv3_2 = conv_layer(conv3_1,shape=[3,3,128,128])
    conv3_3 = conv_layer(conv3_2,shape=[3,3,128,128])
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=(1,8,8,1), strides=(1,8,8,1), padding='SAME') # 1,128
    conv3_flat = tf.reshape(conv3_pool, [-1,128])
    conv3_drop = tf.nn.dropout(conv3_flat,keep_prob=keep_prob)
    
    full1 = tf.nn.relu(full_layer(conv3_drop, 600))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)
    
    yout = full_layer(full1_drop,10)
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yout,labels=y))
    
    train_step = tf.train.AdamOptimizer(learning_rate2).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(yout,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for i in range(steps2):
            batch = cifar.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
            
            if i%500 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
                print("epoch {}, training_acc: {:.4}%".format(i/500, train_accuracy*100))
        
        X = cifar.test.images.reshape(10,1000,32,32,3)
        Y = cifar.test.labels.reshape(10,1000,10)
        
        test_accuracy = np.mean([sess.run(accuracy, feed_dict={x:X[i], y:Y[i], keep_prob:1.0}) for i in range(10)])
        print("test_acc: {}".format(test_accuracy*100))
            
            


# %%
'''
error：AttributeError: 'module' object has no attribute 'to_rgba'
solve:
sudo pip3 install matplotlib==2.2.0
'''

if __name__ == "__main__":
    
    data_path = "datasets/CIFAR10"
    batch_size = 100
    steps = 500000 # 500000/(50000/100)=1000 epochs
    learning_rate = 1e-3
    
    steps2 =10000
    learning_rate2 = 5e-4

    #create_cifar_image()
    
    #run_simple_net()
    
    run_smaller_net()


# %%


