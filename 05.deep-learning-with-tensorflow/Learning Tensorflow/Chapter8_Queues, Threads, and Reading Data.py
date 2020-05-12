'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:18:09
@LastEditTime: 2020-05-06 16:18:09
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## The Input Pipeline
# When dealing with small datasets that can be stored in memory, such as MNIST
# images, it is reasonable to simply load all data into memory, then use feeding to push
# data into a TensorFlow graph. For larger datasets, however, this can become
# unwieldy. A natural paradigm for handling such cases is to keep the data on disk and
# load chunks of it as needed (such as mini-batches for training), such that the only
# limit is the size of your hard drive.
# 
# 
# %% [markdown]
# ## TFRecords
# A TFRecord file is simply a binary file, con‐
# taining serialized input data. Serialization is based on protocol buffers (proto‐
# bufs), which in plain words convert data for storage by using a schema describing the
# data structure, independently of what platform or language is being used (much like
# XML).

# %%
# First, we download the MNIST data to save_dir
# using a utility function from tensor flow.contrib.learn

from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist

import threading
import time


# %%
save_dir ="datasets/mnist"

# download data to save_dir and read it
data_sets = mnist.read_data_sets(save_dir, dtype=tf.uint8, 
                                  reshape=False, validation_size=1000)


# %%
print(data_sets)
print(len(data_sets))
print("  ")
print(data_sets[0].images.shape)
print(data_sets[0].labels.shape)
print(data_sets[1].images.shape)
print(data_sets[1].labels.shape)
print(data_sets[2].images.shape)
print(data_sets[2].labels.shape)
print("  ")

print(data_sets[0].images[58999,27,27,0])# numpy array (59000,28,28,1)
print(data_sets[0].labels[58999])


# %%
data_splits = ["train", "validation", "test"]
for d in range(len(data_splits)):
    print("saving " + data_splits[d])
    data_set = data_sets[d]
    
    # instantiate a TFRecordWriter object, 
    # giving it the path corresponding to the data split
    filename = os.path.join(save_dir, data_splits[d] + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    
    for index in range(data_set.images.shape[0]):
        # go over each image, converting it from a NumPy array to a byte string
        #转化自己数据的时候可以直接一张张图片读取，不用一次读成4维的矩阵
        image = data_set.images[index].tostring()
        
        '''
        Next, we convert images to their protobuf format. tf.train.Example is a structure
        for storing our data. An Example object contains a Features object, which in turn
        contains a map from attribute name to a Feature . A Feature can contain an
        Int64List , a BytesList , or a FloatList (not used here).
        '''
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[image]))
        }))
        
        writer.write(example.SerializeToString())
        
    writer.close()
        
        


# %%
'''
Let’s take a look at what our saved data looks like. We do this with
tf.python_io.tf_record_iterator , an iterator that reads records from 
a TFRecords file
'''
filename = os.path.join(save_dir, 'train.tfrecords')
record_iterator = tf.python_io.tf_record_iterator(filename)
seralized_img_example = next(record_iterator)

'''
serialized_img is a byte string. To recover the structure we used when saving the
image to a TFRecord, we parse this byte string, allowing us to access all the attributes
we stored earlier:
'''

example = tf.train.Example()
example.ParseFromString(seralized_img_example)

image = example.features.feature['image_raw'].bytes_list.value
label = example.features.feature['label'].int64_list.value[0]
width = example.features.feature['width'].int64_list.value[0]
height = example.features.feature['height'].int64_list.value[0]

'''
Our image was saved as a byte string too, so we convert it back to a NumPy array and
reshape it back to a tensor with shape (28,28,1):
'''
img_flat = np.fromstring(image[0], dtype=np.uint8)
img_reshaped = img_flat.reshape((height,width,-1))
img_reshaped.shape

# %% [markdown]
# ## Queues
# A TensorFlow queue is similar to an ordinary queue, allowing us to enqueue new
# items, dequeue existing items, etc. The important difference from ordinary queues is
# that, just like anything else in TensorFlow, the queue is part of a computational graph.
# Its operations are symbolic as usual, and other nodes in the graph can alter its state
# (much like with Variables). This can be slightly confusing at first, so let’s walk through
# some examples to get acquainted with basic queue functionalities.
# ### Enqueuing and Dequeuing
# Here we create a first-in, first-out (FIFO) queue of strings, with a maximal number of
# 10 elements that can be stored in the queue. Since queues are part of a computational
# graph, they are run within a session. In this example, we use a tf.InteractiveSes
# sion() :

# %%
sess= tf.InteractiveSession()


# %%
queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string])
# Behind the scenes, TensorFlow creates a memory buffer for storing the 10 items.

# Just like any other operation in TensorFlow, 
# to add items to the queue, we create an op:
enque_op = queue1.enqueue(["F"])

# before running the op
sess.run(queue1.size())


# %%
# After running the op, our queue now has one item populating it:
enque_op.run()
sess.run(queue1.size())


# %%
# let's add some more items to queue1, and look at its sieze again:
enque_op = queue1.enqueue(["I"])
enque_op.run()
enque_op = queue1.enqueue(["F"])
enque_op.run()
enque_op = queue1.enqueue(["O"])
enque_op.run()

sess.run(queue1.size())


# %%
# Next, we dequeue items. Dequeuing too is an op, 
# whose output evaluates to a tensor corresponding to the dequeued item:

x = queue1.dequeue()
x.eval()

'''
Note that if we were to run xs.eval() one more time, on an empty queue, our main
thread would hang forever. As we will see later in this chapter, in practice we use code
that knows when to stop dequeuing and avoid hanging.
'''

# %% [markdown]
# '''
# Another way to dequeue is by retrieving multiple items at once, with the
# dequeue_many() operation. This op requires that we specify the shape of items in
# advance:
# '''
# queue1 = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
# 
# inputs = queue1.dequeue_many(3)
# inputs.eval()
# %% [markdown]
# ## Multithreading
# A TensorFlow session is multithreaded—multiple threads can use the same session
# and run ops in parallel. Individual ops have parallel implementations that are used by
# default with multiple CPU cores or GPU threads. However, if a single call to
# sess.run() does not make full use of the available resources, one can increase
# throughput by making multiple parallel calls. For example, in a typical scenario, we
# may have multiple threads apply pre-processing to images and push them into a
# queue, while another thread pulls pre-processed images from the queue for training
# (in the next chapter, we will discuss distributed training, which is conceptually
# related, with important differences).

# %%
'''
Note, again, that the enque op does not actually add the random numbers to the
queue (and they are not yet generated) prior to graph execution. Items will be
enqueued using the function add() we create that adds 10 items to the queue by call‐
ing sess.run() multiple times.
'''
#import threading
#import time

gen_random_normal = tf.random_normal(shape=())
queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=())
enque = queue.enqueue(gen_random_normal)

def add():
   for i in range(10):
       sess.run(enque)


# %%
'''
Next, we create 10 threads, each running add() in parallel, thus each pushing 10
items to the queue, asynchronously. We could think (for now) of these random num‐
bers as training data being added into a queue:
'''

threads = [threading.Thread(target=add, args=()) for i in range(10)]

threads


# %%
'''
We have created a list of threads, and now we execute them, printing the size of the
queue at short intervals as it grows from 0 to 100:
'''
for t in threads:
    t.start()

print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))


# %%
x = queue.dequeue_many(10)
print(x.eval())
sess.run(queue.size())

# %% [markdown]
# ## Coordinator and QueueRunner
# In realistic scenarios (as we shall see later in this chapter), it can be more complicated
# to run multiple threads effectively. Threads should be able to stop properly (to avoid
# “zombie” threads, for example, or to close all threads together when one fails), queues
# need to be closed after stopping, and there are other technical but important issues
# that need to be addressed.
# 
# TensorFlow comes equipped with tools to help us in this process. Key among them
# are tf.train.Coordinator , for coordinating the termination of a set of threads, and
# tf.train.QueueRunner , which streamlines the process of getting multiple threads to
# enqueue data with seamless cooperation.

# %%
gen_random_normal = tf.random_normal(shape=())

queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32], shapes=[])
enque = queue.enqueue(gen_random_normal)

def add(coord, i):
    # Any thread can call coord.request_stop() to get all other threads to stop.
    while not coord.should_stop():
        sess.run(enque)
        if i == 11:
            coord.should_stop()

coord = tf.train.Coordinator()

threads = [threading.Thread(target=add, args=(coord,i)) for i in range(10)] 

coord.join(threads)

#启动时顺序启动，但一旦启动便是并行运行？
for t in threads:
    t.start()
    
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))
time.sleep(0.01)
print(sess.run(queue.size()))

# %% [markdown]
# While we can create a number of threads that repeatedly run an enqueue op, it is bet‐
# ter practice to use the built-in tf.train.QueueRunner , which does exactly that, while
# closing the queue upon an exception.

# %%
'''
In this example, we used a tf.RandomShuffleQueue rather than the FIFO queue. A
RandomShuffleQueue is simply a queue with a dequeue op that pops items in random
order. This is useful when training deep neural networks with stochastic gradient-
descent optimization, which requires shuffling the data. 
The min_after_dequeue argument specifies the minimum number of items that will remain in the queue after
calling a dequeue op—a bigger number entails better mixing (random sampling), but
more memory.
'''

# Here we create a queue runner that will run 
# four threads in parallel to enqueue items:

gen_random_normal = tf.random_normal(shape=())
queue = tf.RandomShuffleQueue(capacity=100, dtypes=[tf.float32], min_after_dequeue=1)

enqueue_op = queue.enqueue(gen_random_normal)

queue_run = tf.train.QueueRunner(queue, [enqueue_op]*4)
coord = tf.train.Coordinator()

enqueue_threads = queue_run.create_threads(sess, coord=coord, start=True)

coord.request_stop()

coord.join(enqueue_threads)

print(sess.run(queue.size()))

# %% [markdown]
# ## A Full Multithreaded Input Pipeline
# We now put all the pieces together in a working example with MNIST images, from
# writing data to TensorFlow’s efficient file format, through data loading and pre-
# processing, to training a model. We do so by building on the queuing and multi‐
# threading functionality demonstrated earlier, and along the way introduce some more
# useful components for reading and processing data in TensorFlow.

# %%
# First, we write the MNIST data to TFRecords
from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np


# %%
###### Write TFrecords #####
save_dir = "datasets/mnist"

data_sets = mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)

data_splits = ["train", "validation", "test"]

for d in range(len(data_splits)):
    print("saving" + data_splits[d])
    
    data_set = data_sets[d]
    
    filename = os.path.join(save_dir, data_splits[d]+'.tfrecords')
    
    writer = tf.python_io.TFRecordWriter(filename)
    
    for index in range(data_set.images.shape[0]):
        image = data_set.images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[1]])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[2]])),
            'depth': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[data_set.images.shape[3]])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(
                value=[int(data_set.labels[index])])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())
    
    writer.close()


# %%
##### Read #####
# tells string_input_producer() to produce each filename
# string num_epochs times.
num_epochs = 10

filename = os.path.join(save_dir, "train.tfrecords")

'''
simply creates a QueueRunner behind the scenes, outputting 
filename strings to a queue for our input pipeline. This 
filename queue will be shared among multiple threads
'''
filename_queue = tf.train.string_input_producer(
    [filename], num_epochs=num_epochs)

'''
Next, we read files from this queue using TFRecordReader() , which takes a queue of
filenames and dequeues filename by filename off the filename_queue . Inter‐
nally, TFRecordReader() uses the state of the graph to keep track of the location of
the TFRecord being read, as it loads “chunk after chunk” of input data from the disk:
'''
reader = tf.TFRecordReader()

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={ 'image_raw': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64)
    })

image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([784])


image = tf.cast(image, tf.float32) * (1./255) - 0.5

label = tf.cast(features['label'], tf.int32)


# shuffle the example + batch

'''
shuffle the image instances and collect them into
batch_size batches with tf.train.shuffle_batch() , which internally uses a Random
ShuffleQueue and accumulates examples until it contains batch_size +
min_after_dequeue elements:

The mini-batches that are returned by shuffle_batch() are the
result of a dequeue_many() call on the RandomShuffleQueue that is created internally.
'''
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=128, 
    capacity=2000, 
    min_after_dequeue=1000)


# %%
images_batch.shape


# %%
#### We define our simple softmax classification model as follows:
W = tf.get_variable("W", [28*28,10])
y_pred = tf.matmul(images_batch, W)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=y_pred, labels=label_batch)

loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

init = tf.local_variables_initializer()
sess.run(init)


# %%
'''
Finally, we create threads that enqueue data to queues by calling
tf.train.start_queue_runners() . Unlike other calls, this one is not symbolic and
actually creates the threads (and thus needs to be done after initialization):
'''

##### coordinator ####
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# %%
threads


# %%
'''
Having everything in place, we are now ready to run the multithreaded process, from
reading and pre-processing batches into a queue to training a model. It’s important to
note that we do not use the familiar feed_dict argument anymore—this avoids data
copies and offers speedups, as discussed earlier in this chapter:
'''

try:
    step = 0
    while not coord.should_stop():
        step += 1
        sess.run([train_op])
        
        if step % 500 == 0:
            loss_mean_val = sess.run([loss_mean])
            print(step)
            print(loss_mean_val)
            
except tf.errors.OutOfRangeError:
    print('Done training for %d epochs, %d steps.' % (num_epochs, step))
    
finally:
    # when done, ask the treads to stop.
    coord.request_stop()
    
## wait for threads to finish
coord.join(threads)
sess.close()

# %% [markdown]
# #### We train until a tf.errors.OutOfRangeError error is thrown, indicating that queues are empty and we are done:

# %%
sess = tf.Session()

# example -- get image, label
img1, lbel1 = sess.run([image,label])

# example -- get random batch
images, labels = sess.run([images_batch, labels_batch])

sess.close()


# %%


