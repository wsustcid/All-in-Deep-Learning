'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:15:36
@LastEditTime: 2020-05-06 16:15:36
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### The Importance of Sequence Data
# As discussed in that chapter, exploiting structure is the key to success. As we will see shortly, an immensely important and useful type of structure is the sequential structure. Thinking in terms of data science, this
# fundamental structure appears in many datasets, across all domains. In computer vision, video is a sequence of visual content evolving over time. In speech we have audio signals, in genomics gene sequences; we have longitudinal medical records in healthcare, financial data in the stock market, and so on.
# 
# In our MNIST data, this just means that each 28×28-pixel image can be viewed as a sequence of length 28, each element in the sequence a vector of 28 pixels. Then, the temporal dependencies in the RNN can be imaged as a scanner head, scanning the image from top to bottom (rows) or left to right (columns).
# 
# ### Introduction to Recurrent Neural Networks
# When we receive new information, clearly our “history” and “memory” are not wiped
# out, but instead “updated.” When we read a sentence in some text, with each new
# word, our current state of information is updated, and it is dependent not only on the
# new observed word but on the words that preceded it.
# 
# A fundamental mathematical construct in statistics and probability, which is often
# used as a building block for modeling sequential patterns via machine learning is the
# Markov chain model. Figuratively speaking, we can view our data sequences as
# “chains,” with each node in the chain dependent in some way on the previous node,
# so that “history” is not erased but carried on.
# 
# RNN models are also based on this notion of chain structure, and vary in how exactly
# they maintain and update information. As their name implies, recurrent neural nets
# apply some form of “loop.” As seen in Figure 5-2, at some point in time t, the network
# observes an input x t (a word in a sentence) and updates its “state vector” to h t from
# the previous vector h t-1 . When we process new input (the next word), it will be done
# in some manner that is dependent on h t and thus on the history of the sequence (the
# previous words we’ve seen affect our understanding of the current word).
# 
# 
# %% [markdown]
# ## 1. Vanilla RNN Implementation
# While the structure of natural images is
# well suited for CNN models, it is revealing to look at the structure of images from
# different angles. In a trend in cutting-edge deep learning research, advanced models
# attempt to exploit various kinds of sequential structures in images, trying to capture
# in some sense the “generative process” that created each image. Intuitively, this all
# comes down to the notion that nearby areas in images are somehow related, and try‐
# ing to model this structure.
# 
# Here, to introduce basic RNNs and how to work with sequences, we take a simple
# sequential view of images: we look at each image in our data as a sequence of rows (or
# columns). In our MNIST data, this just means that each 28×28-pixel image can be
# viewed as a sequence of length 28, each element in the sequence a vector of 28 pixels
# (see Figure 5-3). Then, the temporal dependencies in the RNN can be imaged as a
# scanner head, scanning the image from top to bottom (rows) or left to right (col‐
# umns).

# %%
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_dir="datasets/MNIST"
mnist = input_data.read_data_sets(data_dir, one_hot=True)


# %%
element_size = 28 # D
time_steps = 28 # T
num_classes = 10 # 
batch_size = 128 # N
hidden_layer_size = 128 # H

# where to save tensorboard model summaries
# TensorBoard allows you to monitor and explore the model
# structure, weights, and training process
log_dir = "logs/RNN_with_summaries"

# creat placeholders for inputs and labels
inputs = tf.placeholder(tf.float32, shape=[None,time_steps, 
                                           element_size],name='inputs')
y = tf.placeholder(tf.float32, shape=[None,num_classes], name='labels')

# data comes in unrolled form—a vector of 784 pixels.
batch_x, batch_y = mnist.train.next_batch(batch_size)
# reshape data to get 28 squences of 28 pixes (N, T ,D)
batch_x = batch_x.reshape((batch_size, time_steps,element_size))


# %%
## We first create a function used for logging summaries, which we 
# will use later in TensorBoard
## This helper function, taken from the official TensorFlow 
# documentation,simply adds some ops that take care of logging summaries

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        
        tf.summary.histogram('histogram', var)


# %%
# wegiths and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    # (D,H)
    with tf.name_scope('Wx'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    # (H,H)
    with tf.name_scope('Wh'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    # (H,)  
    with tf.name_scope('bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)
        

# %% [markdown]
# ### Applying the RNN step with tf.scan()

# %%
## vanilla RNN step
def rnn_step(previous_hidden_state, x):
    # (N,H)
    current_hidden_state = tf.tanh(
        tf.matmul(x, Wx) + tf.matmul(Wh, previous_hidden_state) + b_rnn)
    
    return current_hidden_state


# %%
### apply the rnn_step across all 28 time steps
'''
First, we reshape the inputs and then the first axis in our
input Tensor represents the time axis, we can iterate across all time steps by using the
built-in tf.scan() function, which repeatedly applies a callable (function) to a
sequence of elements in order

There are several advantages to this approach, chief among them the ability to have a
dynamic number of iterations rather than fixed, computational
speedups and optimizations for graph construction.
'''
# processing inputs to work with scan function 
# (batch_size, time_steps, element_size) -> (time_steps, batch_size, element_size)
processed_inputs = tf.transpose(inputs, perm=[1,0,2])

initial_hidden = tf.zeros([batch_size, hidden_layer_size])

# getting all state vectors across time
all_hidden_states = tf.scan(rnn_step, processed_inputs, 
                            initializer = initial_hidden, name="states")


# %%
#### tf.scan example ##
elems = np.array(["T","e","n","s","o","r", " ", "F","l","o","w"])
scan_sum = tf.scan(lambda a, x: a + x, elems)
sess=tf.InteractiveSession()
sess.run(scan_sum)

# %% [markdown]
# ### Sequential outputs
# In an rnn step, we get a state vector for each time step, multiply it by
# some weights, and get an output vector—our new representation of the data.
# 
# Our input to the RNN is sequential, and so is our output. In this sequence classifica‐
# tion example, we take the last state vector and pass it through a fully connected linear
# layer to extract an output vector (which will later be passed through a softmax activa‐
# tion function to generate predictions). This is common practice in basic sequence
# classification, where we assume that the last state vector has “accumulated” informa‐
# tion representing the entire sequence.
# 
# To implement this, we first define the linear layer’s weights and bias term variables,
# and create a factory function for this layer. Then we apply this layer to all outputs
# with tf.map_fn() , which is pretty much the same as the typical map function that
# applies functions to sequences/iterables in an element-wise manner, in this case on
# each element in our sequence.
# Finally, we extract the last output for each instance in the batch, with negative index‐
# ing (similarly to ordinary Python).

# %%
# weights for output layers
with tf.name_scope("linear_layer_weights") as scope:
    # (H,C)
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal(
            [hidden_layer_size, num_classes], mean=0, stddev=0.01))
        variable_summaries(Wl)
    # (C,)    
    with tf.name_scope("Bias_linar"):
        bl = tf.Variable(tf.truncated_normal(
            [num_classes], mean=0, stddev=0.01))
        variable_summaries(bl)
    
# apply linear layer to state vector
def get_linear_layer(hidden_state):
        
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope("linear_layer_weights") as scope:
    # iterate across time, apple linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
        
    # get last output
    output = all_outputs[-1]
        
    tf.summary.histogram('outputs', output)

# %% [markdown]
# ### RNN classification
# We’re now ready to train a classifier, much in the same way we did in the previous
# chapters. We define the ops for loss function computation, optimization, and predic‐
# tion, add some more summaries for TensorBoard, and merge all these summaries
# into one operation:

# %%
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)
    
with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.0005, 0.9).minimize(cross_entropy)
    
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()


# %%
# get a small test set
test_data = mnist.test.images[:batch_size].reshape((-1,time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # write summaries to log_dir -- used by tensorboard
    train_writer  = tf.summary.FileWriter(log_dir + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(log_dir + '/test',
                                       graph=tf.get_default_graph())
    
    sess.run(tf.global_variables_initializer())
    
    for i in range(50000):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # reshape data to get 28 squences of 28 pixels
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

# %% [markdown]
# ### Visualizing the model with TensorBoard
# TensorBoard is an interactive browser-based tool that allows us to visualize the learn‐
# ing process, as well as explore our trained model.
# To run TensorBoard, go to the command terminal and tell TensorBoard where the
# relevant summaries you logged are:
# tensorboard --logdir=LOG_DIR
# Here, LOG_DIR should be replaced with your log directory. If you are on Windows and
# this is not working, make sure you are running the terminal from the same drive
# where the log data is, and add a name to the log directory as follows in order to
# bypass a bug in the way TensorBoard parses the path:
# tensorboard --logdir=rnn_demo:LOG_DIR
# TensorBoard allows us to assign names to individual log directories by putting a
# colon between the name and the path, which may be useful when working with mul‐
# tiple log directories. In such a case, we pass a comma-separated list of log directories
# as follows:
# tensorboard --logdir=rnn_demo1:LOG_DIR1, rnn_demo2:LOG_DIR2
# In our example (with one log directory), once you have run the tensorboard com‐
# mand, you should get something like the following, telling you where to navigate in
# your browser:
# Starting TensorBoard b'39' on port 6006
# (You can navigate to http://10.100.102.4:6006)
# If the address does not work, go to localhost:6006, which should always work.
# TensorBoard recursively walks the directory tree rooted at LOG_DIR looking for sub‐
# directories that contain tfevents log data. If you run this example multiple times,
# make sure to either delete the LOG_DIR folder you created after each run, or write the
# logs to separate subdirectories within LOG_DIR , such as LOG_DIR /run1/train, LOG_DIR /
# run2/train, and so forth, to avoid issues with overwriting log files, which may lead to
# some “funky” plots.
# Let’s take a look at some of the visualizations we can get. In the next section, we will
# explore interactive visualization of high-dimensional data with TensorBoard—for
# now, we focus on plotting training process summaries and trained weights.
# First, in your browser, go to the Scalars tab. Here TensorBoard shows us summaries
# of all scalars, including not only training and testing accuracy, which are usually most
# interesting, but also some summary statistics we logged about variables (see
# Figure 5-4). Hovering over the plots, we can see some numerical figures.
# In the Graphs tab we can get an interactive visualization of our computation graph,
# from a high-level view down to the basic ops, by zooming in (see Figure 5-5).
# Finally, in the Histograms tab we see histograms of our weights across the training
# process (see Figure 5-6). Of course, we had to explicitly add these histograms to our
# logging in order to view them, with tf.summary.histogram() .
# %% [markdown]
# ## TensorFlow Built-in RNN Functions
# The preceding example taught us some of the fundamental and powerful ways we can
# work with sequences, by implementing our graph pretty much from scratch.
# 
# tf.contrib.rnn.BasicRNNCell and tf.nn.dynamic_rnn()
# TensorFlow’s RNN cells are abstractions that represent the basic operations each
# recurrent “cell” carries out (see Figure 5-2 at the start of this chapter for an illustra‐
# tion), and its associated state. They are, in general terms, a “replacement” of the
# rnn_step() function and the associated variables it required. Of course, there are
# many variants and types of cells, each with many methods and properties. We will see
# some more advanced cells toward the end of this chapter and later in the book.
# 
# Once we have created the rnn_cell , we feed it into tf.nn.dynamic_rnn() . This func‐
# tion replaces tf.scan() in our vanilla implementation and creates an RNN specified
# by rnn_cell .
# As of this writing, in early 2017, TensorFlow includes a static and a dynamic function
# for creating an RNN. What does this mean? The static version creates an unrolled
# graph (as in Figure 5-2) of fixed length. The dynamic version uses a tf.While loop to
# dynamically construct the graph at execution time, leading to faster graph creation,
# which can be significant. This dynamic construction can also be very useful in other
# ways, some of which we will touch on when we discuss variable-length sequences
# toward the end of this chapter.

# %%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('datasets/MNIST', one_hot=True)

element_size = 28; time_steps =28; num_classes=10
batch_size = 128; hidden_layer_size = 128

inputs = tf.placeholder(tf.float32, shape=[None,time_steps,element_size],
                       name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],
                  name='inputs')

# tensorflow build-in functions
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
# 'cell', 'inputs', 'sequence_length=None', 'initial_state=None', 'dtype=None'
outputs, _ = tf.nn.dynamic_rnn(rnn_cell,inputs, dtype=tf.float32)

Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes],
                                    mean=0, stddev=0.01))
bl = tf.Variable(tf.truncated_normal([num_classes],
                                    mean=0, stddev=0.01))

def get_linear_layer(vector):
    return tf.matmul(vector,Wl) + bl

last_rnn_output = outputs[:,-1,:]
final_output = get_linear_layer(last_rnn_output)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=final_output,labels=y))

train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(final_output,1), tf.arg_max(y,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

test_data = mnist.test.images[:batch_size].reshape((-1,time_steps,element_size))
test_label = mnist.test.labels[:batch_size]

for i in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size,time_steps,element_size))
    
    sess.run(train_step, feed_dict={inputs:batch_x, y:batch_y})
    
    if i % 1000 == 0:
        acc, loss = sess.run([accuracy,cross_entropy], 
                             feed_dict={inputs:batch_x, y:batch_y})
        print("{} loss: {} acc: {:.4f}%".format(i, loss, acc))
        

acc_test = sess.run(accuracy, feed_dict={inputs:test_data, y:test_label})
print("test acc: {:.4f}%".format(acc_test))

# %% [markdown]
# ## 2. RNN for Text Sequences
# In the MNIST RNN example we saw earlier, each sequence was of fixed size—the
# width (or height) of an image. Each element in the sequence was a dense vector of 28
# pixels. In NLP tasks and datasets, we have a different kind of “picture.”
# 
# When creating sentences, we sample random digits and map them to the corre‐
# sponding “words” (e.g., 1 is mapped to “One,” 7 to “Seven,” etc.).
# Text sequences typically have variable lengths, which is of course the case for all real
# natural language data (such as in the sentences appearing on this page).
# 
# To make our simulated sentences have different lengths, we sample for each sentence
# a random length between 3 and 6 with np.random.choice(range(3, 7)) —the lower
# bound is inclusive, and the upper bound is exclusive.
# 
# Now, to put all our input sentences in one tensor (per batch of data instances), we
# need them to somehow be of the same size—so we pad sentences with a length
# shorter than 6 with zeros (or PAD symbols) to make all sentences equally sized (artifi‐
# cially). This pre-processing step is known as zero-padding. The following code
# accomplishes all of this:

# %%
import numpy as np
import tensorflow as tf

batch_size = 128; embedding_dimension=64; num_classes=2
hidden_layer_size=32; time_steps=6; element_size=1


# %%
digit_to_word_map = {1:"One", 2:"Two", 3:"Three",
                    4:"Four", 5:"Five", 6:"Six",
                    7:"Seven", 8:"Eight", 9:"Nine"}
digit_to_word_map[0] = "PAD"

even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    
    rand_odd_ints = np.random.choice(range(1,10,2),
                                    rand_seq_len) 
    rand_even_ints = np.random.choice(range(2,10,2),
                                     rand_seq_len)
    
    ## padding
    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints, [0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints, [0]*(6-rand_seq_len))
        
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
        
data = even_sentences + odd_sentences

# same seq length for even , odd sentences
# original sequence lengths
seqlens *= 2 # 将列表内的内容复制一份


# %%
'''
Why keep the original sentence lengths? By zero-padding, we solved one technical
problem but created another: if we naively pass these padded sentences through our
RNN model as they are, it will process useless PAD symbols. This would both harm
model correctness by processing “noise” and increase computation time. We resolve
this issue by first storing the original lengths in the seqlens array and then telling
TensorFlow’s tf.nn.dynamic_rnn() where each sentence ends.
'''

print(even_sentences[0:6], "\n \n", odd_sentences[0:6], 
      "\n \n", seqlens[0:6], seqlens[10000:10006])


# %%
'''
So, we now map words to indices—word identifiers—by simply creating a dictionary
with words as keys and indices as values. We also create the inverse map. Note that
there is no correspondence between the word IDs and the digits each word represents
—the IDs carry no semantic meaning, just as in any NLP application with real data:
'''
# Map from words to indices
word2index_map = {}
index = 0 
for sent in data:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

# Inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)


# %%
'''
This is a supervised classification task—we need an array of labels in the one-hot for‐
mat, train and test sets, a function to generate batches of instances, and placeholders,
as usual.
'''
# First, we create the labels and split the data into train and test sets
labels = [1]*10000 + [0]*10000

for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1 
    labels[i] = one_hot_encoding
    
## shuffle data, label and seqlen
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices] 
seqlens = np.array(seqlens)[data_indices]

train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]


# %%
## generate batches of sentences
## eaach sentence in a batch is simply a list of integer IDs corresponding to teh words
def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    
    x = [[word2index_map[word] for word in data_x[i].lower().split()] 
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    
    return x, y, seqlens


# %%
## create placeholders for data
_inputs = tf.placeholder(tf.int32, shape=[batch_size, time_steps]) # ?
_labels = tf.placeholder(tf.int32, shape=[batch_size, num_classes])

# seqlens for dynamic calculation
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

# %% [markdown]
# ### Supervised Word Embeddings
# 
# The embedding is, in a nutshell, simply a mapping from high-dimensional one-hot vec‐
# tors encoding words to lower-dimensional dense vectors. So, for example, if our
# vocabulary has size 100,000, each word in one-hot representation would be of the
# same size. The corresponding word vector—or word embedding—would be of size
# 300, say. The high-dimensional one-hot vectors are thus “embedded” into a continu‐
# ous vector space with a much lower dimensionality.

# %%

with tf.name_scope("embeddings"):
    embeddings = tf.Variable(tf.random_uniform(
        [vocabulary_size, embedding_dimension], -1.0, 1.0), name='embedding')
    
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

# %% [markdown]
# ### LSTM and Using Sequence Length
# A very popular recurrent network is the long short-term
# memory (LSTM) network. It differs from vanilla RNN by having some special mem‐
# ory mechanisms that enable the recurrent cells to better store information for long
# periods of time, thus allowing them to capture long-term dependencies better than
# plain RNN.
# 
# There is nothing mysterious about these memory mechanisms; they simply consist of
# some more parameters added to each recurrent cell, enabling the RNN to overcome
# optimization issues and propagate information. These trainable parameters act as fil‐
# ters that select what information is worth “remembering” and passing on, and what is
# worth “forgetting.” They are trained in exactly the same way as any other parameter
# in a network, with gradient-descent algorithms and backpropagation.

# %%
'''
We create an LSTM cell with tf.contrib.rnn.BasicLSTMCell() and feed it to
tf.nn.dynamic_rnn() , just as we did at the start of this chapter. We also give
dynamic_rnn() the length of each sequence in a batch of examples, using the _seq
lens placeholder we created earlier. TensorFlow uses this to stop all RNN steps
beyond the last real sequence element. 

It also returns all output vectors over time (in
the outputs tensor), which are all zero-padded beyond the true end of the sequence.
So, for example, if the length of our original sequence is 5 and we zero-pad it to a
sequence of length 15, the output for all time steps beyond 5 will be zero:
'''
## once the scope is used, the code can only be run once 
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
    # states tensor -- the last valid output vector 
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed,
                                       sequence_length=_seqlens,
                                       dtype=tf.float32)
    
weights = {'linear_layer': tf.Variable(tf.truncated_normal(
    [hidden_layer_size, num_classes], mean=0, stddev=0.01))}

biases = {'linear_layer': tf.Variable(tf.truncated_normal(
    [num_classes], mean=0, stddev=0.01))}


# extract the last relevant output and use in a linear layer
final_output = tf.matmul(states[1], weights['linear_layer']) + biases['linear_layer']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=final_output, labels=_labels))

# %% [markdown]
# ### Training Embeddings and the LSTM Classifier

# %%
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(final_output,1), 
                              tf.arg_max(_labels,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                           train_x, train_y,
                                                           train_seqlens)
        sess.run(train_step, feed_dict={_inputs:x_batch, _labels:y_batch,
                                       _seqlens:seqlen_batch})
        
        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict={_inputs:x_batch, _labels:y_batch,
                                       _seqlens:seqlen_batch})
            print("training acc {}: {:.4f}%".format(step,acc))
    
    
    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                        test_x, test_y,
                                                        test_seqlens)
        test_acc = sess.run(accuracy, feed_dict={_inputs:x_test, 
                                                 _labels:y_test,
                                                 _seqlens:seqlen_test})
        print("testing acc {}: {:.4f}%".format(test_batch, test_acc))
        
    
        output_example = sess.run(outputs, feed_dict={_inputs:x_test, 
                                                      _labels:y_test,
                                                      _seqlens:seqlen_test})

        states_example = sess.run(states, feed_dict={_inputs:x_test, 
                                                      _labels:y_test,
                                                      _seqlens:seqlen_test})


# %%
seqlen_test[0]


# %%
output_example.shape


# %%
# We see that for this sentence, whose original length was 4, 
# the last two time steps have zero vectors due to padding.
output_example[0][:6,0:3]


# %%
# ????
## We can see that it conveniently stores for us the last relevant output vector
## —its values match the last relevant output vector before zero-padding.
states_example[1][0][0:3]

# %% [markdown]
# ### Stacking multiple LSTMs
# Earlier, we focused on a one-layer LSTM network for ease of exposition. Adding
# more layers is straightforward, using the MultiRNNCell() wrapper that combines
# multiple RNN cells into one multilayer cell.

# %%
num_LSTM_layers = 2
with tf.variable_scope("lstm"):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,
                                            forget_bias=1.0)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell]*num_LSTM_layers,
                                      state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed,
                                       sequence_length= _seqlens,
                                       dtype=tf.float32)
    


# %%
## To get the final state of the second layer, we simply
#  adapt our indexing a bit:
# extract the final state and use in a linear layer
final_output = tf.matmul(states[num_LSTM_layers-1][1], weights["linear_layer"])
               + biases["linear_layer"]

