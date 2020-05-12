'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:16:16
@LastEditTime: 2020-05-06 16:16:16
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Introduction to Word Embeddings
# Consider the sentence appearing in Figure 6-1: “Our company provides smart agri‐
# culture solutions for farms, with advanced AI, deep-learning.” This sentence may be
# taken from, say, a tweet promoting a company. As data scientists or engineers, we
# now may wish to process it as part of an advanced machine intelligence system, that
# sifts through tweets and automatically detects informative content (e.g., public senti‐
# ment).
# 
# In one of the major traditional natural language processing (NLP) approaches to text
# processing, each of the words in this sentence would be represented with N ID—say,
# an integer. So, as we posited in the previous chapter, the word “agriculture” might be
# mapped to the integer 3452, the word “farm” to 12, “AI” to 150, and “deep-learning”
# to 0.
# 
# While this representation has led to excellent results in practice in some basic NLP
# tasks and is still often used in many cases (such as in bag-of-words text classification),
# it has some major inherent problems. First, by using this type of atomic representa‐
# tion, we lose all meaning encoded within the word, and crucially, we thus lose infor‐
# mation on the semantic proximity between words. In our example, we of course
# know that “agriculture” and “farm” are strongly related, and so are “AI” and “deep-
# learning,” while deep learning and farms don’t usually have much to do with one
# another. This is not reflected by their arbitrary integer IDs.
# 
# Another important issue with this way of looking at data stems from the size of typi‐
# cal vocabularies, which can easily reach huge numbers. This means that naively, we
# could need to keep millions of such word identifiers, leading to great data sparsity
# and in turn, making learning harder and more expensive.
# 
# With images, such as in the MNIST data we used in the first section of Chapter 5, this
# is not quite the case. While images can be high-dimensional, their natural representa‐
# tion in terms of pixel values already encodes some semantic meaning, and this repre‐
# sentation is dense. In practice, RNN models like the one we saw in Chapter 5 require
# dense vector representations to work well.
# 
# We would like, therefore, to use dense vector representations of words, which carry
# semantic meaning. But how do we obtain them?
# %% [markdown]
# ## 1. Word2vec
# Word2vec is a very well-known unsupervised word embedding approach. It is
# actually more like a family of algorithms, all based in some way on exploiting the
# context in which words appear to learn their representation (in the spirit of the distri‐
# butional hypothesis). We focus on the most popular word2vec implementation,
# which trains a model that, given an input word, predicts the word’s context by using
# something known as skip-grams. This is actually rather simple, as the following exam‐
# ple will demonstrate.
# 
# Consider, again, our example sentence: “Our company provides smart agriculture sol‐
# utions for farms, with advanced AI, deep-learning.” We define (for simplicity) the
# context of a word as its immediate neighbors (“the company it keeps”)—i.e., the word
# to its left and the word to its right. So, the context of “company” is [our, provides], the context of “AI” is [advanced, deep-learning], and so on (see Figure 6-1).
# 
# In the skip-gram word2vec model, we train a model to predict context based on an
# input word. All that means in this case is that we generate training instance and label
# pairs such as (our, company), (provides, company), (advanced, AI), (deep-learning,
# AI), etc.
# 
# In addition to these pairs we extract from the data, we also sample “fake” pairs—that
# is, for a given input word (such as “AI”), we also sample random noise words as con‐
# text (such as “monkeys”), in a process known as negative sampling. We use the true
# pairs combined with noise pairs to build our training instances and labels, which we
# use to train a binary classifier that learns to distinguish between them. The trainable
# parameters in this classifier are the vector representations—word embeddings. We
# tune these vectors to yield a classifier able to tell the difference between true contexts
# of a word and randomly sampled ones, in a binary classification setting.
# 
# 
# ### Skip-Grams
# We begin by preparing our data and extracting skip-grams. As in Chapter 5, our data
# comprises two classes of very short “sentences,” one composed of odd digits and the
# other of even digits (with numbers written in English). We make sentences equally
# sized here, for simplicity, but this doesn’t really matter for word2vec training. Let’s
# start by setting some parameters and creating sentences:

# %%
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 64 
embedding_dimension = 5 
negative_samples = 8 
log_dir = "logs/word2vec_intro"

# the digit here has no special meaning, just for creating sentence data 
digit_to_word_map = {1: "One", 2: "Two", 3: "Three",
                    4: "Four", 5: "Five", 6: "Six",
                    7: "Seven", 8: "Eight", 9: "Nine"}

sentences = []

# create two kinds of sentences - seqences of odd and even digits
for i in range(10000):
    rand_odd_ints = np.random.choice(range(1,10,2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2,10,2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))


# %%
print(len(sentences), "\n \n", sentences[:10])


# %%
## as in Chapter 5, we map words to indices by creating a dictionary 
## with words as keys and indices as values, and create the inverse map

word2index_map = {}
index = 0 
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index+=1

index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)


# %%
print(vocabulary_size, len(word2index_map))


# %%
# to prepare the data for word2vec, let's create skip-grams(语义相关对)
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1): ## i=1
        # [[0,2],1]
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                             word2index_map[tokenized_sent[i]]
                            ]
        
        # [1, 0]
        skip_gram_pairs.append([word_context_pair[1],
                               word_context_pair[0][0]])
        # [1, 2]
        skip_gram_pairs.append([word_context_pair[1],
                               word_context_pair[0][1]])


# %%
'''
Each skip-gram pair is composed of target and context word indices (given by the
word2index_map dictionary, and not in correspondence to the actual digit each word
represents). Let’s take a look:
'''
skip_gram_pairs[:10]


# %%
def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch] # (batch,)
    y = [[skip_gram_pairs[i][1]] for i in batch] ## (batch,1)
    
    # x,y is a pair of semantically related word
    return x, y


# %%
# We can generate batches of sequences of word indices, and check out 
# the original sentences with the inverse dictionary we created earlier:

# batch example
x_batch, y_batch = get_skipgram_batch(8)
print(x_batch, "\n")
print(y_batch, "\n")
print([index2word_map[index] for index in x_batch], '\n')
print([index2word_map[index[0]] for index in y_batch])


# %%
# create input and label placeholders
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])

# %% [markdown]
# ### Embedding in TensorFlow
# In Chapter 5, we used the built-in tf.nn.embedding_lookup() function as part of
# our supervised RNN. The same functionality is used here. Here too, word embed‐
# dings can be viewed as lookup tables that **map words to vector values**, which are opti‐
# mized as part of the training process to minimize a loss function. As we shall see in
# the next section, unlike in Chapter 5, here we use a loss function accounting for the
# unsupervised nature of the task, but the embedding lookup, which efficiently
# retrieves the vectors for each word in a given sequence of word indices, remains the
# same:

# %%
with tf.name_scope("embeddings"):
    # (9,5)
    embeddings = tf.Variable(tf.random_uniform(
        [vocabulary_size, embedding_dimension],-1.0, 1.0), 
                             name='embedding')
    # this is essentially a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# %% [markdown]
# ### The Noise-Contrastive Estimation (NCE) Loss Function
# In our introduction to skip-grams, we mentioned we create two types of context–
# target pairs of words: real ones that appear in the text, and “fake” noisy pairs that are
# generated by inserting random context words. Our goal is to learn to distinguish
# between the two, helping us learn a good word representation. We could draw ran‐
# dom noisy context pairs ourselves, but luckily TensorFlow comes with a useful loss
# function designed especially for our task. tf.nn.nce_loss() automatically draws
# negative (“noise”) samples when we evaluate the loss (run it in a session):
# 
# We don’t go into the mathematical details of this loss function, but it is sufficient to
# think of it as a sort of efficient approximation to the ordinary softmax function used
# in classification tasks, as introduced in previous chapters. We tune our embedding
# vectors to optimize this loss function.

# %%
## create variables for the NCE loss
nce_weights = tf.Variable(tf.truncated_normal(
    [vocabulary_size, embedding_dimension],
    stddev=1.0/math.sqrt(embedding_dimension)))

nce_bias = tf.Variable(tf.zeros([vocabulary_size])) 

loss = tf.reduce_mean(
    tf.nn.nce_loss(weights=nce_weights, biases=nce_bias,
                  inputs=embed, labels=train_labels, 
                   num_sampled=negative_samples,
                  num_classes=vocabulary_size))
tf.summary.scalar("NCE_loss", loss)

# %% [markdown]
# ### Learning Rate Decay
# As discussed in previous chapters, gradient-descent optimization adjusts weights by
# making small steps in the direction that minimizes our loss function. The learn
# ing_rate hyperparameter controls just how aggressive these steps are. During
# gradient-descent training of a model, it is common practice to gradually make these
# steps smaller and smaller, so that we allow our optimization process to “settle down”
# as it approaches good points in the parameter space. This small addition to our train
# ing process can actually often lead to significant boosts in performance, and is a good
# practice to keep in mind in general.
# 
# tf.train.exponential_decay() applies exponential decay to the learning rate, with
# the exact form of decay controlled by a few hyperparameters, as seen in the following
# code (for exact details, see the official TensorFlow documentation at http://bit.ly/
# 2tluxP1). Here, just as an example, we decay every 1,000 steps, and the decayed learn‐
# ing rate follows a staircase function—a piecewise constant function that resembles a
# staircase, as its name implies:

# %%
# learning rate decay
global_step = tf.Variable(0, trainable=False)
learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                         global_step=global_step,
                                         decay_steps=1000,
                                         decay_rate=0.95,
                                         staircase=True)
train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# %% [markdown]
# ### Training and Visualizing with TensorBoard
# We train our graph within a session as usual, adding some lines of code enabling cool
# interactive visualization in TensorBoard, a new tool for visualizing embeddings of
# high-dimensional data—typically images or word vectors—introduced for Tensor‐
# Flow in late 2016.
# First, we create a TSV (tab-separated values) metadata file. This file connects embed‐
# ding vectors with associated labels or images we may have for them. In our case, each
# embedding vector has a label that is just the word it stands for.
# We then point TensorBoard to our embedding variables (in this case, only one), and
# link them to the metadata file.
# Finally, after completing optimization but before closing the session, we normalize
# the word embedding vectors to unit length, a standard post-processing step:

# %%
# merge all summary ops
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir,
                                        graph=tf.get_default_graph())
    
    saver = tf.train.Saver()
    
    with open(os.path.join(log_dir,'metadata.tsv'), 'w') as metadata:
        metadata.write('Name\tClass\n')
        for k,v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v,k))
    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name
    
    # Link embedding to its metadata file
    embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)
    
    tf.global_variables_initializer().run()
    
    for step in range(1000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict=
                              {train_inputs:x_batch,
                              train_labels:y_batch})
        
        train_writer.add_summary(summary, step)
        
        if step % 100 == 0:
            saver.save(sess, os.path.join(log_dir,'w2v_model.ckpt'), step)
            loss_value = sess.run(loss,  feed_dict=
                              {train_inputs:x_batch,
                              train_labels:y_batch})
            print("{} loss: {:.5f}".format(step, loss_value))
            

    # Nomalize embedding before using
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normlized_embeddings = embeddings/norm
    normlized_embeddings_matrix = sess.run(normlized_embeddings)

# %% [markdown]
# ### Checking Out Our Embeddings
# Let’s take a quick look at the word vectors we got. We select one word (one) and sort
# all the other word vectors by how close they are to it, in descending order:

# %%
# (vocabulary_size, embedding_dimension)
normlized_embeddings_matrix


# %%
ref_word = normlized_embeddings_matrix[word2index_map["one"]] # (5,)

cosine_dists = np.dot(normlized_embeddings_matrix, ref_word) # (9,)
ff = np.argsort(cosine_dists)[::-1][1:10]
for f in ff:
    print(index2word_map[f])
    print(cosine_dists[f])

# %% [markdown]
# We see that the word vectors representing odd numbers are similar (in terms of the
# dot product) to one, while those representing even numbers are not similar to it (and
# have a negative dot product with the one vector). We learned embedded vectors that
# allow us to distinguish between even and odd numbers—their respective vectors are
# far apart, and thus capture the context in which each word (odd or even
# digit) appeared.
# %% [markdown]
# ### Tensorboard
# Now, in TensorBoard, go to the Embeddings tab. This is a three-dimensional interac‐
# tive visualization panel, where we can move around the space of our embedded vec‐
# tors and explore different “angles,” zoom in, and more (see Figures 6-2 and 6-3). This
# enables us to understand our data and interpret the model in a visually comfortable
# manner. We can see, for instance, that the odd and even numbers occupy different
# areas in feature space.
# 
# ~/tensorflow/examples/logs$ tensorboard --logdir=word2vec_intro
# 
# TensorBoard 1.9.0 at http://desktop:6006 (Press CTRL+C to quit)
# 
# %% [markdown]
# ## 2. Pretrained Embeddings, Advanced RNN
# 
# Here, we show how to take word vectors trained based on web data and incorporate
# them into a (contrived) text-classification task. The embedding method is known as
# GloVe, and while we don’t go into the details here, the overall idea is similar to that of
# word2vec—learning representations of words by the context in which they appear.
# Information on the method and its authors, and the pretrained vectors, is available on
# the project’s website.(https://nlp.stanford.edu/projects/glove/)
# We download the Common Crawl vectors (840B tokens), and proceed to our exam‐
# ple.

# %%
import zipfile
import numpy as np
import tensorflow as tf

path_to_glove = "glove/glove.840B.300d.zip"
PRE_TRAINED = True
GLOVE_SIZE = 300
batch_size =128
emdedding_dimension = 64 
num_classes = 2 
hidden_layer_size = 32 
times_steps = 6 


# %%
# create the contrived, simple simulated data
digit_to_word_map = {1:"One", 2:"Two", 3:"Three",
                    4:"Four", 5:"Five",6:"Six",
                    7:"Seven", 8:"Eight",9:"Nine"}

digit_to_word_map[0] = "PAD_TOKEN"
even_sentences = []
odd_sentences = []
seqlens = []

for i in range(10000):
    rand_seq_len = np.random.choice(range(3,7))
    seqlens.append(rand_seq_len)
    
    rand_odd_ints = np.random.choice(range(1,10,2), rand_seq_len)
    rand_even_ints = np.random.choice(range(2,10,2), rand_seq_len)
    
    if rand_seq_len < 6:
        rand_odd_ints = np.append(rand_odd_ints,
                                 [0]*(6-rand_seq_len))
        rand_even_ints = np.append(rand_even_ints,
                                 [0]*(6-rand_seq_len))
        
    odd_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    even_sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    
data = even_sentences + odd_sentences
# same seq lengths for even, odd sentences
seqlens*=2 

labels = [1]*10000 + [0]*10000 
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1 # the true class label is 1
    labels[i] = one_hot_encoding
           


# %%
print(len(seqlens))
print(np.shape(labels))


# %%
# create the word index map
word2index_map={}
index = 0 
for sent in data:
    for word in sent.split():
        if word not in word2index_map:
            word2index_map[word] = index
            index +=1
    
index2word_map = {index: word for word, index in word2index_map.items()}

vocabulary_size = len(index2word_map)


# %%
word2index_map


# %%
'''
Now, we are ready to get word vectors. There are 2.2 million words in the vocabulary
of the pretrained GloVe embeddings we downloaded, and in our toy example we have
only 9. So, we take the GloVe vectors only for words that appear in our own tiny
vocabulary

We go over the GloVe file line by line, take the word vectors we need, and normalize
them. Once we have extracted the nine words we need, we stop the process and exit
the loop. The output of our function is a dictionary, mapping from each word to its
vector.
'''
def get_glove(path_to_glove, word2index_map):
    embedding_weights = {}
    count_all_words = 0 
    with zipfile.ZipFile(path_to_glove) as z:
        with z.open("glove.840B.300d.txt") as f:
            for line in f: 
                vals = line.split()
                word = str(vals[0].decode("utf-8"))
                if word in word2index_map:
                    print(word)
                    count_all_words+=1 
                    coefs = np.asarray(vals[1:], dtype='float32')
                    coefs /= np.linalg.norm(coefs)
                    embedding_weights[word] = coefs
                if count_all_words == vocabulary_size-1: # other one is PAD
                    break
    
    return embedding_weights

word2embedding_dict = get_glove(path_to_glove, word2index_map)


# %%
# The next step is to place these vectors in a matrix, which is the required format 
# for TensorFlow. In this matrix, each row index should correspond to the word index

'''
Note that for the PAD_TOKEN word, we set the corresponding vector to 0. As we saw in
Chapter 5, we ignore padded tokens in our call to dynamic_rnn() by telling it the
original sequence length.
'''
embedding_matrix = np.zeros((vocabulary_size, GLOVE_SIZE))

for word, index in word2index_map.items():
    if not word == "PAD_TOKEN":
        word_embedding = word2embedding_dict[word]
        embedding_matrix[index,:] = word_embedding


# %%
print(embedding_matrix.shape)
print(word2index_map["PAD_TOKEN"])
print(embedding_matrix[2])


# %%
## create our training and test data
data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]
labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]

train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = data[10000:]
test_seqlens = seqlens[10000:]

def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    
    x = [[word2index_map[word] for word in data_x[i].split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    
    return x, y, seqlens


# %%
# create input placeholders
_inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
embedding_placeholder = tf.placeholder(tf.float32, [vocabulary_size, GLOVE_SIZE])

_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])


# %%
'''
Our embeddings are initialized with the content of embedding_placeholder , using
the assign() function to assign initial values to the embeddings variable. We set
trainable=True to tell TensorFlow we want to update the values of the word vectors,
by optimizing them for the task at hand. However, it is often useful to set
trainable=False and not update these values; for example, when we do not have
much labeled data or have reason to believe the word vectors are already “good” at
capturing the patterns we are after.
'''
# we created an embedding_placeholder , to which we feed the word vectors
if PRE_TRAINED:
    embeddings = tf.Variable(tf.constant(0.0, shape=[vocabulary_size, 
                                                    GLOVE_SIZE]),
                            trainable=True)
    
    # if using pretrained embeddings, assign them to the embedding variable
    embedding_init = embeddings.assign(embedding_placeholder)
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
    
else:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, 
                                                embedding_dimension],
                                              -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

# %% [markdown]
# ### Bidirectional RNN and GRU Cells
# Bidirectional RNN layers are a simple extension of the RNN layers we saw in Chap‐
# ter 5. All they consist of, in their basic form, is two ordinary RNN layers: one layer
# that reads the sequence from left to right, and another that reads from right to left.
# Each yields a hidden representation, the left-to-right vector h , and the right-to-left
# vector h . These are then concatenated into one vector. The major advantage of this
# representation is its ability to capture the context of words from both directions,
# which enables richer understanding of natural language and the underlying seman‐
# tics in text.
# 
# **Gated recurrent unit (GRU) cells are a simplification of sorts of LSTM cells. They also
# have a memory mechanism, but with considerably fewer parameters than LSTM.
# They are often used when there is less available data, and are faster to compute.**

# %%
'''
TensorFlow comes equipped with tf.nn.bidirectional_dynamic_rnn() , which is
an extension of dynamic_rnn() for bidirectional layers. It takes cell_fw and cell_bw
RNN cells, which are the left-to-right and right-to-left vectors, respectively. Here we
use GRUCell() for our forward and backward representations and add dropout for
regularization, using the built-in DropoutWrapper() :
'''
with tf.name_scope("biGRU"):
    with tf.variable_scope("forward"):
        gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)
        
    with tf.variable_scope("backward"):
        gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
        gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)
        
    
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell,
                                                      cell_bw=gru_bw_cell,
                                                      inputs=embed,
                                                     sequence_length=_seqlens,
                                                     dtype=tf.float32,
                                                     scope="BiGRU")
states = tf.concat(values=states, axis=1)


# %%
# We concatenate the forward and backward state vectors by using tf.concat() along
# the suitable axis, and then add a linear layer followed by softmax
weights = {'linear_layer': tf.Variable(tf.truncated_normal([2*hidden_layer_size,
                                                           num_classes],
                                                          mean=0, stddev=0.01))}

biases = {'linear_layer': tf.Variable(tf.truncated_normal([num_classes],
                                                          mean=0, stddev=0.01))}

# extract the final state and use in a linear layer
final_output = tf.matmul(states, weights["linear_layer"]) + biases["linear_layer"]

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=final_output, labels=_labels))

train_step = tf.train.RMSPropOptimizer(0.01,0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(_labels,1),
                              tf.arg_max(final_output,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100


# %%
'''
We are now ready to train. We initialize the embedding_placeholder by feeding it
our embedding_matrix . It’s important to note that we do so after calling
tf.global_variables_initializer() —doing this in the reverse order would over‐
run the pre-trained vectors with a default initializer:
'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(embedding_init, feed_dict={embedding_placeholder:embedding_matrix})
    
    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size,
                                                           train_x,
                                                           train_y,
                                                           train_seqlens)
        sess.run(train_step, feed_dict={_inputs:x_batch,
                                        _labels:y_batch,
                                        _seqlens:seqlen_batch})
        if step % 100 ==0:
            acc = sess.run(accuracy, feed_dict={_inputs:x_batch,
                                        _labels:y_batch,
                                        _seqlens:seqlen_batch})
            print("{} train acc: {:.4f}%".format(step, acc))
            
            
    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size,
                                                           test_x,
                                                           test_y,
                                                           test_seqlens)
        
        batch_acc = sess.run(accuracy, feed_dict={_inputs:x_test,
                                                    _labels:y_test,
                                                    _seqlens:seqlen_test})
        
        print("{} test acc: {:.4f}%".format(test_batch, batch_acc))


# %%



# %%



# %%



# %%



# %%



# %%


