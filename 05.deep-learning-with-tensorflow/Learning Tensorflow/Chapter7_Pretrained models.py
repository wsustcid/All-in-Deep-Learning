'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:17:17
@LastEditTime: 2020-05-06 16:17:17
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Pretrained models with TF-Slim
# Like Keras, it also offers a nice variety of pretrained CNN models to download and use.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets/
# 
# ## TF-Slim
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
# 
# TF-Slim is a relatively new lightweight extension of TensorFlow that, like other
# abstractions, allows us to define and train complex models quickly and
# intuitively. TF-Slim doesn’t require any installation since it’s been merged with Ten‐
# sorFlow.
# 
# This extension is all about convolutional neural networks. CNNs are notorious for
# having a lot of messy boilerplate code. TF-Slim was designed with the goal of opti‐
# mizing the creation of very complex CNN models so that they could be elegantly
# written and easy to interpret and debug by using high-level layers, variable abstrac‐
# tions, and argument scoping, which we will touch upon shortly.
# 
# In addition to enabling us to create and train our own models, TF-Slim has available
# pretrained networks that can be easily downloaded, read, and used: VGG, AlexNet,
# Inception, and more.
# 
# We start this section by briefly describing some of TF-Slim’s abstraction features.
# Then we shift our focus to how to download and use a pretrained model, demonstrat‐
# ing it for the VGG image classification model.
# %% [markdown]
# With TF-Slim we can create a variable easily by defining its initialization, regulariza‐
# tion, and device with one wrapper. For example, here we define weights initialized
# from a truncated normal distribution using L2 regularization and placed on the CPU
# (we will talk about distributing model parts across devices in Chapter 9):

# %%
import tensorflow as tf
from tensorflow.contrib import slim

W = slim.variable('w', shape=[7,7,3, 3],
                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                  regularizer=slim.l2_regularizer(0.07),
                  device='/CPU:0')

# %% [markdown]
# TF-Slim can reduce a lot of
# boilerplate code and redundant duplication. As with Keras or TFLearn, we can define
# a layer operation at an abstract level to include the convolution operation, weights
# initialization, regularization, activation function, and more in a single command:

# %%
net = slim.conv2d(inputs, 64, [11,11], 4, padding='SAME',
                 weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                 weights_regularizer=slim.l2_regularizer(0.0007), scope='conv1')

# %% [markdown]
# TF-Slim extends its elegance even beyond that, providing a clean way to replicate lay‐
# ers compactly by using the repeat , stack , and arg_scope commands.
# 
# repeat saves us the need to copy and paste the same line over and over so that, for
# example, instead of having this redundant duplication:

# %%
net = slim.conv2d(net, 128, [3,3], scope='conv1_1')
net = slim.conv2d(net, 128, [3,3], scope='conv1_2')
net = slim.conv2d(net, 128, [3,3], scope='conv1_3')
net = slim.conv2d(net, 128, [3,3], scope='conv1_4')
net = slim.conv2d(net, 128, [3,3], scope='conv1_5')

# equivalent:
net = slim.repeat(net, 5, slim.conv2d, 128, [3,3], scope='con1')

# %% [markdown]
# But this is viable only in cases where we have layers of the same size. When this does
# not hold, we can use the stack command, allowing us to concatenate layers of differ‐
# ent shapes. So, instead of this:

# %%
net = slim.conv2d(net, 64, [3,3], scope='conv1_1')
net = slim.conv2d(net, 64, [1,1], scope='conv1_2')
net = slim.conv2d(net, 128, [3,3], scope='conv1_3')
net = slim.conv2d(net, 128, [1,1], scope='conv1_4')
net = slim.conv2d(net, 256, [3,3], scope='conv1_5')

# equivalent:
net = slim.stack(net, slim.conv2d, [(64, [3,3]), (64,[1,1]),
                                    (128,[3,3]), (128,[1,1]),
                                    (256,[3,3])], scope='con1')

# %% [markdown]
# Finally, we also have a scoping mechanism referred to as arg_scope , allowing users to
# pass a set of shared arguments to each operation defined in the same scope. Say, for
# example, that we have four layers having the same activation function, initialization,
# regularization, and padding. We can then simply use the slim.arg_scope command,
# where we specify the shared arguments as in the following code:

# %%
with slim.arg_scope([slim.conv2d], padding='VALID',
                   activation_fn=tf.nn.relu,
                   weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                   weights_regularizer=slim.l2_regularizer(0.0007)):
    net = slim.conv2d(inputs, 64, [11,11], scope='conv1')
    net = slim.conv2d(inputs, 128, [11,11], padding='VALID', scope='conv2')
    net = slim.conv2d(inputs, 256, [11,11], scope='conv3')
    net = slim.conv2d(inputs, 256, [11,11], scope='conv4')
    

# %% [markdown]
# The individual arguments inside the arg_scope command can still be overwritten,
# and we can also nest one arg_scope inside another.
# 
# In these examples we used conv2d() : however, TF-Slim has many of the other stan‐
# dard methods for building neural networks. Table 7-4 lists some of the available
# options. For the full list, consult the documentation.
# 
# BiasAdd:     slim.bias_add()
# 
# BatchNorm:    slim.batch_norm()
# 
# Conv2d:    slim.conv2d()
# 
# Conv2dInPlane:    slim.conv2d_in_plane()
# 
# Conv2dTranspose (Deconv):   slim.conv2d_transpose()
# 
# FullyConnected:     slim.fully_connected()
# 
# AvgPool2D :    slim.avg_pool2d()
# 
# Dropout :    slim.dropout()
# 
# Flatten  :   slim.flatten()
# 
# MaxPool2D :   slim.max_pool2d()
# 
# OneHotEncoding   :  slim.one_hot_encoding()
# 
# SeparableConv2  :   slim.separable_conv2d()
# 
# UnitNorm   :  slim.unit_norm
# 
# %% [markdown]
# ## VGGNet
# To illustrate how convenient TF-Slim is for creating complex CNNs, we will build the
# VGG model by Karen Simonyan and Andrew Zisserman that was introduced in
# 2014 (see the upcoming note for more information). VGG serves as a good illustra‐
# tion of how a model with many layers can be created compactly using TF-Slim. Here
# we construct the 16-layer version: 13 convolution layers plus 3 fully connected layers.
# 
# Creating it, we take advantage of two of the features we’ve just mentioned:
# 1. We use the arg_scope feature since all of the convolution layers have the same
# activation function and the same regularization and initialization.
# 2. Many of the layers are exact duplicates of others, and therefore we also take
# advantage of the repeat command.
# 
# The result very compelling—the entire model is defined with just 16 lines of code:
# 
# ### VGG and the ImageNet Challenge
# The ImageNet project is a large database of images collected for the
# purpose of researching visual object recognition. As of 2016 it con‐
# tained over 10 million hand-annotated images.
# Each year (since 2010) a competition takes place called the Image‐
# Net Large Scale Visual Recognition Challenge (ILSVRC), where
# research teams try to automatically classify, detect, and localize
# objects and scenes in a subset of the ImageNet collection. In the
# 2012 challenge, dramatic progress occurred when a deep convolu‐
# tional neural net called AlexNet, created by Alex Krizhevsky, man‐
# aged to get a top 5 (top 5 chosen categories) classification error of
# only 15.4%, winning the competition by a large margin.
# Over the next couple of years the error rate kept falling, from
# ZFNet with 14.8% in 2013, to GoogLeNet (introducing the Incep‐
# tion module) with 6.7% in 2014, to ResNet with 3.6% in 2015. The
# Visual Geometry Group (VGG) was another CNN competitor in
# the 2014 competition that also achieved an impressive low error
# rate (7.3%). A lot of people prefer VGG over GoogLeNet because it
# has a nicer, simpler architecture.
# In VGG the only spatial dimensions used are very small 3×3 filters
# with a stride of 1 and a 2×2 max pooling, again with a stride of 1.
# Its superiority is achieved by the number of layers it uses, which is
# between 16 and 19.
# 

# %%
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                   activation_fn=tf.nn.relu,
                   weights_initializer=tf.truncated.normal_initializer(0.0, 0.01),
                   weights_regularizer=slim.l2_regularizer(0.0005)):
    
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3,3], scope='con1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    
    net = slim.repeat(inputs, 2, slim.conv2d, 128, [3,3], scope='con2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    
    net = slim.repeat(inputs, 2, slim.conv2d, 256, [3,3], scope='con3')
    net = slim.max_pool2d(net, [2,2], scope='pool3')

    net = slim.repeat(inputs, 2, slim.conv2d, 512, [3,3], scope='con4')
    net = slim.max_pool2d(net, [2,2], scope='pool4')

    net = slim.repeat(inputs, 2, slim.conv2d, 512, [3,3], scope='con5')
    net = slim.max_pool2d(net, [2,2], scope='pool5')

    net = slim.fully_connected(net, 4096, scope='fc6')
    net = slim.dropout(net,0.5, scope='dropout6')
    
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net,0.5, scope='dropout7')
    
    net = slim.fully_connected(net, 1000, scope='fc8')

# %% [markdown]
# ## Downloading and using a pretrained model
# First we need to clone the repository where the actual models will reside by running:
# git clone https://github.com/tensorflow/models
# Now we have the scripts we need for modeling on our computer, and we can use
# them by setting the path:
# 
# Next we will download the pretrained VGG-16 (16 layers) model—it is available on
# GitHub, as are other models, such as Inception, ResNet, and more:
# https://github.com/tensorflow/models/tree/master/research/slim
# The downloaded checkpoint file contains information about both the model and the
# variables. Now we want to load it and use it for classification of new images.
# 
# Then we prepare our input image, turning it into a reada‐
# ble TensorFlow format and performing a little pre-processing to make sure that it is
# resized to match the size of the images the model was trained on.
# 
# Now we create the model from the script we cloned earlier. We pass the model func‐
# tion the images and number of classes. The model has shared arguments; therefore,
# we call it using arg_scope , as we saw earlier, and use the vgg_arg_scope() function
# in the script to define the shared arguments. The function is shown in the following
# code snippet.
# 
# vgg_16() returns the logits (numeric values acting as evidence for each class), which
# we can then turn into probabilities by using tf.nn.softmax() . We use the argument
# is_training to indicate that we are interested in forming predictions rather than
# training:
# 
# Now, just before starting the session, we need to load the variables we downloaded
# using slim.assign_from_checkpoint_fn() , to which we pass the containing direc‐
# tory:
# 
# Finally, the main event—we run the session, load the variables, and feed in the images
# and the desired probabilities.
# 
# related resources:
# module import:
# https://stackoverflow.com/questions/4534438/typeerror-module-object-is-not-callable
# 
# http://iot-fans.xyz/2017/11/30/deeplearning/classify/

# %%
from tensorflow.contrib import slim
import sys
sys.path.append("/home/desktop/tensorflow/models/research/slim")

#from datasets. import dataset_utils
from nets import vgg
from preprocessing.vgg_preprocessing import preprocess_image

import os
import tensorflow as tf
#import urllib2

target_dir = '/home/desktop/tensorflow/checkpoints'

'''
# For a URL link, we can load the image as a string with urllib2
url=('http://54.68.5.226/car.jpg')
im_as_string = urllib2.urlopen(url).read()

im = tf.image.decode_jpeg(im_as_string, channels=3)
# or for png
im = tf.image.decode_png(im_as_string, channels=3)
'''

## for an image stored in our computer, we can create a queue of our filenames in the
# target directory, and then read the entire image file by using tf.WholeFileReader() :
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once('/home/desktop/tensorflow/images/lakeside.png'))
image_reader = tf.WholeFileReader()
_, image_file = image_reader.read(filename_queue)

image = tf.image.decode_png(image_file)

image_size = vgg.vgg_16.default_image_size

processed_im = preprocess_image(image, image_size,image_size)

processed_images = tf.expand_dims(processed_im, 0)

with slim.arg_scope(vgg.vgg_arg_scope()):
    logits, _ = vgg.vgg_16(processed_images, num_classes=1000,
                          is_training=False)
    
probabilities = tf.nn.softmax(logits)

def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       activation_fn=tf.nn.relu,
                       weight_regularizer=slim.l2_regularizer(weight_decay),
                       biases_initializer=tf.zeros.initializer):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc
    
load_vars = slim.assign_from_checkpoint_fn(os.path.join(target_dir, 'vgg_16.ckpt'),
                                          slim.get_model_variables('vgg_16'))


# %%



# %%
# We can get the class names by using the following lines:
from datasets.imagenet import create_readable_names_for_imagenet_labels

names = []
with tf.Session() as sess:
    load_vars(sess)
    network_input, probabilities = sess.run([processed_images,
                                            probabilities])
    probabilities = probabilities[0, 0:]
    names_ = create_readable_names_for_imagenet_labels()
    idxs = np.argsort(-probabilities)[:5]
    probs = probabilities[idxs]
    classes = np.array(names_.values())[idxs+1]
    
    for c, p in zip(classes, probs):
        print('class: ' + c + ' |prob: ' + str(p))


# %%



# %%



# %%


