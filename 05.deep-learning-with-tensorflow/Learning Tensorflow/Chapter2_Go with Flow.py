'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:06:12
@LastEditTime: 2020-05-06 16:11:22
'''
"""Classifying MNIST handwritten digits with somax regression

"""

import tensorflow as tf

# ## MNIST
# In this example we will use a simple classifier called **softmax regression**.
# Rather than down‐loading the MNIST dataset (freely available at http://yann.lecun.com/exdb/mnist/) and loading it into our program, we use a built-in utility for retrieving the dataset on the fly. **Such utilities exist for most popular datasets**.


from tensorflow.examples.tutorials.mnist import input_data



data_dir = 'datasets'
num_steps = 1000 # The number of steps we will make in the gradient descent approach
minibatch_size = 100 # controls the number of examples to use for each step



data = input_data.read_data_sets(data_dir, one_hot=True)



# a variable is an element manipulated by the computation
# a placeholder has to be supplied when triggering it
x = tf.placeholder(tf.float32, [None, 784]) # 28x28
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x,W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels= y_true))

learning_rate = 0.5
gd_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))



with tf.Session() as sess:
    # train
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        batch_xs, batch_ys = data.train.next_batch(minibatch_size)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
    
    # test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})



print("Accuracy:{:.4}%".format(ans*100))





