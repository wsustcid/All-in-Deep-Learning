'''
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Version: 1.0.0
@Date: 2020-05-06 16:13:39
@LastEditTime: 2020-05-06 16:13:39
'''
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# TensorFlow as a numerical computation library using dataflow graphs.You will learn how to manage and create a graph, and be introduced to Tensor‐Flow’s “building blocks,” such as constants, placeholders, and Variables.
# 
# Roughly speaking, working with TensorFlow involves two main phases: 
# 1. constructing a graph and
# 2. executing it.

# %%
# Right after we import TensorFlow, a specific empty default graph is formed.
import tensorflow as tf 

# %% [markdown]
# ## 1. Creating a Graph, a Session and running it.
# 
# tf.add()  
# tf.multiply()   
# tf.subtract()   
# tf.divide()  
# tf.pow()   
# tf.mod()  
# tf.logical_and()  
# tf.greater()  
# tf.greater_equal()   
# tf.less_equal()  
# tf.less()  
# tf.negative()  
# tf.logical_not()  
# tf.abs()  
# tf.logical_or()  

# %%
# The contents of these variables should be regarded as the output of the operations
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)
# print(f)


# %%
#we launch the graph in a tf.Session
sess = tf.Session()
outs = sess.run(f)
sess.close()

print("out={}".format(outs))

# %% [markdown]
# ## 2. Constructing and Managing Our Graph

# %%
"""
As mentioned, as soon as we import TensorFlow, a default graph is automatically created 
for us. 
We can create additional graphs and control their association with some given 
operations. 
"""
# tf.Graph() creates a new graph, represented as a TensorFlow object.

print(tf.get_default_graph())

g = tf.Graph()
print(g)

# Since g hasn’t been assigned as the
# default graph, any operation we create will not be associated with it
a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())


# %%
# The with statement is used to wrap the execution of a block with
# methods defined by a context manager
# see more: https://blog.csdn.net/u012609509/article/details/72911564

g1 = tf.get_default_graph()
g2 = tf.Graph()

print(g1 is tf.get_default_graph())

with g2.as_default():
    print(g1 is tf.get_default_graph())
    
print(g1 is tf.get_default_graph())

# The "with" statement can also be used to start a session without having to explicitly close it.

# %% [markdown]
# ## 3. Fetches

# %%
'''
the variable it was assigned to as an argument to the sess.run() method. This argument
is called fetches , corresponding to the elements of the graph we wish to compute.
'''

with tf.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs = sess.run(fetches)
    
print("outs = {}".format(outs))
print(type(outs[0]))

# %% [markdown]
# ## 4. Data Types

# %%
c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)

# %% [markdown]
# ## 5. Casting
# performing an operation with two nonmatching data types will result in an exception.

# %%
x = tf.constant([1,2,3], name="x", dtype=tf.float32)
print(x.dtype)

x = tf.cast(x, tf.int64)
print(x.dtype)
print(x)
'''
with tf.Session() as sess:
    sess.run(x)

print(x)

'''

# %% [markdown]
# ## 6. Tensor Arrays and Shapes
# For example, a 1×1 tensor is a scalar, a 1×n tensor is a vector, an n×n tensor is a matrix, and an n×n×n tensor is just a three-dimensional array. This, of course, generalizes to any dimension. TensorFlow regards all the data units that flow in the graph as tensors, whether they are multidimensional arrays, vectors, matrices, or scalars.
# 
# To initialize high-dimensional arrays, we can use Python lists or NumPy arrays as inputs. In the following example, we use as inputs a 2×3 matrix using a Python list and then a 3D NumPy array of size 2×2×3 (two matrices of size 2×3)
# 
# tf.constant(value)      
# tf.fill(shape, value)   
# tf.zeros(shape)  
# tf.zeros_like(tensor)  
# tf.ones(shape)   
# tf.ones_like(tensor)   
# tf.random_normal(shape,mean, stddev)   
# tf.truncated_normal(shape, mean,stddev)    
# tf.random_uniform(shape, minval,maxval)    
# tf.random_shuffle(tensor)   

# %%
import numpy as np

c = tf.constant([[1,2,3],
                 [4,5,6]])
print("Python list input: {}".format(c.get_shape()))

c = tf.constant(np.array([
                  [[1,2,3],
                   [4,5,6]], 
        
                  [[6,5,4],
                   [3,2,1]]
                 ]))
print("3d numpy array input:{}".format(c.get_shape()))

# %% [markdown]
# We can generate random numbers from a normal distribution
# using tf.random.normal() , passing the shape, mean, and standard deviation as the
# first, second, and third arguments, respectively. Another two examples for useful ran‐
# dom initializers are the truncated normal that, as its name implies, cuts off all values
# below and above two standard deviations from the mean, and the uniform initializer
# that samples values uniformly within some interval [a,b)
# 
# One example is the sequence generator tf.linspace(a, b,
# n) that creates n evenly spaced values from a to b .

# %%
'''
tf.InteractiveSession() allows you to replace the usual tf.Ses
sion() , so that you don’t need a variable holding the session for
running ops.
'''
sess = tf.InteractiveSession()
c = tf.linspace(0.0, 4.0, 5)
print("The content of 'c':\n {} \n".format(c.eval()))
sess.close()
# print(c)


# %%
A = tf.constant([[1,2,3],
                 [4,5,6]])
print(A.get_shape())

x = tf.constant([1,0,1])
print(x.get_shape())

'''
In order to multiply them, we need to add a dimension to x , transforming it from a
1D vector to a 2D single-column matrix.
'''
x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(A,x)

sess = tf.InteractiveSession()
print('matmul result: {}'.format(b.eval()))
sess.close()


# %%
'''
Prefixes are especially useful when we would like to divide a graph into subgraphs
with some semantic meaning.
'''
with tf.Graph().as_default(): # without this line, the name of c1 and c2 will change after each run
    c1 = tf.constant(4, dtype=tf.float64,name='c')
    c2 = tf.constant(4, dtype=tf.int32,name='c')
    print(c1.name)
    print(c2.name)

    with tf.name_scope("prefix_name"):
        c3 = tf.constant(4,dtype=tf.int32,name='c')
        c4 = tf.constant(4,dtype=tf.float32,name='c')

    print(c3.name)
    print(c4.name)

# %% [markdown]
# ## 7. Variables and Placeholders
# The optimization process serves to **tune the parameters** of some given model. For
# that purpose, TensorFlow uses special objects called Variables.
# 
# Using Variables is done in two stages. 
# - First we call the tf.Variable() function in order to create a Variable and define what value it will be initialized with. 
# - We then have to explicitly perform an initialization operation by running the session with the tf.global_variables_initializer() method, which allocates the memory for the Variable and sets its initial values.
# 
# TensorFlow, has designated built-in structures for feeding input values. These structures are called placeholders.
# - Placeholders can be thought of as empty Variables that will be filled with data later on. 
# - We use them by first constructing our graph and only when it is executed feeding them with the input data.
# - Placeholders have an optional shape argument. If a shape is not fed or is passed as None , then the placeholder can be fed with data of any size. 
# - It is common to use None for the dimension of a matrix that corresponds to the number of samples (usually rows), while having the length of the features (usually columns) fixed.
# 
# The input data is passed to the session.run() method as a dictionary, where each key corresponds to a placeholder variable name, and the matching values are the data values given in the form of a list or a NumPy array.

# %%
'''
Note that if we run the code again, we see that a new variable is created each time, as
indicated by the automatic concatenation of _1 to its name:

This could be very inefficient when we want to reuse the model (complex models
could have many variables!); for example, when we wish to feed it with several differ‐
ent inputs. To reuse the same variable, we can use the tf.get_variables() function
instead of tf.Variable() . More on this can be found in “Model Structuring” on page
203 of the appendix.
'''

var = tf.Variable(tf.random_normal((1,5),0,1), name='var')
print("pre run: {}".format(var))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    post_var = sess.run(var)

print("\npost run: {}".format(post_var))


# %%
x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32, shape=(5,10))
    w = tf.placeholder(tf.float32, shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)
    xwb = xw + b
    
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        out1 = sess.run(xwb, feed_dict={x: x_data, w: w_data})
        out2= sess.run(s, feed_dict={x: x_data, w: w_data})
    
    print("xwb = {}".format(xwb))
    print("out1 = {}".format(out1))
    print("out2 = {}".format(out2))
    # 只运行out2,xwb 能否print?

# %% [markdown]
# ## 8. Optimization
# cross-entropy:
# https://github.com/rdipietro/jupyter-notebooks/blob/master/friendly-intro-to-cross-entropy-loss/A%20Friendly%20Introduction%20to%20Cross-Entropy%20Loss.ipynb
# 
# ### Example 1: Linear regression

# %%
import numpy as np
# creating data and simulate results
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2

noise = np.random.randn(1,2000)*0.1

y_data = np.matmul(w_real,x_data.T) + b_real + noise


# %%
num_steps = 10

g = tf.Graph()
wb = []

with g.as_default():
    x = tf.placeholder(tf.float32, shape=[None,3])
    y_true = tf.placeholder(tf.float32, shape=None)
    
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]], dtype=tf.float32, name='weights')
        b = tf.Variable(0, dtype=tf.float32, name='bias')
        y_pred = tf.matmul(w, tf.transpose(x)) + b
        
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
    
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)
        
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(num_steps):
            sess.run(train, feed_dict={x: x_data, y_true: y_data})
            
            if (step%5==0):
                print(step, sess.run([w,b]))
                wb.append(sess.run([w,b]))
        
        print(num_steps, sess.run([w,b]))

# %% [markdown]
# ### Example 2: logistic regression. 
# Again we wish to retrieve the weights and bias compo‐
# nents in a simulated data setting, this time in a logistic regression framework. Here
# the linear component w T x + b is the input of a nonlinear function called the logistic
# function. What it effectively does is squash the values of the linear part into the inter‐
# val [0, 1].
# We then regard these values as probabilities from which binary yes/1 or no/0 out‐
# comes are generated.This is the nondeterministic (noisy) part of the model.

# %%


