## 一、标准化（Z-Score），或者去除均值和方差缩放

公式为：(X-mean)/std  计算时对每个属性/每列分别进行。

将数据按期属性（按列进行）减去其均值，并处以其方差。得到的结果是，对于每个属性/每列来说所有数据都聚集在0附近，方差为1。

实现时，有两种不同的方式：

- 使用sklearn.preprocessing.scale()函数，可以直接将给定数据进行标准化。

  ```python
  >>> from sklearn import preprocessing
  >>> import numpy as np
  >>> X = np.array([[ 1., -1.,  2.],
  ...               [ 2.,  0.,  0.],
  ...               [ 0.,  1., -1.]])
  >>> X_scaled = preprocessing.scale(X)
   
  >>> X_scaled                                          
  array([[ 0.  ..., -1.22...,  1.33...],
         [ 1.22...,  0.  ..., -0.26...],
         [-1.22...,  1.22..., -1.06...]])
   
  >>>#处理后数据的均值和方差
  >>> X_scaled.mean(axis=0)
  array([ 0.,  0.,  0.])
   
  >>> X_scaled.std(axis=0)
  array([ 1.,  1.,  1.])
  ```

  

- 使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。

  ```python
  >>> scaler = preprocessing.StandardScaler().fit(X)
  >>> scaler
  StandardScaler(copy=True, with_mean=True, with_std=True)
   
  >>> scaler.mean_                                      
  array([ 1. ...,  0. ...,  0.33...])
   
  >>> scaler.std_                                       
  array([ 0.81...,  0.81...,  1.24...])
   
  >>> scaler.transform(X)                               
  array([[ 0.  ..., -1.22...,  1.33...],
         [ 1.22...,  0.  ..., -0.26...],
         [-1.22...,  1.22..., -1.06...]])
   
   
  >>>#可以直接使用训练集对测试集数据进行转换
  >>> scaler.transform([[-1.,  1., 0.]])                
  array([[-2.44...,  1.22..., -0.26...]])
  ```

  

## 二、将属性缩放到一个指定范围(归一化？)

除了上述介绍的方法之外，另一种常用的方法是将属性缩放到一个指定的最大和最小值（通常是1-0）之间，这可以通过preprocessing.MinMaxScaler类实现。

使用这种方法的目的包括：

1、对于方差非常小的属性可以增强其稳定性。

2、维持稀疏矩阵中为0的条目。

```python
>>> X_train = np.array([[ 1., -1.,  2.],
...                     [ 2.,  0.,  0.],
...                     [ 0.,  1., -1.]])
...
>>> min_max_scaler = preprocessing.MinMaxScaler()
>>> X_train_minmax = min_max_scaler.fit_transform(X_train)
>>> X_train_minmax
array([[ 0.5       ,  0.        ,  1.        ],
       [ 1.        ,  0.5       ,  0.33333333],
       [ 0.        ,  1.        ,  0.        ]])
 
>>> #将相同的缩放应用到测试集数据中
>>> X_test = np.array([[ -3., -1.,  4.]])
>>> X_test_minmax = min_max_scaler.transform(X_test)
>>> X_test_minmax
array([[-1.5       ,  0.        ,  1.66666667]])
 
 
>>> #缩放因子等属性
>>> min_max_scaler.scale_                             
array([ 0.5       ,  0.5       ,  0.33...])
 
>>> min_max_scaler.min_                               
array([ 0.        ,  0.5       ,  0.33...])
```

当然，在构造类对象的时候也可以直接指定最大最小值的范围：feature_range=(min, max)，此时应用的公式变为：



X_std=(X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))

X_scaled=X_std/(max-min)+min



**问题：**
scikit-learn中fit_transform()与transform()到底有什么区别，能不能混用？

二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
fit_transform(partData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该partData进行转换transform，从而实现数据的标准化、归一化等等。。
根据对之前部分fit的整体指标，对剩余的数据（restData）使用同样的均值、方差、最大最小值等指标进行转换transform(restData)，从而保证part、rest处理方式相同。
必须先用fit_transform(partData)，之后再transform(restData)
如果直接transform(partData)，程序会报错
如果fit_transfrom(partData)后，使用fit_transform(restData)而不用transform(restData)，虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。
**实验：**

使用preprocessing.MinMaxScaler()对象对数据进行归一化。原理是：(x-xMin)/(xMax - xMin)，从而将所有数据映射到【0,1】区间。

```python
import numpy as np 

from sklearn.preprocessing import MinMaxScaler

data = np.array(np.random.randint(-100,100,24).reshape(6,4))

data
Out[55]: 
array([[ 68, -63, -31, -10],
       [ 49, -49,  73,  18],
       [ 46,  65,  75, -78],
       [-72,  30,  90, -80],
       [ 95, -88,  79, -49],
       [ 34, -81,  57,  83]])

train = data[:4]

test = data[4:]

train
Out[58]: 
array([[ 68, -63, -31, -10],
       [ 49, -49,  73,  18],
       [ 46,  65,  75, -78],
       [-72,  30,  90, -80]])

test
Out[59]: 
array([[ 95, -88,  79, -49],
       [ 34, -81,  57,  83]])

minmaxTransformer = MinMaxScaler(feature_range=(0,1))

#先对train用fit_transformer(),包括拟合fit找到xMin,xMax,再transform归一化
train_transformer = minmaxTransformer.fit_transform(train)

#根据train集合的xMin，xMax,对test集合进行归一化transform.
#(如果test中的某个值比之前的xMin还要小，依然用原来的xMin；同理如果test中的某个值比之前的xMax还要大，依然用原来的xMax.
#所以，对test集合用同样的xMin和xMax，**有可能不再映射到【0,1】**)
test_transformer = minmaxTransformer.transform(test)

train_transformer
Out[64]: 
array([[ 1.        ,  0.        ,  0.        ,  0.71428571],
       [ 0.86428571,  0.109375  ,  0.85950413,  1.        ],
       [ 0.84285714,  1.        ,  0.87603306,  0.02040816],
       [ 0.        ,  0.7265625 ,  1.        ,  0.        ]])

test_transformer
Out[65]: 
array([[ 1.19285714, -0.1953125 ,  0.90909091,  0.31632653],
       [ 0.75714286, -0.140625  ,  0.72727273,  1.66326531]])

#如果少了fit环节，直接transform(partData),则会报错

minmaxTransformer = MinMaxScaler(feature_range=(0,1))

train_transformer2 = minmaxTransformer.transform(train)
Traceback (most recent call last):

  File "<ipython-input-68-a2aeaf2132be>", line 1, in <module>
    train_transformer2 = minmaxTransformer.transform(train)

  File "D:\Program Files\Anaconda3\lib\site-packages\sklearn\preprocessing\data.py", line 352, in transform
    check_is_fitted(self, 'scale_')

  File "D:\Program Files\Anaconda3\lib\site-packages\sklearn\utils\validation.py", line 690, in check_is_fitted
    raise _NotFittedError(msg % {'name': type(estimator).__name__})

NotFittedError: This MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.

#如果对test也用fit_transform(),则结果跟之前不一样。对于许多机器学习算法来说，对于train和test的处理应该统一。

test_transformer2 = minmaxTransformer.fit_transform(test)

test_transformer2
Out[71]: 
array([[ 1.,  0.,  1.,  0.],
       [ 0.,  1.,  0.,  1.]])

test_transformer
Out[72]: 
array([[ 1.19285714, -0.1953125 ,  0.90909091,  0.31632653],
       [ 0.75714286, -0.140625  ,  0.72727273,  1.66326531]])
```





## 三、正则化（Normalization）

正则化的过程是将每个样本缩放到单位范数（每个样本的范数为1），如果后面要使用如二次型（点积）或者其它核方法计算两个样本之间的相似性这个方法会很有用。

Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。

​             p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p

该方法主要应用于文本分类和聚类中。例如，对于两个TF-IDF向量的l2-norm进行点积，就可以得到这两个向量的余弦相似性。

1. 可以使用preprocessing.normalize()函数对指定数据进行转换：

   ```python
   >>> X = [[ 1., -1.,  2.],
   ...      [ 2.,  0.,  0.],
   ...      [ 0.,  1., -1.]]
   >>> X_normalized = preprocessing.normalize(X, norm='l2')
    
   >>> X_normalized                                      
   array([[ 0.40..., -0.40...,  0.81...],
          [ 1.  ...,  0.  ...,  0.  ...],
          [ 0.  ...,  0.70..., -0.70...]])
   
   ```

   

2. 可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换:

   ```python
   >>> normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
   >>> normalizer
   Normalizer(copy=True, norm='l2')
    
   >>>
   >>> normalizer.transform(X)                            
   array([[ 0.40..., -0.40...,  0.81...],
          [ 1.  ...,  0.  ...,  0.  ...],
          [ 0.  ...,  0.70..., -0.70...]])
    
   >>> normalizer.transform([[-1.,  1., 0.]])             
   array([[-0.70...,  0.70...,  0.  ...]])
   ```

   