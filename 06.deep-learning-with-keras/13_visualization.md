
## 模型可视化

`keras.utils.vis_utils` 模块提供了一些绘制 Keras 模型的实用功能(使用 `graphviz`)。

以下实例，将绘制一张模型图，并保存为文件：
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` 有 4 个可选参数:

- `show_shapes` (默认为 False) 控制是否在图中输出各层的尺寸。
- `show_layer_names` (默认为 True) 控制是否在图中显示每一层的名字。
- `expand_dim`（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
- `dpi`（默认为 96）控制图像 dpi。

此外，你也可以直接取得 `pydot.Graph` 对象并自己渲染它。
例如，ipython notebook 中的可视化实例如下：

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

----

## 训练历史可视化

Keras `Model` 上的 `fit()` 方法返回一个 `History` 对象。`History.history` 属性是一个记录了连续迭代的训练/验证（如果存在）损失值和评估值的字典。这里是一个简单的使用 `matplotlib` 来生成训练/验证集的损失和准确率图表的例子：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```



## 这些深度学习网络图画图工具，总有一款适合你！

全能言有三 [睿慕课](javascript:void(0);) *4月22日*

​    

[![img](https://mmbiz.qpic.cn/mmbiz_jpg/ibVw1NGwIHfiaH9YiaRHwF3cPzTNU0pPQHNZBjubUq7A5FrNZsSVDDkiclb394DX5SlHRDhlyWOSGaukyqtkhWuLYQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489798&idx=1&sn=af58416aa84d4e76d91840297ab2e485&chksm=ebe6b232dc913b24369f69f3a40e86c32b8bb2f7077f0043658fa47f392c4f6683fc130e4bac&scene=21#wechat_redirect)



作者 · 龙鹏-言有三

来源 · 知乎

编辑 · Tony





— 正文 —



> 本文我们聊聊如何才能画出炫酷高大上的神经网络图，下面是常用的几种工具。





## 

## **▌****1.NN-SVG**



这个工具可以非常方便的画出各种类型的图，是下面这位小哥哥开发的，来自于麻省理工学院弗兰克尔生物工程实验室, 该实验室开发可视化和机器学习工具用于分析生物数据。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/vJe7ErxcLmiaIib91uXdIgzQQgCZDalOT3iaibmDmIFIrl1xjeJygBhTNm7uLLSVYf3hBLDrr2ek8DNMhxslxoCjiaQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

来源于：知乎



github地址：https://github.com/zfrenchee



画图工具体验地址：http://alexlenail.me/NN-SVG/



可以绘制的图包括以节点形式展示的 FCNN style，这个**特别适合传统的全连接神经网络的绘制**。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



以平铺网络结构展示的 LeNet style，用二维的方式，适合查看每一层 featuremap 的大小和通道数目。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



以三维 block 形式展现的 AlexNet style，可以更加真实地展示卷积过程中高维数据的尺度的变化，目前只支持卷积层和全连接层。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



**这个工具可以导出非常高清的 SVG 图，值得体验。**





## **▌****2.PlotNeuralNet**



这个工具是萨尔大学计算机科学专业的一个学生开发的，一看就像计算机学院的嘛。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



首先我们看看效果，其github链接如下，将近 4000 star：



https://github.com/HarisIqbal88/PlotNeuralNet



看看人家这个 fcn-8 的可视化图，颜值奇高。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



**使用的门槛相对来说就高一些了，用 LaTex 语言编辑，所以可以发挥的空间就大**了，你看下面这个 softmax 层，这就是会写代码的优势了。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎





其中的一部分代码是这样的，会写吗。



*pic[shift={(0,0,0)}] at (0,0,0) {Box={name=crp1,caption=SoftmaxLoss: $E_mathcal{S}$ ,%fill={rgb:blue,1.5;red,3.5;green,3.5;white,5},opacity=0.5,height=20,width=7,depth=20}};*



相似的工具还有：https://github.com/jettan/tikz_cnn





## **▌****3.ConvNetDraw**



ConvNetDraw 是一个使用配置命令的 CNN 神经网络画图工具，开发者是香港的一位程序员，Cédric cbovar。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



采用如下的语法直接配置网络，可以简单调整 x，y，z 等 3 个维度，github 链接如下：

https://cbovar.github.io/ConvNetDraw/



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



使用方法如上图所示，只需输入模型结构中各层的参数配置。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



挺好用的。不过它目标分辨率太低了，放大之后不清晰，达不到印刷的需求。





## **▌****4.Draw_Convnet**



这一个工具名叫 draw_convnet，由 Borealis 公司的员工 Gavin Weiguang Ding 提供。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



**简单直接，是纯用 python 代码画图的**，



https://github.com/gwding/draw_convnet



看看画的图如下，核心工具是 matplotlib，图不酷炫，但是好在规规矩矩，可以严格控制，论文用挺合适的。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



类似的工具还有：https://github.com/yu4u/convnet-drawer





## **▌****5.Netscope**



下面要说的是这个，我最常用的，caffe 的网络结构可视化工具，大名鼎鼎的 netscope，由斯坦福 AI Lab 的 Saumitro Dasgupta 开发，找不到照片就不放了，地址如下：



https://github.com/ethereon/netscope



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



**左边放配置文件，右边出图，非常方便进行网络参数的调整和可视化**。这种方式好就好在各个网络层之间的连接非常的方便。





## **▌****其他**



再分享一个有意思的，不是画什么正经图，但是把权重都画出来了。



http://scs.ryerson.ca/~aharley/vis/conv/



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



看了这么多，有人已经在偷偷笑了，上 PPT 呀，想要什么有什么，想怎么画就怎么画。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



不过妹子呢？怎么不来开发一个粉色系的可视化工具呢？类似于这样的



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

来源于：知乎



那么，你都用什么画呢？欢迎留言分享一下！





· In the end ·





好课推荐：

**免费报名 | 增强型物理交互式机器人躯体设计与应用**

点击下方👇图片，查看

[![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489833&idx=1&sn=65f3693e1e29ca242fe16ca0c80cd33c&chksm=ebe6b21ddc913b0bfa2286994d20b37f4410fd3b4b3292840b6b052b51377d946bda6a14f54d&scene=21#wechat_redirect)



推荐阅读：

[文章·人形机器人，穷途末路还是光明未来？](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489157&idx=1&sn=5b88187db14f962afb80de96cb954e00&chksm=ebe6bdb1dc9134a7719a44730f26d05501bcb14f80e4f8dcc18ef984a6585c7e03d8808fc963&scene=21#wechat_redirect)

[文章·卡尔曼滤波：从入门到精通](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489146&idx=1&sn=c1a335983b206c77768297c36d5a8649&chksm=ebe6bd4edc9134580d48ded1a13df1c9933b7fb2df976a2ed63304d630f5b974418a61415b85&scene=21#wechat_redirect)

[推荐 · 机器人设计不走弯路，从工作视角全面讲解！](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489039&idx=1&sn=fd136f57bd2b0807ea586b8099001470&chksm=ebe6bd3bdc91342d4a30eeb78ca52a73206bdca4601b75327ba615c0a7b6de73e4e58f8d124c&scene=21#wechat_redirect)

[资料 · 2018全球工程前沿报告，文末免费领取](http://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247488937&idx=1&sn=c6f2fd2e617c2f0fb7b021e9b771d723&chksm=ebe6be9ddc91378b789fd5e0d0ed23321f4c000f991508eaf978e583946a36f3859e94d9eeec&scene=21#wechat_redirect)





![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**觉得有用，麻烦给个在看~**  **![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)**

文章转载自公众号

![有三AI](http://wx.qlogo.cn/mmhead/Q3auHgzwzM7VtedxYZ1AqMMNx43dvVicoCIlKYgsrbkGsm5jhib6qyicw/0) **有三AI** 

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzI4NTcyMDE1NQ==&mid=2247489923&idx=1&sn=b4d7270de88f185d3cd856307ff95b7c&chksm=ebe6b2b7dc913ba14b88e8324219e36978c2963e3dda1c5782b535b28e919a17179c56a34793&mpshare=1&scene=24&srcid=&pass_ticket=L9MmezP0euAiloYx0ZpV0zn%2FH1NZipsFj7Qrzod8QKwih5XAiPWVLPugpr7yoxrL##)