今年深入研究深度图像处理，最近正好发现3D卷积神经网络，感觉这个可能会是个不错的新的研究方向。做下总结......

首先，近年来已经出现了一些相应的研究，文末会给出引用。

关于CNN不做笔记，网上相应讲解很多。

这里针对图像识别领域的3D卷积神经网络做些笔记和总结讨论。仍在学习中......

CNN的强大毋庸置疑，但是目前是以2D为输入，图像识别领域也就是以RGB图像输入。然而近年行为动作的识别以及深度传感器的普及，出现了一些3D CNN模型的研究。比较突出的有TPAMI2013关于人体行为的识别和CVPR2015的基于3D卷积神经网络的手势识别。



![img](https://pic2.zhimg.com/80/64518f54335a22366aa275f5595a1c05_hd.png)

3D CNN architecture [1]





![img](https://pic2.zhimg.com/80/79b98bb4c9cae9c83a5ddb156573828d_hd.png)

3D CNN architecture [2]



3D CNN模型的主要特性有：

1）通过3D卷积操作核去提取数据的时间和空间特征，在CNN的卷积层使用3D卷积。

2）3D CNN模型可以同时处理多幅图片，达到附加信息的提取。

3）融合时空域的预测。

继续学习......

References:

[1] SJi, MYang, and KYu. 3D convolutional neural networks for human action recognition. IEEE Transactions on Pattern Analysis & Machine Intelligence, 35(1):221–231, 2013.

[2] Pavlo Molchanov, Shalini Gupta, Kihwan Kim, Jan Kautz, and Santa Clara. Hand Gesture Recognition with 3D Convolutional Neural Networks. pages 1–7, 2015.





https://blog.csdn.net/weicao1990/article/details/80283443



https://www.zhihu.com/question/266352189



https://github.com/keras-team/keras/issues/1359

https://github.com/ellisdg/3DUnetCNN