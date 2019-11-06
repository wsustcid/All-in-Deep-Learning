# 安装

```python
# 之前已安装GPU驱动已经cuda,和virtualenv
# 创建基于python3的虚拟环境,虚拟环境命名为pytorch(可自定义)
virtualenv --no-site-packages -p python3 pytorch
# 激活虚拟环境
source pytorch/bin/activate
# 根据你自己的软件环境，按照官网https://pytorch.org/给出的指令进行安装对应版本的pytorch
# 我电脑的cuda 版本是9.0，官网最新版本的pytroch最低要求9.2，懒得升级了，我直接安装cpu版本的
pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    
# 查看是否安装成功
python
>>> import torch
>>> print(torch.__version__)
1.3.0+cpu
>>> print(torch.cuda.is_available())
False

```

![](/media/ubuntu16/F/Deep-learning-tutorial/deep-learning-tutorial-with-pytorch/assets/install.png)

# 入门

1. 完成官网的60min入门教程<https://pytorch.org/tutorials/>

<https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>

2. 参考 [pytorch/examples](https://link.zhihu.com/?target=https%3A//github.com/pytorch/examples) 实现一个最简单的例子(比如训练mnist )。

# 熟悉

通读doc [PyTorch doc](https://link.zhihu.com/?target=http%3A//pytorch.org/docs/) 尤其是[autograd](https://link.zhihu.com/?target=http%3A//pytorch.org/docs/notes/autograd.html)的机制，和[nn.module](https://link.zhihu.com/?target=http%3A//pytorch.org/docs/nn.html%23torch.nn.Module) *,optim 等*相关内容。文档现在已经很完善，而且绝大部分文档都是**作者亲自写**的，质量很高。(也有一些官网的中文翻译<https://pytorch-cn.readthedocs.io/zh/latest/>) 

快速阅读整理成自己文档

# 强化

**第四步** **论坛讨论** [PyTorch Forums](https://link.zhihu.com/?target=https%3A//discuss.pytorch.org/) 。论坛很活跃，而且质量很高，pytorch的维护者(作者)回帖很及时的。每天刷一刷帖可以少走很多弯路，避开许多陷阱,消除很多思维惯性.尤其看看那些阅读量高的贴，刷帖能从作者那里学会如何写出bug-free clean and elegant 的代码。如果自己遇到问题可以先搜索一下，一般都能找到解决方案，找不到的话大胆提问，大家都很热心的。

**第五步** **阅读源代码** fork pytorch，pytorch-vision等。相比其他框架，pytorch代码量不大，而且抽象层次没有那么多，很容易读懂的。通过阅读代码可以了解函数和类的机制，此外它的很多函数,模型,模块的实现方法都如教科书般经典。还可以关注官方仓库的issue/pull request, 了解pytorch开发进展，以及避坑。