## 1. Schedule and Syllabus

Spring 2017

| Event Type          | Date               | Description                                                  | Course Materials                                             |
| ------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Lecture 1           | Tuesday  April 4   | **Course Introduction**  Computer vision overview  Historical context  Course logistics | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture1.pdf) [[video\]](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 2           | Thursday  April 6  | **Image Classification**  The data-driven approach  K-nearest neighbor  Linear classification I | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture2.pdf) [[video\]](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[python/numpy tutorial\]](http://cs231n.github.io/python-numpy-tutorial) [[image classification notes\]](http://cs231n.github.io/classification) [[linear classification notes\]](http://cs231n.github.io/linear-classify) |
| Lecture 3           | Tuesday  April 11  | **Loss Functions and Optimization**  Linear classification II Higher-level representations, image features Optimization, stochastic gradient descent | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture3.pdf) [[video\]](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[linear classification notes\]](http://cs231n.github.io/linear-classify) [[optimization notes\]](http://cs231n.github.io/optimization-1) |
| Lecture 4           | Thursday  April 13 | **Introduction to Neural Networks**  Backpropagation Multi-layer Perceptrons The neural viewpoint | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) [[video\]](https://www.youtube.com/watch?v=d14TUNcbn1k&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [[backprop notes\]](http://cs231n.github.io/optimization-2) [[linear backprop example\]](http://cs231n.stanford.edu/2017/handouts/linear-backprop.pdf) [[derivatives notes\]](http://cs231n.stanford.edu/2017/handouts/derivatives.pdf) (optional)  [[Efficient BackProp\]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (optional) related: [[1\]](http://colah.github.io/posts/2015-08-Backprop/), [[2\]](http://neuralnetworksanddeeplearning.com/chap2.html), [[3\]](https://www.youtube.com/watch?v=q0pm3BrIUFo) (optional) |
| Lecture 5           | Tuesday  April 18  | **Convolutional Neural Networks**  History  Convolution and pooling  ConvNets outside vision | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf) [[video\]](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [ConvNet notes](http://cs231n.github.io/convolutional-networks/) |
| Lecture 6           | Thursday  April 20 | **Training Neural Networks, part I**  Activation functions, initialization, dropout, batch normalization | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf) [[video\]](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [Neural Nets notes 1](http://cs231n.github.io/neural-networks-1/) [Neural Nets notes 2](http://cs231n.github.io/neural-networks-2/) [Neural Nets notes 3](http://cs231n.github.io/neural-networks-3/) tips/tricks: [[1\]](http://research.microsoft.com/pubs/192769/tricks-2012.pdf), [[2\]](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), [[3\]](http://arxiv.org/pdf/1206.5533v2.pdf) (optional)  [Deep Learning [Nature\]](http://www.nature.com/nature/journal/v521/n7553/full/nature14539.html) (optional) |
| A1 Due              | Thursday  April 20 | **Assignment #1 due**  kNN, SVM, SoftMax, two-layer network  | [[Assignment #1\]](http://cs231n.github.io/assignments2017/assignment1/) |
| Lecture 7           | Tuesday  April 25  | **Training Neural Networks, part II**  Update rules, ensembles, data augmentation, transfer learning | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf) [[video\]](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [Neural Nets notes 3](http://cs231n.github.io/neural-networks-3/) |
| Proposal due        | Tuesday  April 25  | Couse Project Proposal due                                   | [[proposal description\]](http://cs231n.stanford.edu/project.html) |
| Lecture 8           | Thursday  April 27 | **Deep Learning Software**  Caffe, Torch, Theano, TensorFlow, Keras, PyTorch, etc | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture8.pdf) [[video\]](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 9           | Tuesday  May 2     | **CNN Architectures**  AlexNet, VGG, GoogLeNet, ResNet, etc  | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf) [[video\]](https://www.youtube.com/watch?v=DAOcjicFr1Y&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGGNet](https://arxiv.org/abs/1409.1556), [GoogLeNet](https://arxiv.org/abs/1409.4842), [ResNet](https://arxiv.org/abs/1512.03385) |
| Lecture 10          | Thursday  May 4    | **Recurrent Neural Networks**  RNN, LSTM, GRU  Language modeling  Image captioning, visual question answering  Soft attention | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf) [[video\]](https://www.youtube.com/watch?v=6niqTuYFZLQ&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [DL book RNN chapter](http://www.deeplearningbook.org/contents/rnn.html) (optional) [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086), [char-rnn](https://github.com/karpathy/char-rnn), [neuraltalk2](https://github.com/karpathy/neuraltalk2) |
| A2 Due              | Thursday  May 4    | **Assignment #2 due**  Neural networks, ConvNets             | [[Assignment #2\]](http://cs231n.github.io/assignments2017/assignment2/) |
| Midterm             | Tuesday  May 9     | **In-class midterm** Location: [Various](https://piazza.com/class/j0vi72697xc49k?cid=1272) (**not** our usual classroom) |                                                              |
| Lecture 11          | Thursday  May 11   | **Detection and Segmentation**  Semantic segmentation  Object detection  Instance segmentation | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf) [[video\]](https://www.youtube.com/watch?v=nDPWywWRIRo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 12          | Tuesday  May 16    | **Visualizing and Understanding**  Feature visualization and inversion  Adversarial examples  DeepDream and style transfer | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf) [[video\]](https://www.youtube.com/watch?v=6wcs6szJWMY&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)  [DeepDream](https://github.com/google/deepdream) [neural-style](https://github.com/jcjohnson/neural-style) [fast-neural-style](https://github.com/jcjohnson/fast-neural-style) |
| Milestone           | Tuesday  May 16    | Course Project Milestone due                                 |                                                              |
| Lecture 13          | Thursday  May 18   | **Generative Models**  PixelRNN/CNN  Variational Autoencoders  Generative Adversarial Networks | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) [[video\]](https://www.youtube.com/watch?v=5WoItGTWV54&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 14          | Tuesday  May 23    | **Deep Reinforcement Learning**  Policy gradients, hard attention  Q-Learning, Actor-Critic | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture14.pdf) [[video\]](https://www.youtube.com/watch?v=lvoHnicueoE&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Guest Lecture       | Thursday  May 25   | **Invited Talk: Song Han**  Efficient Methods and Hardware for Deep Learning | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture15.pdf) [[video\]](https://www.youtube.com/watch?v=eZdOkDtYMoo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| A3 Due              | Friday  May 26     | **Assignment #3 due**                                        | [[Assignment #3\]](http://cs231n.github.io/assignments2017/assignment3/) |
| Guest Lecture       | Tuesday  May 30    | **Invited Talk: Ian Goodfellow**  Adversarial Examples and Adversarial Training | [[slides\]](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture16.pdf) [[video\]](https://www.youtube.com/watch?v=CIfsB_EYsVI&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) |
| Lecture 16          | Thursday  June 1   | Student spotlight talks, conclusions                         | [slides]                                                     |
| Poster Due          | Monday  June 5     | **Poster PDF due**                                           | [[poster description\]](http://cs231n.stanford.edu/project.html) |
| Poster Presentation | Tuesday June 6     |                                                              |                                                              |
| Final Project Due   | **Monday June 12** | Final course project due date                                | [[reports\]](http://cs231n.stanford.edu/2017/reports.html)   |

## 2. Preparation

**I strongly suggest that you setup a new computer which has been installed Ubuntu 16.04 and do not change any default settings before staring this deep learning tutorial !**

### 2.1Markdown

1. You should be familiar with basic markdown syntax to read .md files and write your own tutorials.

2. Install typora to read and wirte md files.

```
wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -
# add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update
# install typora
sudo apt-get install typora
```

3. lantern and chrome should also be installed.

   https://github.com/getlantern/lantern

   https://support.google.com/chrome/answer/95346?co=GENIE.Platform%3DDesktop&hl=zh-Hans

### 2.2 Default settings of Ubuntu 16.04

#### pip

- 由于ubuntu系统自带python2.7（默认）和python3.5，所以不需要自己安装python

  可以使用python -V和python3 -V查看已安装python版本。

- 在不同版本的python中ubuntu默认没有安装pip，所以需要自己手动安装pip。在不同版本中安装pip，可以使用以下命令：

  ```
  sudo apt-get install python-pip # recommended!
  
  sudo apt-get install python3-pip
  ```

- 安装完成后可以使用pip -V (# pip 8.1.1 from /usr/lib/python2.7/dist-packages (python 2.7)) 和pip3 -V查看看装的pip版本。

- 在使用pip安装其他库时，默认的python版本可以直接使用pip install XXXX

  另外的python版本可以使用python3 -m pip install XXXX 或pip3 install XXXX

### 2.3 virtualenv

VirtualEnv用于在一台机器上创建多个独立的Python虚拟运行环境，多个Python环境相互独立，互不影响，它能够：

- 在没有权限的情况下安装新套件
- 不同应用可以使用不同的套件版本
- 套件升级不影响其他应用

虚拟环境是在Python解释器上的一个私有复制，你可以在一个隔绝的环境下安装packages，不会影响到你系统中全局的Python解释器。虚拟环境非常有用，因为它可以防止系统出现包管理混乱和版本冲突的问题。为每个应用程序创建一个虚拟环境可以确保应用程序只能访问它们自己使用的包，从而全局解释器只作为一个源且依然整洁干净去更多的虚拟环境。另一个好处是，虚拟环境不需要管理员权限。

**安装**

```
pip install virtualenv
```



**基本使用**

1.为一个工程创建一个虚拟环境：

```
cd my_project_dir
virtualenv venv　　#venv为虚拟环境目录名，目录名自定义
```

virtualenv venv 将会在当前的目录中创建一个文件夹，包含了Python可执行文件，以及 pip 库的一份拷贝，这样就能安装其他包了。虚拟环境的名字（此例中是 venv ）可以是任意的；若省略名字将会把文件均放在当前目录。

在任何你运行命令的目录中，这会创建Python的拷贝，并将之放在叫做 venv 的文件中。

你可以选择使用一个Python解释器：

```
virtualenv -p /usr/bin/python2.7 venv　 　# -p参数指定Python解释器程序路径
```

这将会使用 /usr/bin/python2.7 中的Python解释器。

 2.要开始使用虚拟环境，其需要被激活：

```
source venv/bin/activate
```

从现在起，任何你使用pip安装的包将会放在 venv 文件夹中，与全局安装的Python隔绝开。

像平常一样安装包，比如：

```
pip install requests
```

3.如果你在虚拟环境中暂时完成了工作，则可以停用它：

```
. venv/bin/deactivate
```

这将会回到系统默认的Python解释器，包括已安装的库也会回到默认的。

4. 要删除一个虚拟环境，只需删除它的文件夹。（执行 rm -rf venv ）。

