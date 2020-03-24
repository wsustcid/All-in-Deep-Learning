# 1. 软件安装

## 1.1 安装准备

### 基本概念

显卡：（GPU）主流是Nvidia的GPU，深度学习本身需要大量计算。GPU的并行计算能力，在过去几年里恰当地满足了深度学习的需求。AMD的GPU基本没有什么支持，可以不用考虑。

驱动：没有显卡驱动，就不能识别GPU硬件，不能调用其计算资源。但是呢，Nvidia在Linux上的驱动安装特别麻烦，得屏蔽第三方显卡驱动。

CUDA：是Nvidia推出的只能用于自家GPU的并行计算框架。只有安装这个框架才能够进行复杂的并行计算。主流的深度学习框架也都是基于CUDA进行GPU并行加速的，几乎无一例外。还有一个叫做cudnn，是针对深度卷积神经网络的加速库。

### 版本选择

1. 安装前一定要仔细检索，查看各版本间的对应关系，

   详见:https://blog.csdn.net/MahoneSun/article/details/80809042

   - Latest TensorFlow supports cuda 8-10. cudnn 6-7.
   - Each TensorFlow binary has to work with the version of cuda and cudnn it was built with. If they don't match, you have to change either the TensorFlow binary or the Nvidia softwares.
   - Official `tensorflow-gpu` binaries (the one downloaded by pip or conda) are built with **cuda 9.0, cudnn 7 since TF 1.5**, and **cuda 10.0, cudnn 7 since TF 1.13.** These are written in the [release notes](https://github.com/tensorflow/tensorflow/releases). You have to use the matching version of cuda if using the official binaries.

2. 本次安装：

   ```
   CUDA 9.0 + cuDNN 7.1 + tensorflow-gpu 1.9
   ```

3. cuda & driver version
```
CUDA 10.0: 410.48
CUDA  9.2: 396.xx
CUDA  9.1: 390.xx (update)
CUDA  9.0: 384.xx
CUDA  8.0  375.xx (GA2)
CUDA  8.0: 367.4x
CUDA  7.5: 352.xx
CUDA  7.0: 346.xx
CUDA  6.5: 340.xx
CUDA  6.0: 331.xx
CUDA  5.5: 319.xx
CUDA  5.0: 304.xx
CUDA  4.2: 295.41
CUDA  4.1: 285.05.33
CUDA  4.0: 270.41.19
CUDA  3.2: 260.19.26
CUDA  3.1: 256.40
CUDA  3.0: 195.36.15
```

4. 解决相关安装包和ros的冲突：

.bashrc 文件中注释掉关于ros环境变量的source 语句

注意：
安装时不能激活conda等虚拟环境，否则nvidia-smi无法检测到

### 相关资源：
标准教程:https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07
https://blog.csdn.net/wf19930209/article/details/81877822

1. https://blog.csdn.net/X_kh_2001/article/details/81147073
2. https://blog.csdn.net/fdqw_sph/article/details/78745375
3. https://www.jianshu.com/p/4823d7b2ae6c
4. https://www.jianshu.com/p/ec46b2790ad1
5. https://www.cnblogs.com/kevinzhk/p/8451233.html

## 1.2 Install NVIDIA drives

### Windows

<https://blog.csdn.net/qq_37296487/article/details/83028394>

- 禁用nouveau驱动

```
sudo gedit /etc/modprobe.d/blacklist.conf
```

- 在文本最后添加：（禁用nouveau第三方驱动，之后也不需要改回来）

```
blacklist nouveau
options nouveau modeset=0
```

- 然后执行：

```
sudo update-initramfs -u
```

- 使用标准Ubuntu 仓库进行自动化安装
  首先，检测你的NVIDIA显卡型号和推荐的驱动程序的模型。在命令行中输入如下命令：

```
$ ubuntu-drivers devices
modalias : pci:v000010DEd00001B80sv00001043sd000085AAbc03sc00i00
vendor   : NVIDIA Corporation
driver   : xserver-xorg-video-nouveau - distro free builtin
driver   : nvidia-384 - distro non-free recommended
```

从输出结果可以看到，目前系统已连接Nvidia GeFrand GTX 显卡，建议安装驱动程序是 nvidia-384版本的驱动。如果您同意该建议，请再次使用Ubuntu驱动程序命令来安装所有推荐的驱动程序。

输入以下命令：

```
$ sudo ubuntu-drivers autoinstall
```

一旦安装结束，重新启动系统，你就完成了(可在设置中查看)。

```
nvidia-smi #若列出GPU的信息列表，表示驱动安装成功
nvidia-settings #若弹出设置对话框，亦表示驱动安装成功
```

另，重启后，执行：lsmod | grep nouveau。如果没有屏幕输出，说明禁用nouveau成功。



## 1.3 Install CUDA

- 查看你显卡驱动的版本，然后去Nvidia官网下载相对应版本的CUDA包，选择.run文件，因为deb会把之前显卡驱动的版本给覆盖掉，可能会出很多的问题，运行run文件

  - 注意：安装cuda时加上命令--no-opengl-libs，不加这个选项会进入循环登陆。例如：在终端运行指令 sudo sh cuda_8.0.27_linux.run --no-opengl-libs  （本机只有一个显卡，先不加试一下）

  ```
  cd 下载/tensorflow-gpu/
  chmod +x cuda_9.0.176_384.81_linux.run
  ./cuda_9.0.176_384.81_linux.run
  ```

  - 之后是一些提示信息,ctrl+c 直接结束后输入 accept。
    **接下来很重要的地方是在提示是否安装显卡驱动时,一定选择 no(之前安装过对应显卡版本的驱动)**
    其他各项提示选择是,并默认安装路径即可。提示有 y 的输入 y,没有则按 enter 键。安装完毕。
  - 安装过程中提示缺少几个.so文件，忽略了，下次可以尝试修复一下

- 安装完成后，在PATH里添加该CUDA路径，我选择在/etc里直接添加个全局的

  ```
  sudo vim /etc/profile
  ```

  然后在文件末添加如下两句：

  ```
  export PATH=/usr/local/cuda-9.0/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
  ```

  然后source一下，最好重启一下，不然后面导入tensorflow时会报错提示, 到时仍然出现错误请尝试下面的路径添加方案。目前先跳过

  ```python
  错误：ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory 
  问题：找不到cuda9.0的版本。 
  出现该错误的主要原因：cuda未安装或者cuda的版本有问题
  对于tensorflow 1.7版本，只接受cuda 9.0（9.1也不可以！），和cudnn 7.0，所以如果你安装了cuda9.1和cudnn7.1或以上版本，那么你需要重新安装9.0和7.0版本。
  
  安装完正确的版本后，确认你在你的~/.bashrc（或者~/.zshrc）文件中加入了下面环境变量
  
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64
  export PATH=$PATH:/usr/local/cuda-9.0/bin
  export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-9.0
  ```

  

- 测试是否安装成功： cd到CUDA样例里，NVIDIA_CUDA-9.0_Samples，然后其实可以全部make一下所有样例，时间比较长，但也可以cd到单独样例里然后make

  - 全部编译：

    在NVIDIA_CUDA-9.0_Samples目录下，

  ```
  make
  ```

  等待比较长的时间，卡死的话Ctrl+c重新来一次; 

  - 这个过程成功后，来测试一下：在NVIDIA_CUDA-9.0_Samples/1_Utilities/deviceQuery里，

  ```
  ./deviceQuery
  ```

  输出大概长这样就没问题了：PASS就是通过了。

  - 接着测试一下CUDA的通信组件是否成功了：

    ```
    cd ../bandwidthTest
    ./bandwidthTest
    ```

    输出大概长这样：PASS就是成功了。

## 1.4 Install cuDNN

1. 仍然去官网下载source，<https://developer.nvidia.com/rdp/cudnn-download>下载第一个，就是cuDNN v7.1.4 Library for Linux
2. 下载之后，

```
tar -zxvf cudnn-9.0-linux-x64-v7.1.tgz
```

解压，然后

```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

这样就成功了。测试的话去官网下载下来例程跑一下就OK了*-


## 1.5 Install TensorFlow with pip

### 1.5.1 Install the Python development environment on your system

Note: 

- 深度学习大多使用python3, 但ROS大多使用python2, 相关功能包大都是基于python2的, 千万不要更改系统默认的python版本，否则以后编译别人的功能包时系统默认调用你修改后的python版本对应的相关依赖，会导致无法编译。虽然有对应的解决方案，但会进入恶性循环。你需要使用python3时就输入python3, 安装python3相关的包时就pip3 install, 也最好不要升级pip, 一切保持默认。

1. Check if your Python environment is already configured: Requires Python 3.4, 3.5, or 3.6

```bsh
python3 --version
pip3 --version
virtualenv --version
```

If these packages are already installed, skip to the next step.
Otherwise, install [Python](https://www.python.org/), the [pip package manager](https://pip.pypa.io/en/stable/installing/), and [Virtualenv](https://virtualenv.pypa.io/en/stable/):

```bsh
sudo apt-get install python-pip python-dev python-virtualenv   # for Python 2.7
sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n
```

### 1.5.2 Create a virtual environment (recommended)

Python virtual environments are used to isolate package installation from the system.

Create a new virtual environment by choosing a Python interpreter and making a `./venv`directory to hold it:

```
mkdir ~/tensorflow  # somewhere to work out of
cd ~/tensorflow
# Choose one of the following Python environments for the /venv directory:
#virtualenv --system-site-packages venv            # Use python default (Python 2.7
#virtualenv --system-site-packages -p python3 venv # Use Python 3.n

virtualenv --no-site-packages -p python3 venv (recommended)
```

Activate the virtual environment using a shell-specific command:

```bsh
source ~/tensorflow/venv/bin/activate      # bash, sh, ksh, or zsh
source ~/tensorflow/venv/bin/activate.csh  # csh or tcsh
. ~/tensorflow/venv/bin/activate.fish      # fish
```

When virtualenv is active, your shell prompt is prefixed with `(venv)`.

Install packages within a virtual environment without affecting the host system setup. Start by upgrading `pip`:

```bsh
# pip install --upgrade pip
pip list  # show packages installed within the virtual environment
```

And to exit virtualenv later:

```bsh
deactivate  # don't exit until you're done using TensorFlow
```

### 1.5.3 Install the TensorFlow pip package

Choose one of the following TensorFlow packages to install [from PyPI](https://pypi.org/project/tensorflow/):

- `tensorflow` —Current release for CPU-only *(recommended for beginners)*
- `tensorflow-gpu` —Current release with [GPU support](https://www.tensorflow.org/install/gpu) *(Ubuntu and Windows)*
- `tf-nightly` —Nightly build for CPU-only *(unstable)*
- `tf-nightly-gpu` —Nightly build with [GPU support](https://www.tensorflow.org/install/gpu) *(unstable, Ubuntu and Windows)*

Package dependencies are automatically installed. These are listed in the [`setup.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)file under `REQUIRED_PACKAGES`.

```bsh
pip install tensorflow-gpu==1.9.0
pip install tensorflow-gpu==1.12.0
```

Verify the install:

```bsh
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

```
python
import tensorflow as tf
tf.__version__
# 查询tensorflow安装路径为:
tf.__path__
```

**Success:** TensorFlow is now installed. Read the [tutorials](https://www.tensorflow.org/tutorials) to get started.







## 1.6 Keras

https://keras-cn.readthedocs.io/en/latest/for_beginners/keras_linux/

```python
pip install keras
```

The pip install command also supports a --pre flag that will enable installing pre-releases and development releases.**

***Remark:***

- 2.2.4 版本BatchNormalization() 函数使用 axis=1时（针对通道在前的输入）会报错，这是版本bug，目前无法解决，建议降低到2.16版本：

  ```python
  pip install keras==2.1.6
  ```

  

## 1.7 Jupyter notebook

### 1.7.1 Installation

## Prerequisite: Python

While Jupyter runs code in many programming languages, Python is a requirement (Python 3.3 or greater, or Python 2.7) for installing the Jupyter Notebook itself.

## Installing Jupyter using Anaconda

We **strongly recommend** installing Python and Jupyter using the [Anaconda Distribution](https://www.anaconda.com/downloads), which includes Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.

First, download [Anaconda](https://www.anaconda.com/downloads). We recommend downloading Anaconda’s latest Python 3 version.

Second, install the version of Anaconda which you downloaded, following the instructions on the download page.

Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

```
jupyter notebook
```

See [Running the Notebook](https://jupyter.readthedocs.io/en/latest/running.html#running) for more details.

## Installing Jupyter with pip

As an existing or experienced Python user, you may wish to install Jupyter using Python’s package manager, pip, instead of Anaconda.

If you have Python 3 installed (which is recommended):

```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```

If you have Python 2 installed:

```
python -m pip install --upgrade pip
python -m pip install jupyter
```

Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

```
jupyter notebook
```

See [Running the Notebook](https://jupyter.readthedocs.io/en/latest/running.html#running) for more details.



#### 错误修复

出现 `UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 4: ordinal not in range(128)` 错误，是由于python2对于中文安装环境的bug.

解决方案：

```python
# default setting:
echo $LANGUAGE  # zh_CN:zh
export LANGUAGE=en_US
```



In case any Chinese users would like any actually useful help with this issue, while you should upgrade to using Python 3 versions of Jupyter if possible, if you cannot upgrade for whatever reason: try setting the environment variable `LANGUAGE=en_US` so that Jupyter doesn't try to use its Chinese translation, which clearly doesn't work well on Python 2. Or if that doesn't work, try one of `LC_ALL`, `LC_MESSAGES`, or `LANG`as described here: <https://docs.python.org/2/library/gettext.html#gettext.bindtextdomain>



Jupyter notebook添加python2 python3内核

1. ipython kernel install --name python2  
2. ipython kernel install --name python3  





## 1.8 

```bash
pip install -U scikit-learn
```

