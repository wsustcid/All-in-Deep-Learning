### Jupyter notebook

要想使用python2,建议还是建立一个python2的虚拟环境

#### 增加内核（暂时不可用）

本例的Jupyter安装在Python3下，以增加Python2内核为例。

首先确认在Python3下已安装了内核：

```
ipython kernel install --user#orpython3 -m ipykernel install --user
```

然后确保Python2下安装了ipykernel

```
sudo pip2 install -U ipykernel
```

然后运行如下命令：

```
python2 -m ipykernel install --user
```

但是上述方法会出现错误：

[ERROR: ipykernel requires Python version 3.4 or above](https://stackoverflow.com/questions/52733094/error-ipykernel-requires-python-version-3-4-or-above)

I am using Ubuntu 16.04 lts. My default python binary is python2.7. When I am trying to install ipykernel for hydrogen in atom editor, with the following com

Starting with version 5.0 of the [kernel](https://ipykernel.readthedocs.io/en/latest/changelog.html#id3), and version 6.0 of [IPython](https://ipython.readthedocs.io/en/stable/whatsnew/version6.html#ipython-6-0), compatibility with Python 2 was dropped. As far as I know, the only solution is to install an earlier release.

In order to have Python 2.7 available in the Jupyter Notebook I installed IPython 5.7, and ipykernel 4.10. If you want to install earlier releases of IPython or ipykernel you can do the following:

- Uninstall IPython

```
pip uninstall ipython
```

- Reinstall IPython

```
python2 -m pip install ipython==5.7 --user
```

- Install ipykernel

```
python2 -m pip install ipykernel==4.10 --user
```

