---
layout: page
mathjax: true
permalink: /assignments2017/assignment1/
---

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- develop proficiency in writing efficient **vectorized** code with numpy
- implement and apply a k-Nearest Neighbor (**kNN**) classifier
- implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- implement and apply a **Softmax** classifier
- implement and apply a **Two layer neural network** classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using **higher-level representations** than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

## Setup
### Working locally
Get the code as a zip file [here](http://cs231n.stanford.edu/assignments/2017/spring1617_assignment1.zip). As for the dependencies:

**Installing Python 3.5+:**
You can find instructions for Ubuntu [here](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-ubuntu-16-04).

Ubuntu 16.04 ships with both Python 3 and Python 2 pre-installed. To make sure that our versions are up-to-date, let’s update and upgrade the system with `apt-get`:

```
sudo apt-get update
sudo apt-get -y upgrade
```

The `-y` flag will confirm that we are agreeing for all items to be installed, but depending on your version of Linux, you may need to confirm additional prompts as your system updates and upgrades.

Once the process is complete, we can check the version of Python 3 that is installed in the system by typing:

```
python3 -V
```

You will receive output in the terminal window that will let you know the version number. The version number may vary, but it will look similar to this:

```
OutputPython 3.5.2
```

To manage software packages for Python, let’s install **pip**:

**Note: 时刻注意区分python2 和python3 的指令，需要用什么python环境，就根据相关指令安装，不可混淆！**

```
sudo apt-get install -y python3-pip
```

比如查看通过不同python指令安装的pip:

```
pip --version
pip 8.1.1 from /usr/lib/python2.7/dist-packages (python 2.7)
pip3 --version
pip 8.1.1 from /usr/lib/python3/dist-packages (python 3.5)
```

A tool for use with Python, **pip** installs and manages programming packages we may want to use in our development projects. You can install Python packages by typing:

```
pip3 install package_name
```

Here, `package_name` can refer to any Python package or library, such as Django for web development or NumPy for scientific computing. So if you would like to install NumPy, you can do so with the command `pip3 install numpy`.

There are a few more packages and development tools to install to ensure that we have a robust set-up for our programming environment:

```
sudo apt-get install build-essential libssl-dev libffi-dev python-dev
```

Once Python is set up, and pip and other tools are installed, we can set up a virtual environment for our development projects.



## Step 2 — Setting Up a Virtual Environment

Virtual environments enable you to have an isolated space on your computer for Python projects, ensuring that each of your projects can have its own set of dependencies that won’t disrupt any of your other projects.

Setting up a programming environment provides us with greater control over our Python projects and over how different versions of packages are handled. This is especially important when working with third-party packages.

You can set up as many Python programming environments as you want. Each environment is basically a directory or folder in your computer that has a few scripts in it to make it act as an environment.

We need to first install the **venv** module, part of the standard Python 3 library, so that we can create virtual environments. Let’s install venv by typing:

```
sudo apt-get install -y python3-venv
```

With this installed, we are ready to create environments. Let’s choose which directory we would like to put our Python programming environments in, or we can create a new directory with `mkdir`, as in:

```
mkdir environments
cd environments
```

Once you are in the directory where you would like the environments to live, you can create an environment by running the following command:

```
python3 -m venv my_env
```

Essentially, this sets up a new directory that contains a few items which we can view with the `ls`command:

```
ls my_env
Output
bin include lib lib64 pyvenv.cfg share
```

Together, these files work to make sure that your projects are isolated from the broader context of your local machine, so that system files and project files don’t mix. This is good practice for version control and to ensure that each of your projects has access to the particular packages that it needs. Python Wheels, a built-package format for Python that can speed up your software production by reducing the number of times you need to compile, will be in the Ubuntu 16.04 `share` directory.

To use this environment, you need to activate it, which you can do by typing the following command that calls the activate script:

```
source my_env/bin/activate
```

Your prompt will now be prefixed with the name of your environment, in this case it is called my_env. Your prefix may look somewhat different, but the name of your environment in parentheses should be the first thing you see on your line:

This prefix lets us know that the environment my_env is currently active, meaning that when we create programs here they will use only this particular environment’s settings and packages.

**Note:** Within the virtual environment, you can use the command `python` instead of `python3`, and `pip`instead of `pip3` if you would prefer. If you use Python 3 on your machine outside of an environment, you will need to use the `python3` and `pip3` commands exclusively. 

After following these steps, your virtual environment is ready to use.

## Step 3 — Creating a Simple Program

Now that we have our virtual environment set up, let’s create a simple “Hello, World!” program. This will make sure that our environment is working and gives us the opportunity to become more familiar with Python if we aren’t already.

To do this, we’ll open up a command-line text editor such as nano and create a new file:

```
nano hello.py
```

Once the text file opens up in the terminal window we’ll type out our program:

```python
print("Hello, World!")
```

Exit nano by typing the `control` and `x` keys, and when prompted to save the file press `y`.

Once you exit out of nano and return to your shell, let’s run the program:

```
python hello.py
```

The hello.py program that you just created should cause your terminal to produce the following output:

```
OutputHello, World!
```

To leave the environment, simply type the command `deactivate` and you will return to your original directory.



**Virtual environment:**
If you decide to work locally, we recommend using [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for the project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies for the code are installed globally on your machine. To set up a virtual environment, run the following:

```bash
cd assignment1
virtualenv -p /usr/bin/python3 env       # Create a virtual environment (python3)

source env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install dependencies (start lantern)
# Work on the assignment for a while ...
deactivate                       # Exit the virtual environment
```

```
When there is a error reporting that Could not find a version that satisfies the requirement site==0.0.1...

change it into "sites==0.0.1", and then more problems appear. I fixed it by following:

$ sudo apt-get install libncurses5-dev
$ pip install -r requirements.txt
and there are no errors appear.
```



Note that every time you want to work on the assignment, you should run 

**`source .env/bin/activate`**

 (from within your `assignment1` folder) to re-activate the virtual environment, and `deactivate` again whenever you are done**.**

- some test output:

```
pip -V
pip 18.1 from /media/desktop/新加卷/DeepLearning/CS231n/cs231n.github.io/assignments/assignment1/env/lib/python3.5/site-packages/pip (python 3.5)
$ pip3 -V
pip 18.1 from /media/desktop/新加卷/DeepLearning/CS231n/cs231n.github.io/assignments/assignment1/env/lib/python3.5/site-packages/pip (python 3.5)
```



### Download data:
Once you have the starter code (regardless of which method you choose above), you will need to download the CIFAR-10 dataset.
Run the following from the `assignment1` directory:

```bash
cd cs231n/datasets
./get_datasets.sh
```

### Start IPython:
After you have the CIFAR-10 data, you should start the IPython notebook server from the
`assignment1` directory, with the `jupyter notebook` command. 

If you are unfamiliar with IPython, you can also refer to our [IPython tutorial](http://cs231n.github.io/ipython-tutorial/).

**Note: if your virtual environment installed correctly (as per the assignment handouts), then you shouldn’t have to install from the install instructions on the website. Just remember to run source .env/bin/activate in your assignment folder.**

Once you have it installed, start it with this command:

```
jupyter notebook
```

- An IPython notebook is made up of a number of **cells**. Each cell can contain Python code. You can execute a cell by clicking on it and pressing `Shift-Enter`. 

### Some Notes

**NOTE 1:** This year, the `assignment1` code has been tested to be compatible with python versions `2.7`, `3.5`, `3.6` (it may work with other versions of `3.x`, but we won't be officially supporting them). You will need to make sure that during your `virtualenv` setup that the correct version of `python` is used. You can confirm your python version by (1) activating your virtualenv and (2) running `which python`.

### Submitting your work:
Whether you work on the assignment locally or using Google Cloud, once you are done
working run the `collectSubmission.sh` script; this will produce a file called
`assignment1.zip`. Please submit this file on [Canvas](https://canvas.stanford.edu/courses/66461/).

### Q1: k-Nearest Neighbor classifier (20 points)

The IPython Notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine (25 points)

The IPython Notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier (20 points)

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network (25 points)
The IPython Notebook **two\_layer\_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features (10 points)

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

### Q6: Cool Bonus: Do something extra! (+10 points)

Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or anything fun to look at? Or maybe you can experiment with a spin on the loss function? If you try out something cool we'll give you up to 10 extra points and may feature your results in the lecture.
