# Learning Pytorch step by step

## 1. Get started

### 1.1 Installation

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

## 1.2 快速入门

**本节目的：**

- 用最快的时间，掌握pytorch最核心的用法；
- 知道训练一个网络的完整代码的构成部分，整体写代码的流程心中有数；
- 能够阅读懂别人的代码。(别人的代码中有的地方可能还看不懂，比如一些高级操作，主体部分能够看懂，其他少量操作可以通过检索学习)

**具体做法：**

1. 查看中文快速入门教程，如官网的60min入门教程（注意学技术刚开始要看中文资料，解决疑问看英文）

   - 官方中文文档地址：<https://pytorch.apachecn.org/>

     先完成如下部分：PyTorch 深度学习: 60 分钟极速入门

     - [什么是 PyTorch？](https://pytorch.apachecn.org/docs/1.0/blitz_tensor_tutorial.html)
     - [Autograd：自动求导](https://pytorch.apachecn.org/docs/1.0/blitz_autograd_tutorial.html)
     - [神经网络](https://pytorch.apachecn.org/docs/1.0/blitz_neural_networks_tutorial.html)
     - [训练分类器](https://pytorch.apachecn.org/docs/1.0/blitz_cifar10_tutorial.html)
     - [可选：数据并行处理](https://pytorch.apachecn.org/docs/1.0/blitz_data_parallel_tutorial.html)

### 1.3 上手

本节目的：

- 通过实际运行/阅读一些典型例子，**并整理**，作为自己的代码库，要写自己的代码时，就从这里例子中寻找，或直接修改，或模仿着写；
- 不会写代码的原因是脑子里没有相关内容，不知从何下手，所以认为自己“不会写”。所以这一步的关键是要找大量的例子（和自己研究内容相关的），快速阅读，**整理保存下来。**不要想着用的时候搜，你都不知道有这个东西，怎么能搜到呢？得先见过，才能搜到。
- 首先可以先把下面的教程通读整理一遍；我的做法是：去保存这些教程的github，把整个项目克隆下来，因为这些教程都是用markdown写的，所以我根据目录，逐个读这些markdown,边读边整理，主要也就是删掉废话，调调格式，这样读完一遍也就整理完了，而不是采用读网页，然后复制粘贴的方式。这样速度较快，很快就能形成自己的文档，便于后期根据不同文档主题进行补充。可以采用一个主题一个文档的方式，对这个教程重新归纳整理（参考我的keras教程）
- 下面例子比较少，也可以自己搜一些github上star较多的pytorch example汇总。作为自己的“代码库”
- 最后，为了检验自己的学习成果，可以自己尝试复现一些经典网络，先从复现经典网络结构做起，然后在结合数据集，自己写数据接口。
- 下面的内容和自己很不相关的可以不看，但也可以扫一眼。

- [数据加载和处理教程](https://pytorch.apachecn.org/docs/1.0/data_loading_tutorial.html)
- [用例子学习 PyTorch](https://pytorch.apachecn.org/docs/1.0/pytorch_with_examples.html)
- [迁移学习教程](https://pytorch.apachecn.org/docs/1.0/transfer_learning_tutorial.html)
- [混合前端的 seq2seq 模型部署](https://pytorch.apachecn.org/docs/1.0/deploy_seq2seq_hybrid_frontend_tutorial.html)
- [Saving and Loading Models](https://pytorch.apachecn.org/docs/1.0/saving_loading_models.html)
- [What is torch.nn really?](https://pytorch.apachecn.org/docs/1.0/nn_tutorial.html)

- 图像
  - [Torchvision 模型微调](https://pytorch.apachecn.org/docs/1.0/finetuning_torchvision_models_tutorial.html)
  - [空间变换器网络教程](https://pytorch.apachecn.org/docs/1.0/spatial_transformer_tutorial.html)
  - [使用 PyTorch 进行图像风格转换](https://pytorch.apachecn.org/docs/1.0/neural_style_tutorial.html)
  - [对抗性示例生成](https://pytorch.apachecn.org/docs/1.0/fgsm_tutorial.html)
  - [使用 ONNX 将模型从 PyTorch 传输到 Caffe2 和移动端](https://pytorch.apachecn.org/docs/1.0/super_resolution_with_caffe2.html)
- 文本（不是自己领域的可以先不读）
  - [聊天机器人教程](https://pytorch.apachecn.org/docs/1.0/chatbot_tutorial.html)
  - [使用字符级别特征的 RNN 网络生成姓氏](https://pytorch.apachecn.org/docs/1.0/char_rnn_generation_tutorial.html)
  - [使用字符级别特征的 RNN 网络进行姓氏分类](https://pytorch.apachecn.org/docs/1.0/char_rnn_classification_tutorial.html)
  - Deep Learning for NLP with Pytorch
    - [PyTorch 介绍](https://pytorch.apachecn.org/docs/1.0/nlp_pytorch_tutorial.html)
    - [使用 PyTorch 进行深度学习](https://pytorch.apachecn.org/docs/1.0/nlp_deep_learning_tutorial.html)
    - [Word Embeddings: Encoding Lexical Semantics](https://pytorch.apachecn.org/docs/1.0/nlp_word_embeddings_tutorial.html)
    - [序列模型和 LSTM 网络](https://pytorch.apachecn.org/docs/1.0/nlp_sequence_models_tutorial.html)
    - [Advanced: Making Dynamic Decisions and the Bi-LSTM CRF](https://pytorch.apachecn.org/docs/1.0/nlp_advanced_tutorial.html)
  - [基于注意力机制的 seq2seq 神经网络翻译](https://pytorch.apachecn.org/docs/1.0/seq2seq_translation_tutorial.html)
- 生成
  - [DCGAN Tutorial](https://pytorch.apachecn.org/docs/1.0/dcgan_faces_tutorial.html)
- 强化学习
  - [Reinforcement Learning (DQN) Tutorial](https://pytorch.apachecn.org/docs/1.0/reinforcement_q_learning.html)
- 扩展 PyTorch
  - [用 numpy 和 scipy 创建扩展](https://pytorch.apachecn.org/docs/1.0/numpy_extensions_tutorial.html)
  - [Custom C++ and CUDA Extensions](https://pytorch.apachecn.org/docs/1.0/cpp_extension.html)
  - [Extending TorchScript with Custom C++ Operators](https://pytorch.apachecn.org/docs/1.0/torch_script_custom_ops.html)
- 生产性使用
  - [Writing Distributed Applications with PyTorch](https://pytorch.apachecn.org/docs/1.0/dist_tuto.html)
  - [使用 Amazon AWS 进行分布式训练](https://pytorch.apachecn.org/docs/1.0/aws_distributed_training_tutorial.html)
  - [ONNX 现场演示教程](https://pytorch.apachecn.org/docs/1.0/ONNXLive.html)
  - [在 C++ 中加载 PYTORCH 模型](https://pytorch.apachecn.org/docs/1.0/cpp_export.html)
- 其它语言中的 PyTorch
  - [使用 PyTorch C++ 前端](https://pytorch.apachecn.org/docs/1.0/cpp_frontend.html)

- - - https://pytorch.apachecn.org/docs/1.0/torchvision_utils.html)



## 1.3 强化

**本节目的：**

- 这时你就要开始做自己的项目了，但做项目时会发现很多现成的例子最是不太适用，无法继续照搬，这时你就要加深对一些语句的理解，探究底层他真正实现了什么操作，原理是什么，根据自己的问题，去下面的教程中或源码中或网络检索中寻找答案。
- 这时候也要注意积累自己的训练网络的技巧。
- 也没有理由再说自己不会写了，因为从顶层到底层，你都知道去哪找答案，你可以什么都不会，什么都记不住，但你已经有了自己的答案库，然后不断扩充自己的答案库。这个答案库的核心就是你在第二部分整理的核心教程；然后根据自己的实际需求，比如需要新的数据接口，上网找一找资源，理解了，跑通了，整合到上一部分的 “数据加载与处理”教程中。这样不断扩充，半年后，你就发现，你虽然一开始遇到问题还是不会，但自己思考后，就可以想出解决方案了。

中文文档

- 注解
  - [自动求导机制](https://pytorch.apachecn.org/docs/1.0/notes_autograd.html)
  - [广播语义](https://pytorch.apachecn.org/docs/1.0/notes_broadcasting.html)
  - [CUDA 语义](https://pytorch.apachecn.org/docs/1.0/notes_cuda.html)
  - [Extending PyTorch](https://pytorch.apachecn.org/docs/1.0/notes_extending.html)
  - [Frequently Asked Questions](https://pytorch.apachecn.org/docs/1.0/notes_faq.html)
  - [Multiprocessing best practices](https://pytorch.apachecn.org/docs/1.0/notes_multiprocessing.html)
  - [Reproducibility](https://pytorch.apachecn.org/docs/1.0/notes_randomness.html)
  - [Serialization semantics](https://pytorch.apachecn.org/docs/1.0/notes_serialization.html)
  - [Windows FAQ](https://pytorch.apachecn.org/docs/1.0/notes_windows.html)
- 包参考
  - torch
    - [Tensors](https://pytorch.apachecn.org/docs/1.0/torch_tensors.html)
    - [Random sampling](https://pytorch.apachecn.org/docs/1.0/torch_random_sampling.html)
    - [Serialization, Parallelism, Utilities](https://pytorch.apachecn.org/docs/1.0/torch_serialization_parallelism_utilities.html)
    - Math operations
      - [Pointwise Ops](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_pointwise_ops.html)
      - [Reduction Ops](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_reduction_ops.html)
      - [Comparison Ops](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_comparison_ops.html)
      - [Spectral Ops](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_spectral_ops.html)
      - [Other Operations](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_other_ops.html)
      - [BLAS and LAPACK Operations](https://pytorch.apachecn.org/docs/1.0/torch_math_operations_blas_lapack_ops.html)
  - [torch.Tensor](https://pytorch.apachecn.org/docs/1.0/tensors.html)
  - [Tensor Attributes](https://pytorch.apachecn.org/docs/1.0/tensor_attributes.html)
  - [数据类型信息](https://pytorch.apachecn.org/docs/1.0/type_info.html)
  - [torch.sparse](https://pytorch.apachecn.org/docs/1.0/sparse.html)
  - [torch.cuda](https://pytorch.apachecn.org/docs/1.0/cuda.html)
  - [torch.Storage](https://pytorch.apachecn.org/docs/1.0/storage.html)
  - [torch.nn](https://pytorch.apachecn.org/docs/1.0/nn.html)
  - [torch.nn.functional](https://pytorch.apachecn.org/docs/1.0/nn_functional.html)
  - [torch.nn.init](https://pytorch.apachecn.org/docs/1.0/nn_init.html)
  - [torch.optim](https://pytorch.apachecn.org/docs/1.0/optim.html)
  - [Automatic differentiation package - torch.autograd](https://pytorch.apachecn.org/docs/1.0/autograd.html)
  - [Distributed communication package - torch.distributed](https://pytorch.apachecn.org/docs/1.0/distributed.html)
  - [Probability distributions - torch.distributions](https://pytorch.apachecn.org/docs/1.0/distributions.html)
  - [Torch Script](https://pytorch.apachecn.org/docs/1.0/jit.html)
  - [多进程包 - torch.multiprocessing](https://pytorch.apachecn.org/docs/1.0/multiprocessing.html)
  - [torch.utils.bottleneck](https://pytorch.apachecn.org/docs/1.0/bottleneck.html)
  - [torch.utils.checkpoint](https://pytorch.apachecn.org/docs/1.0/checkpoint.html)
  - [torch.utils.cpp_extension](https://pytorch.apachecn.org/docs/1.0/docs_cpp_extension.html)
  - [torch.utils.data](https://pytorch.apachecn.org/docs/1.0/data.html)
  - [torch.utils.dlpack](https://pytorch.apachecn.org/docs/1.0/dlpack.html)
  - [torch.hub](https://pytorch.apachecn.org/docs/1.0/hub.html)
  - [torch.utils.model_zoo](https://pytorch.apachecn.org/docs/1.0/model_zoo.html)
  - [torch.onnx](https://pytorch.apachecn.org/docs/1.0/onnx.html)
  - [Distributed communication package (deprecated) - torch.distributed.deprecated](https://pytorch.apachecn.org/docs/1.0/distributed_deprecated.html)
- torchvision 参考
  - [torchvision.datasets](https://pytorch.apachecn.org/docs/1.0/torchvision_datasets.html)
  - [torchvision.models](https://pytorch.apachecn.org/docs/1.0/torchvision_models.html)
  - [torchvision.transforms](https://pytorch.apachecn.org/docs/1.0/torchvision_transforms.html)
  - [torchvision.utils](