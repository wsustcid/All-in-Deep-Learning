# Learning PyTorch step by step

## 1. Quick Start

**Objects：**

- 用最快的时间，掌握pytorch最核心的用法；
- 知道训练一个网络的完整代码的构成部分，整体写代码的流程心中有数；
- 能够阅读懂别人的代码。(别人的代码中有的地方可能还看不懂，比如一些高级操作，主体部分能够看懂，其他少量操作可以通过检索学习)

**Steps：**

1. 查看中文快速入门教程，如官网的60min入门教程（注意学技术刚开始可以看中文资料，解决疑问看英文）

   - 官方中文文档地址：<https://pytorch.apachecn.org/>

     先完成如下部分：PyTorch 深度学习: 60 分钟极速入门

     - [什么是 PyTorch？](https://pytorch.apachecn.org/docs/1.0/blitz_tensor_tutorial.html)
     - [Autograd：自动求导](https://pytorch.apachecn.org/docs/1.0/blitz_autograd_tutorial.html)
     - [神经网络](https://pytorch.apachecn.org/docs/1.0/blitz_neural_networks_tutorial.html)
     - [训练分类器](https://pytorch.apachecn.org/docs/1.0/blitz_cifar10_tutorial.html)
     - [可选：数据并行处理](https://pytorch.apachecn.org/docs/1.0/blitz_data_parallel_tutorial.html)

## 2. Start to Use

**Objects：**

- 通过阅读一些典型例子（官方文档是首选），**并整理保存**，积累自己的代码库。（写自己的代码时，从这里例子中或直接修改，或模仿创造）
- 也可以自己搜一些github上star较多的pytorch example汇总，并在实际使用的过程中不断扩充自己的“代码库”

**Steps:**

- 保存官方文档（clone it from github)， 边读边整理。可以采用一个主题一个文档的方式，对这个教程重新归纳整理，便于后期根据不同文档主题进行补充。
- 独立复现一些经典网络结构，然后在结合数据集，自己写**数据接口**。
- 形成自己代码框架.





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



## 3. Skilled to use

**Objects：**

- 读源码。加深对一些语句的理解，探究底层实现与原理。（以根据自己的项目需求实现一些高级操作）
- 注意积累自己的训练网络的技巧。



**Steps**

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