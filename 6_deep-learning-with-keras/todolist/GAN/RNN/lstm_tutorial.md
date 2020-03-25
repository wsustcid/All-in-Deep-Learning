## Long Short-Term Memory (LSTM)

Long Short-Term Memory (LSTM) Recurrent Neural Networks are designed for sequence prediction problems and are a state-of-the-art deep learning technique for challenging prediction problems.

Here’s how to get started with LSTMs in Python:

- Step 1

  : Discover the promise of LSTMs.

  - [The Promise of Recurrent Neural Networks for Time Series Forecasting](https://machinelearningmastery.com/promise-recurrent-neural-networks-time-series-forecasting/)

- Step 2

  : Discover where LSTMs are useful.

  - [Making Predictions with Sequences](https://machinelearningmastery.com/sequence-prediction/)
  - [A Gentle Introduction to Long Short-Term Memory Networks by the Experts](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)
  - [Introduction to Models for Sequence Prediction](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)

- Step 3

  : Discover how to use LSTMs on your project.

  - [The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)
  - [Long Short-Term Memory Networks (Mini-Course)](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)
  - [Long Short-Term Memory Networks With Python](https://machinelearningmastery.com/lstms-with-python/) (***my book***)

You can see all [LSTM posts here](https://machinelearningmastery.com/category/lstm/). Below is a selection of some of the most popular tutorials using LSTMs in Python with the Keras deep learning library.

#### Data Preparation for LSTMs

- [How to Reshape Input Data for Long Short-Term Memory Networks](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)
- [How to One Hot Encode Sequence Data](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)
- [How to Remove Trends and Seasonality with a Difference Transform](https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/)
- [How to Scale Data for Long Short-Term Memory Networks](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)
- [How to Prepare Sequence Prediction for Truncated BPTT](https://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)
- [How to Handle Missing Timesteps in Sequence Prediction Problems](https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/)

#### LSTM Behaviour

- [A Gentle Introduction to Backpropagation Through Time](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
- [Demonstration of Memory with a Long Short-Term Memory Network](https://machinelearningmastery.com/memory-in-a-long-short-term-memory-network/)
- [How to Use the TimeDistributed Layer for Long Short-Term Memory Networks](https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/)
- [How to use an Encoder-Decoder LSTM to Echo Sequences of Random Integers](https://machinelearningmastery.com/how-to-use-an-encoder-decoder-lstm-to-echo-sequences-of-random-integers/)
- [Attention in Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)

#### Modeling with LSTMs

- [Generative Long Short-Term Memory Networks](https://machinelearningmastery.com/gentle-introduction-generative-long-short-term-memory-networks/)
- [Stacked Long Short-Term Memory Networks](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)
- [Encoder-Decoder Long Short-Term Memory Networks](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
- [CNN Long Short-Term Memory Networks](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)
- [Diagnose Overfitting and Underfitting of LSTM Models](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/)
- [How to Make Predictions with Long Short-Term Memory Models](https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/)

#### LSTM for Time Series

- [On the Suitability of LSTMs for Time Series Forecasting](https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/)
- [Time Series Forecasting with the Long Short-Term Memory Network](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
- [Multi-step Time Series Forecasting with Long Short-Term Memory Networks](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/)
- [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)





# Crash Course in Recurrent Neural Networks for Deep Learning

There is another type of neural network that is dominating difficult machine learning problems that involve **sequences of inputs** called recurrent neural networks.

- Recurrent neural networks have connections that have loops, adding feedback and memory to the networks over time. This memory allows this type of network to learn and generalize across sequences of inputs rather than individual patterns.
- A powerful type of Recurrent Neural Network called the Long Short-Term Memory Network has been shown to be particularly effective **when stacked into a deep configuration**, achieving state-of-the-art results on a diverse array of problems from language translation to automatic captioning of images and videos.

In this post you will get a crash course in recurrent neural networks for deep learning, acquiring just enough understanding to start using LSTM networks in Python with Keras.

After reading this post, you will know:

- The limitations of Multilayer Perceptrons that are addressed by recurrent neural networks.
- The problems that must be addressed to make Recurrent Neural networks useful.
- The details of the Long Short-Term Memory networks used in applied deep learning.

### Support For Sequences in Neural Networks

There are some problem types that are best framed involving either a sequence as an input or an output.

- For example, consider a univariate time series problem, like the price of a stock over time. This dataset can be framed as a prediction problem for a classical feedforward multilayer Perceptron network by defining a windows size (e.g. 5) and training the network to learn to make short term predictions from the fixed sized window of inputs.

This would work, but is very limited:

- The window of inputs adds memory to the problem, but is limited to just a fixed number of points and must be chosen with sufficient knowledge of the problem. 
- A naive window would not capture the broader trends over minutes, hours and days that might be relevant to making a prediction. 
- From one prediction to the next, the network only knows about the specific inputs it is provided.



Consider the following taxonomy (分类) of sequence problems that require a mapping of an input to an output (taken from Andrej Karpathy).

- **One-to-Many**: sequence output, for image captioning.
- **Many-to-One**: sequence input, for sentiment classification.
- **Many-to-Many**: sequence in and out, for machine translation.
- **Synched Many-to-Many**: synced sequences in and out, for video classification.

*Remark:*

- *Sentiment classification: is a special task of text classification whose objective is to classify a text according to the sentimental polarities of opinions it contains (Pang et al., 2002), e.g., favorable or unfavorable, positive or negative.*
- *We can also see that a one-to-one example of input to output would be an example of a classical feed forward neural network for a prediction task like image classification.*

## Recurrent Neural Networks

Recurrent Neural Networks or RNNs are a special type of neural network designed for sequence problems.

<img src= http://images2015.cnblogs.com/blog/947235/201608/947235-20160821234331464-1137952568.png />

Given a standard feed-forward multilayer Perceptron network, a recurrent neural network can be thought of as the addition of loops to the architecture. 

- For example, in a given layer, each neuron may pass its signal latterly (sideways) in addition to forward to the next layer. 
- The output of the network may feedback as an input to the network with the next input vector. And so on.

The recurrent connections add state or memory to the network and allow it to learn broader abstractions from the input sequences.

### 1. How to Train Recurrent Neural Networks

The staple technique for training feed forward neural networks is to back propagate error and update the network weights.

- [Backpropagation](https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/) breaks down in a recurrent neural network, because of the recurrent or loop connections. This was addressed with a modification of the Backpropagation technique called [Backpropagation Through Time](https://en.wikipedia.org/wiki/Backpropagation_through_time) or BPTT.
- Instead of performing backpropagation on the recurrent network as stated, the structure of the network is unrolled, where copies of the neurons that have recurrent connections are created. For example a single neuron with a connection to itself (A->A) could be represented as two neurons with the same weight values (A->B).
- This allows the cyclic graph of a recurrent neural network to be turned into an acyclic graph (非循环图) like a classic feed-forward neural network, and Backpropagation can be applied.

### 2. How to Have Stable Gradients During Training

When Backpropagation is used in very deep neural networks and in unrolled recurrent neural networks, the gradients that are calculated in order to update the weights can become unstable.

They can become very large numbers called exploding gradients or very small numbers called the [vanishing gradient problem](https://machinelearningmastery.com/how-to-fix-vanishing-gradients-using-the-rectified-linear-activation-function/). These large numbers in turn are used to update the weights in the network, making training unstable and the network unreliable.

This problem is alleviated (缓解) in deep multilayer Perceptron networks through the use of the Rectifier transfer function, and even more exotic but now less popular approaches of using unsupervised pre-training of layers.

In recurrent neural network architectures, this problem has been alleviated using a new type of architecture called the **Long Short-Term Memory** Networks that allows deep recurrent networks to be trained.

## Long Short-Term Memory Networks

- The Long Short-Term Memory or LSTM network is a recurrent neural network that is trained using Backpropagation Through Time and **overcomes the vanishing gradient problem**.

- As such it can be used to create large (stacked) recurrent networks, that in turn can be used to address difficult sequence problems in machine learning and achieve state-of-the-art results.

<img src=http://reset.pub/pic/ml/rnn/lstm.jpg />

- Instead of neurons, LSTM networks have **memory blocks** that are connected into layers. A block has components that make it smarter than a classical neuron and a memory for recent sequences.  A block contains gates that manage the block’s state and output. 
- There are three types of gates within a memory unit:
  - **Forget Gate**: conditionally decides what information to discard from the unit.
  - **Input Gate**: conditionally decides which values from the input to update the memory state.
  - **Output Gate**: conditionally decides what to output based on input and the memory of the unit.

## Resources

We have covered a lot of ground in this post. Below are some resources that you can use to go deeper into the topic of recurrent neural networks for deep learning.

Resources to learn more about Recurrent Neural Networks and LSTMs.

- [Recurrent neural network on Wikipedia](https://en.wikipedia.org/wiki/Recurrent_neural_network)
- [Long Short-Term Memory on Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by Andrej Karpathy
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Deep Dive into Recurrent Neural Nets](http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/)
- [A Beginner’s Guide to Recurrent Networks and LSTMs](http://deeplearning4j.org/lstm.html)

Popular tutorials for implementing LSTMs.

- [LSTMs for language modeling with TensorFlow](https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html)
- [RNN for Spoken Word Understanding in Theano](http://deeplearning.net/tutorial/rnnslu.html)
- [LSTM for sentiment analysis in Theano](http://deeplearning.net/tutorial/lstm.html)

Primary sources on LSTMs.

- [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) [pdf], 1997 paper by Hochreiter and Schmidhuber
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/abs/10.1162/089976600300015015), 2000 by Schmidhuber and Cummins that add the forget gate
- [On the difficulty of training Recurrent Neural Networks](http://arxiv.org/pdf/1211.5063v2.pdf) [pdf], 2013

People to follow doing great work with LSTMs.

- [Alex Graves](http://www.cs.toronto.edu/~graves/)
- [Jürgen Schmidhuber](http://people.idsia.ch/~juergen/)
- [Ilya Sutskever](http://www.cs.toronto.edu/~ilya/)
- [Tomas Mikolov](http://www.rnnlm.org/)



# Demonstration of Memory with a Long Short-Term Memory Network in Python

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network capable of learning over long sequences. This differentiates them from regular multilayer neural networks that do not have memory and can only learn a mapping between input and output patterns.

After completing this tutorial, you will know:

- How to define a small sequence prediction problem **that only an RNN like LSTMs can solve** using memory.
- How to transform the problem representation so that it is suitable for learning by LSTMs.
- How to design an LSTM to solve the problem correctly.



## Sequence Problem Description

The problem is to predict values of a sequence one at a time. The two sequences to be learned are as follows:

- 3, 0, 1, 2, 3
- 4, 0, 1, 2, 4

- We can see that the first value of the sequence is repeated as the last value of the sequence. This is the indicator that provides context to the model as to which sequence it is working on.
- The conflict is the transition from the second to last items in each sequence. In sequence one, a “2” is given as an input and a “3” must be predicted, whereas in sequence two, a “2” is given as input and a “4” must be predicted.

***This is a problem that a multilayer Perceptron and other non-recurrent neural networks cannot learn.***

- A wrinkle is that there is conflicting information between the two sequences and that the model must know the context of each one-step prediction (e.g. the sequence it is currently predicting) in order to correctly predict each full sequence.

- This wrinkle is important to prevent the model from memorizing each single-step input-output pair of values in each sequence, as a sequence unaware model may be inclined to do.
- This is a simplified version of “*Experiment 2*” used to demonstrate LSTM long-term memory capabilities in Hochreiter and Schmidhuber’s 1997 paper [Long Short Term Memory](http://dl.acm.org/citation.cfm?id=1246450) ([PDF](http://www.bioinf.jku.at/publications/older/2604.pdf)).

## Extensions

This section lists ideas for extensions to the examples in this tutorial.

- **Tuning**. The configurations for the LSTM (epochs, units, etc.) were chosen after some trial and error. It is possible that a much simpler configuration can achieve the same result on this problem. Some search of parameters is required.
- **Arbitrary Alphabets**. The alphabet of 5 integers was chosen arbitrarily. This could be changed to other symbols and larger alphabets.
- **Long Sequences**. The sequences used in this example were very short. The LSTM is able to demonstrate the same capability on much longer sequences of 100s and 1000s of time steps.
- **Random Sequences**. The sequences used in this tutorial were linearly increasing. New sequences of random values can be created, allowing the LSTM to devise a generalized solution rather than one specialized to the two sequences used in this tutorial.
- **Batch Learning**. Updates were made to the LSTM after each time step. Explore using batch updates to see if this improves learning or not.
- **Shuffle Epoch**. The sequences were shown in the same order each epoch during training and again during evaluation. Randomize the order of the sequences so that sequence 1 and 2 are fit within an epoch, which might improve the generalization of the model to new unseen sequences with the same alphabet.

Did you explore any of these extensions?
Share your results in the comments below. I’d love to see what you came up with.

## Further Reading

I strongly recommend reading the original 1997 LSTM paper by Hochreiter and Schmidhuber; it is very good.

- [Long Short Term Memory](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735), 1997 [[PDF](http://www.bioinf.jku.at/publications/older/2604.pdf)]

