# Animated RNN, LSTM and GRU

## Recurrent neural network cells in GIFs

Recurrent neural networks (RNNs) are a class of artificial neural networks which are often used with sequential data. The 3 most common types of recurrent neural networks are

1. vanilla RNN,
2. long short-term memory (LSTM), proposed by [Hochreiter and Schmidhuber in 1997](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory?source=post_page---------------------------), and
3. gated recurrent units (GRU), proposed by [Cho *et. al* in 2014](https://arxiv.org/abs/1409.1259?source=post_page---------------------------).

> Note that I will use “RNNs” to collectively refer to neural network architectures that are inherently recurrent, and “vanilla RNN” to refer to the simplest recurrent neural network architecture as shown in Fig. 1.

There are many illustrated diagrams for recurrent neural networks out there. My personal favorite is the one by Michael Nguyen in [this article](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21?source=post_page---------------------------) published in Towards Data Science, because he provides us with intuition on these models and more importantly the beautiful illustrations that make it easy for us to understand. But the motivation behind my post is to have a better visualization what happens in these cells, and how the nodes are being shared and how they transform to give the output nodes. I was also inspired by the Michael’s nice animations.

This article looks into vanilla RNN, LSTM and GRU cells. It is a short read and is for those who have read up on these topics. (I recommend reading Michael’s article before reading this post.) It is important to note that the following animations are sequential to guide the human eyes, but do not reflect the chronological order during vectorised machine computation.

Here is the legend that I have used for the illustrations.

<img src=imgs/a_0.png >

​																	Fig. 0: Legend for animations

In my animations, I have used an input size of **3 (green) and 2 hidden units (red) with a batch size of 1**.

Let’s begin!

------

## Vanilla RNN

<img src=imgs/a_1.gif >

​                                                                  Fig. 1: Animated vanilla RNN cell

- *t* — time step
- *X —* input
- *h —* hidden state
- length of *X —* size/dimension of input
- length of *h —* no. of hidden units. Note that different libraries call them differently, but they mean the same:
  \- Keras — `state_size` *,*`units`- PyTorch — `hidden_size` 
  \- TensorFlow — `num_units`

## LSTM

<img src=imgs/a_2.gif >

​                                                                      Fig. 2: Animated LSTM cell

- *C —* cell state

Note that the dimension of the cell state is the same as that of the hidden state.

## GRU

<img src=imgs/a_3.gif >

​                                                                                Fig. 3: Animated GRU cell

------

Hope these animations helped you in one way or another! Here is a summary of the cells in static images:

<img src=imgs/a_4.png >

​                                                                                      Fig. 4: Vanilla RNN cell



<img src=imgs/a_5.png >

​																							Fig. 5: LSTM cell



<img src=imgs/a_6.png >

​																								Fig. 6: GRU cell

## Notes

I used Google Drawing to create these graphics.

## References