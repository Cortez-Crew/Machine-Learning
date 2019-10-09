"""
My notes for TensorFlow
https://github.com/easy-tensorflow/easy-tensorflow/
↖ ↗ ↘ ↙

"""

import tensorflow.compat.v1 as tf

"""
    Now that we've covered the basics of how tensorflow works, we can start trying to construct a simple
    neural network. To do this, we'll use one hidden layer and 200 hidden units (neurons).
    
    This will look like:
    h = ReLU(Wx + b)
    
        [ ReLU ]
               ↖
             [ Add ]
            ↗       ↖
         { b }   [ MatMul ]
                  ↗       ↖
                { W }   { x }
                
    Input x is going to be any kind of input data. This can be anything from pictures, numbers, audio, etc.
    The idea is that we will feed the network inputs and train the trainable parameters (W, b) by backpropagating
    the error signal. Ideally, you will feed a network all your input data, then find the error signal, and 
    go backwards to update the parameters. This is known as 'Gradient Decent'
    
    **  In full scale neural networks, since there are going to be thousands to millions of inputs, gradient decent
        is incredibly time consuming for a computer to figure out in one run. Therefore, the full input data is split
        into smaller pieces (mini-batches) of size B (mini-batch size). Then the mini-batches are fed one by one.
        This is called 'Stochastic Gradient Decent'. Each process of feeding in a mini-batch, computing the error 
        signal, and updating the parameters (Weights, biases) is an 'Iteration'.
"""


def neural_network_basics():
    """"""
    """"
        The first step to making our network is to make a placeholder for our input. The important part is choosing
        the right input data size.

        Here, we'll use a size of 784. This is useful for using MNIST 28x28 images.
        The shape is (None, 784) because we want the network to be able to accept any number of inputs, so this is
        used like a wildcard.
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='X')

    """
        In addition to the input, we need to set up our weights and biases. These will be variables.
        Generally, Weights (W) are initialized randomly from a normal distribution. Here we'll say with a mean of 0,
        and a standard deviation of 0.01. Biases however, can be small constant values, for this we'll just use 0.
    """