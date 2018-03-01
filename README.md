# neuralNet
This repository contains an implementation of a self-written neural network which can present (so-far) fully connected networks.
It is based on an object-oriented approach using a Neuron as a basic unit. In this spirit, a layer is a neuron of neurons. Such structure allows to compute forward and backward passes on either the neuron or the layer level. Here it depends on the size of the weight matrix, on the sample size and on the batch size which of both approaches will result in the best performance. 

__neuralNet__ includes:
* neuralNet
  * Neuron.py          --- Definition of the neuron object
  * Layer.py           --- Definition of the layer object inheriting from Neuron 
  * Net.py             --- Wrapper for a network = list of layers
  * Costfunctions.py   --- Abstract Class for costfuntions with some exemplay implementations
* data
  * Samples.py         --- Class for handling Samples
  * DisplayData.py     --- Class to plot image data
* main.py              --- Exemplary Implementation of a Network training on the MNIST DATABASE


# Using the neuralNet for the MNIST database
A a proof of principle, a example network with two layers (25 hidden, 10 output) is implemented to predict the handwritten digits
of the MNIST database. TANH is used as an activation function. Without hyperparameter fine tuning, a Train classification accuracy of 98.5% is achieved after 3000 epochs.

![GitHub Logo](/images/logo.png)
Format: ![Alt Text](url)
