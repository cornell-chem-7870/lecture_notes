---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: comp-prob-solv
  language: python
  name: python3
---

# A Crash Course on Neural Networks


## Learning Objectives

- Develop a working knowledge of three types of neural networks: feedforward neural networks (FNNs), convolutional neural networks (CNNs), and transformers.
    - Understand the role that symmetry plays in the design of CNNs.
    - Understand the attention mechanism in transformers.
- Understand the role of minibatching in training neural networks
- Develop best practices for training neural networks, including how to choose the learning rate, how to initialize the weights and biases, and how to prevent overfitting.

## Neural Network Architectures

Below, we discuss three of the most commonly used types of neural network architectures: feedforward neural networks (FNNs), convolutional neural networks (CNNs), and transformers.  Each of these architectures is designed to take advantage of different types of structure in the data, and each has its own strengths and weaknesses.

### Feedforward Neural Networks (FNNs)

Feed forward neural networks (FNNs) are some of the first, and simplest, types of neural networks.  Mathematically, a feedforward neural network is a composition of ''layers'' of the form 

$$
f(x) = \sigma(Wx + b)
$$

where $W$ is a matrix of weights, $b$ is a vector of biases, and $\sigma$ is a non-linear activation function.  The parameters $W$ and $b$ are the weights and biases of the network respectively.  Common choices for activation functions include the sigmoid function, the hyperbolic tangent function, and the rectified linear unit (ReLU) function, plotted below:

```{python}
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].plot(x, sigmoid)
axes[0].set_title('Sigmoid')
axes[1].plot(x, tanh)
axes[1].set_title('Tanh')
axes[2].plot(x, relu)
axes[2].set_title('ReLU')
plt.show()
```

Currently, the ReLU function is the most commonly used activation function in deep learning, due to its simplicity and effectiveness in training deep networks.  

When we compose multiple layers of the form above, we get a feedforward neural network.  For example, a feedforward neural network with two hidden layers can be written as

$$
f(x) = W_4 \sigma_3(W_3 \sigma_2(W_2 \sigma_1(W_1 x + b_1) + b_2) + b_3) + b_4
$$

In the hidden layers, each layer takes the output of the previous layer.  The number of rows in each matrix is the number of ``neurons'' in that layer.  This must match the number of columns in the next weight matrix $W_{i+1}$.  Precisely how big these weight matrices are is a design choice, and is one of the hyperparameters of the network.  However, it as also one that doesn't matter *too* much as long as it is sufficiently large.  In practice, 32 x 32 or 64 x 64 are good choices for starting values, and you can scale up from there if you have the computational resources and desire more performance.  The number of hidden layers is also a design choice, and is another hyperparameter of the network.  In practice, 2-4 hidden layers are often sufficient for many tasks.  

Care must be designing the last layer of the network, as it must be designed to output the correct type of output for the task at hand.  For example, if we are doing binary classification, we might want the last layer to output a single value between 0 and 1, which can be interpreted as a probability.  In this case, we might use a sigmoid activation function in the last layer.  
If we are doing regression, we might want the last layer to output a single value from $(-\infty, \infty)$, in which case we might not use any activation function in the last layer.
If we are doing multi-class classification, we might want the last layer to output a vector of probabilities for each class, in which case we might use a softmax activation function in the last layer.  The softmax function is defined as

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

This generalizes the sigmoid function to multiple classes (as an exercise, you can verify that the softmax function reduces to the sigmoid function when there are only two classes).  The softmax function takes a vector of real numbers and outputs a vector of probabilities that sum to 1.


### Convolutional Neural Networks (CNNs)

Feed forward neural networks are very general, and can be used for a wide variety of tasks.  However, their limits quickly became apparent when they were applied to tasks like image recognition.  Consider applying a feedforward neural network to a 128x128 pixel image.  The input layer would have 128x128 = 16,384 neurons, and even if the first hidden layer had 64 neurons, then the weight matrix $W_1$ would have 16,384 x 64 = 1,048,576 parameters -- already a very large number of parameters.  
Moreover, feed-forward networks do not leverage the spatial structure of images.  If we shift an image to the left, the feedforward neural network would have to learn to recognize the same object in a different position, which is inefficient and requires a lot of data.  Convolutional neural networks (CNNs) were designed to address these issues by leveraging the spatial structure of images and reducing the number of parameters.

The key behind a convolutional network is to structure the linear operation in each layer by constraining it to act as a convolution.  For simplicity, consider a one-dimensional signal, with values $x_1, x_2, \ldots, x_n$.  A convolutional layer convolves a kernel (or filter) of weights $w_1, w_2, \ldots, w_k$ with the input signal to produce an output signal.  The convolution operation can be written as

$$
y_i = \sum_{j=1}^{k} w_j x_{i+j-1} + b
$$

(Technically this is a cross-correlation operation.)
This operation is repeated across the entire input signal, producing an output signal of length $n-k+1$.  The kernel weights $w_j$ are shared across the entire input signal.
In the Figure below, we give an illustration for a two-dimensional image, where the kernel is a 3x3 matrix of weights that is convolved with the input image to produce an output image.

![Convolutional Neural Network](images/cnn_example.png)

Convolutional neural networks substantially reduce the number of parameters compared to feedforward neural networks.  Moreover, they naturally respect translation symmetry in the signal: if we shift the input signal the output will shift in the same way.  

### Equivariance

Formally, respecting symmetries is known as *equivariance* to translation.  This is a group-theoretic property, and can be expressed succinctly using the following commutative diagram:

$$
\begin{array}{ccc}
\text{Input} & \xrightarrow{f} & \text{Output} \\
\downarrow{T} & & \downarrow{T'} \\
\text{Input} & \xrightarrow{f} & \text{Output}
\end{array}
$$

where $T$ and $T'$ are translation operators on the input and output spaces respectively.  The diagram says that if we first apply the function $f$ to the input, and then translate the output, we get the same result as if we first translate the input, and then apply the function $f$.  In other words, the function $f$ commutes with the action of the translation group. 


## Transformers

TBD....

### The Attention Mechanism


TBD....
