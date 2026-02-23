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

# Nonlinear Regression

## Learning Objectives
- Discuss the general framework of Nonlinear Regression.
- Discuss Logistic regression and its loss functional.
- Give a brief overview of neural networks and their loss functionals.
- Discuss density estimation.


## Nonlinear Regression

In the beginning of this class, we repeatedly discussed linear regression: the act of determining a linear relation between two variables.
Here, we extend this to nonlinear regression.

In general, nonlinear regression has the following form.
We have some parameterized function $f(x, \theta)$, where $\theta$ is a vector of parameters.
Additionally, we have a functional $L$ that measures the quality for $f$, which we call the *loss function*.
Our goal is to find the parameters $\theta$ that minimize the loss functional,

$$
\theta^* = \arg \min_\theta L(f(x, \theta), y)
$$

where $y$ is the data we are trying to fit.
If $f$ is differentiable with respect to $\theta$, we can use gradient descent or Newton's method to find the minimum.

```{note}
In the case of linear regression, we had $f(x, \theta) = Ax + b$, where $A$ is a matrix and $b$ is a vector.
The loss functional was the mean square $L(f(x, \theta), y) = ||Ax + b - y||^2$.
```

While accurate, the above definition is a bit abstract.  In this lecture, we discuss some more concrete examples.

## Example: Logistic Regression

Logistic regression is a common method for binary classification.
Here, each $x$ is associated with one of two outcomes which we denote $0$ and $1$.
Our goal is to fit a function that predicts the probability of $x$ being associated with outcome $1$.
To be able to use gradient descent, rather than directly predicting 0's or 1's we us a function that goes smoothly between the two: a sigmoid function.
The sigmoid function is given by:

$$
    f(x, \theta) = \frac{1}{1 + e^{-\theta^T x}}
$$

where $\theta$ is a vector of parameters.
For our loss functional, one option would have been to use the mean square error.  However, this can lead to some practical issues.  The biggest is that gradients can be very small.
To demonstrate this, let's consider an example in one dimension.
We have a point at $x = 1$, and the true label is $y = 0$.  However, our current parameter is set to $\theta=10$.
Consequently, our prediction is set to

$$
 \frac{1}{1 + e^{-10}}  \approx 0.9999546
$$

For this point, our mean square error is then given by $(0.9999546 - 0)^2 \approx 0.9999092$.
 To fix this, let's halve  our parameter, setting it to $\theta=5$.
Now, our prediction is given by

$$
 \frac{1}{1 + e^{-5}}  \approx 0.9933071
$$

Our mean square error is now $(0.9933071 - 0)^2 \approx 0.9867$.  Despite the fact the we have halved our parameter, our mean square error has barely decreased!

Consequently, it is common to use a different loss functional, called the *cross-entropy loss*.
The cross-entropy loss is given by:

$$
L(f(x, \theta), y) = -y \log(f(x, \theta)) - (1-y) \log(1-f(x, \theta))
$$

Note that since $y$ is either $0$ or $1$, this is equivalent to:

$$
L(f(x, \theta), y) = \begin{cases}
    -\log(f(x, \theta)) & \text{if } y = 1 \\
    -\log(1-f(x, \theta)) & \text{if } y = 0
\end{cases}
$$

This loss function has much better properties than the mean square error.  The gradients when we are far from the true solution are much larger.  Moreover, it is convex, which helps with higher-order optimization methods.

## Extension to Neural Networks

In general, linear regression is a good go-to method for predicting continuous values, logistic regression is a good go-to method for predicting binary values.
However, sometimes we might want a more complex function than a linear function. 
In this case we can employ a *neural network*.
At their most basic, neural networks are a class of functions that are composed of many layers of linear functions, each followed by a nonlinear function.
Denoting the nonlinear function as $\sigma$, we can write the neural network as 

$$
NN(x, \theta) =  \theta_n \sigma (\theta_{n-1} \sigma( \cdots \sigma( \theta_2 \sigma( \theta_1 x))))
$$

where we have divided our parameters into $n$ matrices, each denoted $\theta_i$.
Different neural networks differ on the number of layers, if and how we constrain the parameters, and how we choose the nonlinear function $\sigma$.
Common choices for $\sigma$ include the sigmoid function discussed above, 
the hyperbolic tangent function 

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}},
$$

and (perhaps the most popular option) the rectified linear unit (ReLU) function 

$$
\text{ReLU}(x) = \max(0, x)
$$

Once we have chosen the form for our neural network, we tune it to predict values.
For predicting continuous values, we use a linear loss functional 

$$
L(f(x, \theta), y) = ||f(x, \theta) - y||^2
$$

where $||\cdot||$ is the Euclidean norm.  For predicting binary values, we use the cross-entropy loss functional

$$
L(f(x, \theta), y) = -y \log(f(x, \theta)) - (1-y) \log(1-f(x, \theta))
$$

where we set

$$
f(x, \theta) = \frac{1}{1 + e^{-NN(x, \theta)}}s
$$

with $NN(x, \theta)$ being the neural network we defined above.


## Example: Density Estimation

Given a set of samples $x_i$ from some probability distribution drawn from an unknown probability density $\rho$, we might wish to recover an approximation of the underlying probability density.
This is known as *density estimation*.
One approach to solving this problem is to parameterize a set of possible probability densities $f(x, \theta)$, and write down a loss functional that measures the difference between the true probability density and our approximation.
As it happens, we can generalize the cross-entropy loss functional to do this.

For general probability densities $p(x)$ and $q(x)$, the cross entropy is given by:

$$
H(p, q) = -\int p(x) \log(q(x)) dx
$$

This is a measure of how well $q$ approximates $p$: as $q$ gets closer to $p$, the cross-entropy goes down.
We can use this to define a loss functional for density estimation by replacing $p$ with the true probability density $\rho$ and $q$ with our approximation $f(x, \theta)$.
Substituting this into the cross-entropy, we get:

$$
 -\int \rho(x) \log(f(x, \theta)) dx
$$

We can approximate this using an average over our samples to get a loss funcitonal that we can minimize:

$$
L(f(x, \theta), y) = -\frac{1}{N} \sum_{i=1}^N \log(f(x_i, \theta))
$$

where $N$ is the number of samples.
Our remaining task is to parameterize $f$.  One common approach is to use a *Gaussian mixture model* (GMM).
A GMM is a weighted sum of Gaussians, given by:

$$
f(x, \theta) = \sum_{i=1}^K w_i \mathcal{N}(x | \mu_i, \Sigma_i)
$$

where $w_i$ is the weight of the $i$th Gaussian, $\mu_i$ is the mean of the $i$th Gaussian, and $\Sigma_i$ is the covariance of the $i$th Gaussian.
The weights $w_i$ are constrained to be non-negative and sum to 1, while the means $\mu_i$ and covariances $\Sigma_i$ can be any real-valued matrix.

```{note}
Depending on the dimension of the data, gradient-based optimizers may not be the best choice for this problem.
If you are interested in pursuing this direction for your research, it may be worth looking into the *Expectation-Maximization* (EM) algorithm.
```

## Example: Learning Force Fields

Force fields are physical models that predict the potential energy of a system based on its configuration.
This potential energy is, in principle, computable from first principles using quantum mechanics.
However, this is often far to computationally expensive to be practical.  Instead, we approximate this energy by a sum of particle-particle interactions.  For instance, electrostatic interactions are given by the Coulomb potential:

$$
V(r) = \frac{q_1 q_2}{r}
$$

where $q_1$ and $q_2$ are the (effective) charges of the two particles, and $r$ is the distance between them.
Interatomic repulsion and van der Waals interactions are given by the Lennard-Jones potential:

$$
V(r) = 4 \epsilon \left( \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right)
$$

where $\epsilon$ is the depth of the potential well, and $\sigma$ is the distance at which the potential is zero.
We might also have additional terms that depend on the angle between three particles, or the dihedral angle between four particles.
However, in all of these cases we need to tune parameters of these functions to approximate the quantum mechanical energies as well as possible.

This is a linear regression problem: we have a set of quantum energies $y$, a complicated function $f$ that approximates them given a structure $x$, and we want to find the parameters that minimize the difference between the two.
Again, we can calculate the mean-squared in the predicted and the true energies.  But this often isn't enough to get good force fields.
The reason for this is that even if a function converges, that doesn't mean its derivative converges.
As a classic example, consider the function $1 / k \sin(k x)$.  As $k$ goes to infinity, this function converges to $0$ for all $x$.  However, its derivative diverges.
This is a problem for force fields, because the forces are given by the derivative of the potential energy with respect to the coordinates.
Consequently, we need to ensure that the forces converge as well.

This is the basis for the *force matching* method.
In this method, we minimize the difference between the forces predicted by our model and the forces predicted by quantum mechanics, in addition to the difference between the energies.
The loss functional is given by:

$$
L(f(x, \theta), y) = ||f(x, \theta) - y||^2 + ||\sum_i (\frac{\partial}{\partial x_i} f(x, \theta) - f_{iy})||^2
$$

where $f_{iy}$ is the force predicted by quantum mechanics in coordinate $i$ for the $y$th sampled configuration.
