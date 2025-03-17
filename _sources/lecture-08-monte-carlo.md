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

# Monte Carlo

## Learning Objectives

- Understand the basic mathematical principles behind Monte Carlo methods.
- Be able to implement simple Monte Carlo Algorithms
- Understand the convergence rate of Monte Carlo methods and what governs the error.

## The Problem: Evaluating High-dimensional Integrals

In Bayesian inference, we might want to calculate the posterior mean, variance, or some other statistic.
To recap, the posterior distribution is given by Bayes' theorem,

$$
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}
$$

where $D$ is the data, $\theta$ is the parameter, $p(D|\theta)$ is the likelihood, $p(\theta)$ is the prior, and $p(D)$ is the evidence.
The posterior mean is given by

$$
E[\theta|D] = \int \theta p(\theta|D)d\theta
$$

and the posterior variance is given by

$$
\text{Var}[\theta|D] = \int (\theta - E[\theta|D])^2 p(\theta|D)d\theta
$$

So far, we have calculated these expectations against the posterior distribution either analytically or 
using numerical methods such as grid approximation or numerical integration.
Unfortunately, these methods do not scale well to high-dimensional problems.
Calculating the posterior requires evaluating the evidence,

$$
p(D) = \int p(D|\theta)p(\theta)d\theta
$$

If we have a single parameter, this is trivial.  If we have three, we have to evaluate the integral over a three-dimensional space: not impossible, but potentially expensive.
If we have 100 parameters, we have to evaluate the integral over a 100-dimensional space: this is impossible using numerical integration.

We have a similar problem in statistical mechanics.
Most experimental measurements are averages over a large number of particles.
At thermal equilibrium, these averages are given by the Boltzmann distribution and are given by

$$
  E[f] = \frac{\int f(x) e^{-\beta H(x)}dx}{\int e^{-\beta H(x')}dx'}
$$

where $f(x)$ is the function we want to calculate, $H(x)$ is the Hamiltonian, and $\beta = 1/kT$.
Each point $x$ is a point in the phase space of the system, which is a 6N-dimensional space for a system with N particles.
Again, this is impossible to calculate using numerical integration for even modest values of N.
Clearly, we need another way to calculate these integrals.

## Monte Carlo Methods

So far, we have been considering a random variable and then calculating its expectation by integrating over a probability density.
Monte Carlo methods flip this paradigm.  Given a high-dimensional integral, we (1) find a way to write it as an expectation over a probability density, and then (2) estimate the expectation using random samples drawn from the corresponding distribution.
If we choose our random samples well, the law of large numbers tells us that the sample mean will converge to the expected value as the number of samples goes to infinity.

### Convergence of Monte Carlo

As a specific example, let's assume we are estimating the expectation of a function $f(x)$ over a probability density $p(x)$.
For simplicity, let's assumed that the true expectation of $f(x)$ is zero and its variance is $\sigma^2$.

$$
\begin{split}
E[f] &= \int f(x) p(x) dx  = 0 \\
\text{Var}[f] &= \int f(x)^2 p(x) dx = \sigma^2
\end{split}
$$

We are going to approximate this expectation using a sample mean

$$
\hat{E}[f] = \frac{1}{N}\sum_{i=1}^N f(x_i)
$$

where $x_i$ are samples drawn independently and identically from the distribution associated with the density $p(x)$.  To evaluate 
the error in this estimate, we will calculate the variance of the sample mean.

We first note that since our samples are drawn independently and identically, they have a probability density

$$
p(x_1, x_2, \ldots, x_N) = p(x_1)p(x_2)\ldots p(x_N)
$$

We immediately see that the expectation of the sample mean is the true expectation

$$
E[\hat{E}[f]] = E[ \frac{1}{N}\sum_{i=1}^N f(x_i)] = \frac{1}{N}\sum_{i=1}^N \int f(x_i) p(x_1) p(x_2) \ldots p(x_N) dx_i = \int f(x) p(x) dx = 0
$$

Consequently, the variance of the sample estimator is just given by the expectation of its square.

$$
E[\hat{E}[f]^2] = E[\frac{1}{N^2}\sum_{i=1}^N \sum_{j=1}^N f(x_i)f(x_j)] = \frac{1}{N^2}\sum_{i=1}^N \sum_{j=1}^N \int f(x_i)f(x_j) p(x_1) p(x_2) \ldots p(x_N) dx_i dx_j
$$

Since the samples are drawn independently, these expectations are zero unless $i = j$.  Consequently, the variance of the sample mean is given by

$$
\text{Var}[\hat{E}[f]] = \frac{1}{N^2}\sum_{i=1}^N \int f(x_i)^2 p(x_i) dx_i = \frac{\sigma^2}{N}
$$

This tells us that the variance of the sample mean decreases as $1/N$.  Similarly, we expect the root mean square error to decrease as $1/\sqrt{N}$.


### Monte Carlo Integration: Calculating Pi

As a simple demonstration of Monte Carlo integration, consider the problem of calculating $\pi$.
The area of a circle is given by $\pi r^2$.
The area of a square that circumscribes the circle, in turn, is given by $4r^2$.
Consequently, if we can calculate the fraction of the area of the square that is covered by the circle, we can calculate $\pi$: this fraction is given by $\pi/4$.
This fraction can be written as an expectation over a probability density:

$$
\frac{\pi}{4} = \int_{-1}^1 \int_{-1}^1 \mathbb{1}(x^2 + y^2 \leq 1) p(x, y) dx dy
$$

where $\mathbb{1}(x^2 + y^2 \leq 1)$ is the indicator function that is 1 if $x^2 + y^2 \leq 1$ and 0 otherwise,
and $p(x, y)$ is the uniform distribution over the square.

To estimate this integral, we can draw $N$ samples from the uniform distribution over the square and calculate the fraction of samples that fall within the circle.
The fraction of samples that fall within the circle will converge to $\pi/4$ as $N$ goes to infinity.  In code, this looks as follows:

```{python}
import numpy as np

def in_circle(x, y):
    return x**2 + y**2 <= 1

def estimate_pi(N):
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    return 4 * np.mean(in_circle(x, y))

estimate_pi(1000)
```

### Monte Carlo Integration: Calculating Posterior Means by Sampling the Prior

Lets say we are attempting to calculate the posterior mean of a parameter $\theta$ and the prior is a multi-dimensional Gaussian distribution.
We can approximate the posterior mean as follows:

1. Draw $N$ samples from the prior distribution.
2. Calculate the likelihood for each sample.
3. Calculate the approximations

$$
\begin{split}
p(D) = \int p(D|\theta)p(\theta)d\theta \approx \frac{1}{N}\sum_{i=1}^N p(D|\theta_i) \\
\int \theta p(\theta|D)d\theta \approx \frac{1}{N}\sum_{i=1}^N \theta_i p(\theta_i|D)
\end{split}
$$

4. Calculate the posterior mean by dividing the two approximations.

This gives a Monte Carlo estimate of the posterior mean.  
