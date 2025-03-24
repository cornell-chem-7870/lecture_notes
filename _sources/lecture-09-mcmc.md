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

# Markov Chain Monte Carlo

## Learning Objectives

- Understand the basic principles behind Markov Chain Monte Carlo (MCMC) methods.
- Understand how Gibbs sampling works and how to apply it to the Ising model.

## Markov Chain Monte Carlo

In the previous examples of Monte Carlo, we sampled from simple distributions such as the uniform distribution or the normal distribution.   But what if we have a complicated distribution?  For instance, in statistical mechanics we might wish to sample from the Boltzmann distribution, which is given by

$$
\pi(x) = \frac{1}{Z} e^{-\beta H(x)}
$$

where $Z$ is the partition function and $H(x)$ is the Hamiltonian.  In general, $H$ can be a very complicated function of the system's state $x$.  For instance, in a system of $N$ particles, $x$ might be a list of $3N$ coordinates, and $H(x)$ might be a sum of pairwise interactions between particles.  

$$
H(q, p) = \sum_{i=1}^N \sum_{j=i+1}^N V(q_i - q_j) + \frac{1}{2} \sum_{i=1}^N V(p_i)
$$

where $V$ is the interaction potential between particles and we have separated the coordinates into position coordinates $q$ and momentum coordinates $p$.  If our system has $N=100$ particles, then $H$ is a function of $600$ variables, and there is no easy way to sample from $p(x)$ directly.  In another example, in Bayesian Inference, we might wish to calculate averages over the posterior distribution, which is given by Bayes' theorem

$$
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}
$$

where $D$ is the data, $\theta$ is the parameter, $p(D|\theta)$ is the likelihood, $p(\theta)$ is the prior, and $p(D)$ is the evidence.
In prior examples, we have discussed attempting to sample from $p(\theta)$, and mentioned how this may not be optimal at large amounts of data, when $P(D | \theta)$ is very peaked. 
If we could sample from $p(\theta | D)$ we could avoid this problem.  However, this can be a very complicated distribution, no less so because the evidence $p(D)$, much like the partition function $Z$, typically cannot be calculated directly.

### Sampling using Markov Chains

To address these problems, we give up on sampling statistically  samples.  Instead, we draw correlated samples through a technique called ``Markov chain Monte Carlo`` (MCMC).  In MCMC, we draw new samples from a distribution that is conditioned on the prior sample.
We then repeat this process until we have enough samples.  If we choos our sampling distribution correctly, averages over our samples will converge to the averages over the target distribution.  This is the basic idea behind MCMC.

Mathematically, we have 
1. A distribution $\pi(x)$ that we want to sample from.
2. A probability density $p(x_t | x_{t-1})$ that encodes the probability of moving from $x_{t-1}$ to $x_t$. We call this the transition probability.
(As always, we present our results in terms of continuous variables with probability densities, but the same ideas apply to discrete variables with probability mass functions.)

For the algorithm to work, we require that $\pi$ is the stationary distribution of the transition probability, i.e.,

$$
\int p(x_t | x_{t-1}) \pi(x_{t-1}) dx_{t-1} = \pi(x_t)
$$

If this condition holds, if we (a) randomly draw a sample $X_0$ from $\pi(x)$, and (b) draw a sample $X_1$ from $p(X_1 | X_0)$, and (c) draw a sample $X_2$ from $p(X_2 | X_1)$, and so on, then each sample $X_0, X_1, X_2, \ldots$ will be individually distributed according to $\pi(x)$.  The sequence of random variables $X_0, X_1, X_2, \ldots$ is called a Markov chain. 

```{note}
Having $\pi$ as a stationary distribution of the transition probability is a necessary condition for the algorithm to work, but it is not sufficient.  The transition probability must also be ergodic, meaning that it is possible to get from any state to any other state in a finite number of steps, and be aperiodic, meaning that the chain does not get stuck in a loop.  These are, however, much more difficult conditions to check.
```


## Gibbs Sampling and the Ising Model

One simple Markov  chain Monte Carlo example is  Gibbs sampling.  In Gibbs sampling, at each step we grab a single coordinate from our description of the system and update it by sampling from the conditional distribution of that coordinate given all the others.
Specifically, we update $x^k$, the $k$'th coordinate of $x$, by sampling the new value from the conditional distribution

$$
p(x^k | x^1, \ldots, x^{k-1},  x^{k+1}, \ldots, x^N) = \frac{\pi(x^1, \ldots, x^{k-1}, x^k, x^{k+1}, \ldots, x^N)}{\int \pi(x^1, \ldots, x^N) dx^1, \ldots, dx^{k-1}, dx^{k+1}, \ldots, dx^N}
$$


We will show that this process preserves the distribution $\pi(x)$ later, in our discussion of the Metroplis-Hastings algorithm.
For now, let's consider a specific example of Gibbs sampling.

<!-- Writing this out every time is going to be tiring, so we will adopt a shorthand notation where we write $x^{i \neq k}$ to denote all of the coordinates except for $x^k$.  In this notation, we can write the conditional distribution as

$$
p(x^k | x^{i \neq k}) = \frac{\pi(x^k, x^{i \neq k})}{\int \pi(y^k, x^{i \neq k})  dy^{ k}}.
$$ -->



###  The Ising Model

The Ising model is a simple model of a magnet.  In the Ising model, we have a lattice of spins, each of which can be either up or down.  The energy of the system is given by

$$
H(s_1, \ldots, s_M) = -J/2 \sum_{j} \sum_{i\in N_j \rangle} s_i s_j
$$

where the sum is over nearest neighbors ($N_j$ are all of the spinns neighboring spin $j$), $s_i$ is the spin at site $i$ and takes a value of $+1$ for up and $-1$ for down, and $J$ is the coupling constant.  The probability of seeing a collection of spins up or down is given by the Boltzmann distribution

$$
\pi(s_1, \ldots s_M) = \frac{1}{Z} e^{-\beta H(s_1, \ldots, s_M)}
$$

where $Z$ is the partition function.  
<!-- In Gibbs sampling, we randomly pick a single spin (denoted $s_k$) and update it according to the conditional distribution -->
To apply Gibbs sampling to the Ising model, we randomly pick a single spin (denoted $s_k$).
The conditional distribution of that spin given all the others has the probability mass function

$$
p(s_k | s_{i \neq k}) = \frac{e^{-\beta H(s_1, \ldots, s_k, \ldots, S_m )  } }{
    e^{-\beta H(s_1, \ldots, +1, \ldots, S_m )}   + 
    e^{-\beta H(s_1, \ldots, -1, \ldots, S_m )}
}
= 
\frac{e^{\beta J \sum_{i \in N_k} s_i} }{
    e^{\beta J \sum_{i \in N_k} s_i}   + 
    e^{\beta -J \sum_{i \in N_k} s_i}
}
$$

This is gives us a simple algorithm for sampling from the Ising Model:
1. We pick a random spin $s_k$.
2. We calculate the conditional probability of that spin being up or down.
3. We sample from that conditional probability to get the new value of $s_k$.
4. We repeat steps 1-3 until we have enough samples.

Note that this conditional probability is a function of only the spins neighboring $s_k$.  This means that we can update spins in parallel as long as they are not neighbors.  

## References

- If you want an example of how to implement Gibbs sampling on a 2D normal distribution where we alternate between sampling from the conditional distribution of $x$ given $y$ and $y$ given $x$, see [this blog post](https://mr-easy.github.io/2020-05-21-implementing-gibbs-sampling-in-python/)
- This recorded [video lecture](https://www.youtube.com/watch?v=vTUwEu53uzs) gives an introduction to Markov chain Monte Carlo and its applications for Bayesian Inference in Statistics.