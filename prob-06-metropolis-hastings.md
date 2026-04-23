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

# Detailed Balance and the Metropolis-Hastings Algorithm

## Learning Objectives

- Generalize Gibbs Sampling to the Metropolis-Hastings algorithm.
- Connect Detailed Balance to preserving the target distribution.


##  Detailed Balance

Last class, we considered the Gibbs sampler.  We said, without any particular justification, that the Gibbs sampler preserves the target distribution.  To show that this is true, we first observe that the Gibbs sampler obeys a property called "detailed balance":

$$
\pi(x) p(x' | x)  = \pi(x') p(x | x')
$$

Any Markov chain that obeys detailed balance will have a stationary distribution that is equal to the target distribution,
as

$$
\int \pi(x) p(x' | x) dx = \int \pi(x') p(x | x') dx = \pi(x')
$$

### The Gibbs Sampler and Detailed Balance

To see that the Gibbs sampler obeys detailed balance, we observe that the transition probability is given by

$$
p(y | x) = p(y_k| x_{i \neq k}) \prod_{i \neq k} \delta(y_i - x_i)
$$

where $x_k$ is the $k$'th variable.  From the definition of detailed balance and of the conditional probability density, we have that

$$
\pi(x) p(y | x) = \pi(x)  \left( \frac{\pi(x_1, \ldots, y_k, \ldots x_M)}{\int  \pi(x_1, \ldots, z_k, \ldots x_M) } dz_k\right) \left(\prod_{i \neq k} \delta(y_i - x_i)\right)
$$

Using the properties of the delta function, this is equal to

$$
\pi(x) p(y | x) = \pi(y_1, \ldots, x_k, \ldots, y_M)  \left( \frac{\pi(y_1, \ldots, y_k, \ldots y_M)}{\int  \pi(y_1, \ldots, z_k, \ldots y_M) } dz_k\right) \left(\prod_{i \neq k} \delta(x_i - y_i)\right)
$$

which is in turn equal to $\pi(y) p(x | y)$, so the Gibbs sampler obeys detailed balance.


## The Metropolis-Hastings Algorithm

The Metropolis-Hastings algorithm is a generalization of the Gibbs sampling algorithm.  In the Metropolis-Hastings algorithm, we propose a new state $x'$ from a proposal distribution $q(x' | x)$, and then accept the new state with probability

$$
\alpha(x, x') = \min\left(1, \frac{\pi(x') q(x | x')}{\pi(x) q(x' | x)}\right)
$$

If we accept the new state, we set $x = x'$, otherwise we keep the old state.  
(In many applications, people choose a symmetric proposal distribution, so that $q(x' | x) = q(x | x')$.  This makes evaluating this ratio easier.)
The transition probability for Metropolis Hastings is given by 

$$
p(x' | x) = q(x' | x) \alpha(x, x') + \delta(x' - x) (1 - \int q(x' | x) \alpha(x, x') dx')
$$

To show that the Metropolis-Hastings algorithm obeys detailed balance, we consider two cases.  If $x' = x$, then the equation is trivially satisfied.  If $x' \neq x$, then the delta function is zero, and we have

$$
\begin{split}
\pi(x) p(x' | x) =& \pi(x) q(x' | x) \alpha(x, x') 
    =  \min \left(\pi(x) q(x' | x), \pi(x') q(x | x')\right) \\
     =&
     \pi(x') q(x | x') \alpha(x', x) = \pi(x') p(x | x')
\end{split}
$$

Consequently, the Metropolis-Hastings algorithm obeys detailed balance.
Note that if we set $q(x' | x)$ to be the conditional distribution of the $i$'th variable given all the others, then the Metropolis-Hastings algorithm reduces to the Gibbs sampling algorithm.x

