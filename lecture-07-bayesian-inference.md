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

# Bayesian Inference

## Learning Objectives

- Develop a sense for constructing Priors and likelihoods
- Understand how to evaluate the evidence (and when you might not want to).
- Understand how to use the posterior to make predictions.
- Become familiar with working in log-space for numerical stability.

## Bayesian Inference

In our previous discussion of Bayesian Inference, we advanced an approach for evaluating statistical hypotheses.  Given a set of data, $D$, and hypothesis for the model that generated the data, $H$, we can evaluate the posteriar probability of the hypothesis given the data, $P(H|D)$ using Bayes Rule:

$$
P(H|D) = \frac{P(D|H)P(H)}{P(D)}.
$$

To recap, the terms in this equation are as follows.
- $P(H|D)$ is the posterior probability of the hypothesis given the data.
- $P(D|H)$ is the likelihood of the data given the hypothesis, which represents the probability of observing the data given that the hypothesis is true.
- $P(H)$ is the prior probability of the hypothesis, which represents our belief in the hypothesis before seeing the data.
- $P(D)$ is the evidence, or the probability of the data.

We first discuss the construction of the prior, before moving on to the likelihood and finally the evidence.

### Prior

The prior is the probability of the hypothesis before seeing the data: typically this is something that you choose based on your knowledge of the system.  In general, the hypothesis is encoded in one or more parameters that specify the hypothesis.  

For example, if you are trying to determine the probability that a coin comes up heads you might choose a prior for the probability of heads.  Some examples of priors are:
- If you have no idea what the probability is, you might choose a prior that is uniform over the interval $[0, 1]$.
- If you have good reason to believe that the coin is fair, you might choose a prior that is peaked at $0.5$.
- If say someone came to you with a glimmer in their eye and said "Want to place a bet that this coin will come up heads?" you probably have a good reason to believe they are trying to cheat you and might choose a prior that has larger values at very low probabilities!

Priors can also encode experimental constraints.  For example, say you are determining the time it takes for a radioactive isotope to decay.  In this case, our hypothesis is specified by the decay constant..  You might not have a good idea of what the decay constant is, but you do know that it must be positive.  In this case, you might choose a prior that is uniform over the interval $[0, \infty)$, but zero from $-\infty$ to $0$.

Typically, we call uniform, or very flat priors "uninformative:" in most cases, we use them when we have no information about the system. 
However, if we have strong beliefs about the system we might use an "informative" prior, which prefers certain values of the hypothesis over others.

### Likelihood

The likelihood is the probability of observing the data given the hypothesis.  Constructing the likelihood is often the most challenging part of Bayesian Inference, and often involves explicitly modelling any noise or randomness in the data.  For example, if you are measuring the time it takes for a radioactive isotope to decay, you might model the decay time as a random variable with an exponential distribution.  The likelihood is then the probability of observing the data given the decay time.  We can associate this likelihood with the probability density function of the random variable,

$$
p(x | \lambda) = \lambda e^{-\lambda x},
$$

where $d$ is the observed decay time, and $\lambda$ is the rate parameter of the exponential distribution.  The likelihood is then the probability of observing the data given the rate parameter.  In general, it is common to also refer to the probability density or mass function as the likelihood as well, and just have which one we are talking about be clear from context.  Hereafter we will refer to the probability density, and trust the reader to generalize to the probability mass function.


Another common example is the Gaussian likelihood, where we assume the observed data comes from a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$.  The probability density for the likelihood is given by.

$$
p(y | \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(y - \mu)^2}{2\sigma^2}}.
$$

### Evidence

The final piece needed for Bayes Rule is the evidence, or the probability of the data.  This is often the most challenging part of Bayesian Inference, as it involves integrating over all possible hypotheses.  Fortunately, in many cases, we might not need it:
- If we only want to find the most probable hypothesis (aka the *maximum a posteriori* or MAP estimate), we can ignore the evidence, as it is a constant that does not depend on the hypothesis.
- If we want to compare the relative probabilities of two hypotheses, we can calculate the ratio of the posteriors, called the *Bayes factor*, given in the equation below.  In this case, the evidence cancels out.

$$
\text{Bayes Factor} = \frac{P(H_1|D)}{P(H_2|D)} = \frac{P(D|H_1)P(H_1)}{P(D|H_2)P(H_2)}.
$$

However, in other cases we might need to calculate the evidence, or at least the probability density or probability mass function associated.  
Denote the possible value for the random variable as $x$ and the parameter specifying the hypothesis as $\theta$.  
For example, if we want to calculate expected values over the posterior:
- The expected value for a given  parameter is given by $\int \theta p(\theta|x) d\theta$. 
- Similarly, its variance is given by $\int \theta^2 p(\theta|x) d\theta - \left(\int \theta p(\theta|x) d\theta\right)^2$.
In both of these cases we need to know the probability density for the posterior.
From Bayes Rule, we expect that

$$
p(\theta | x ) \propto {p(x | \theta) p(\theta)}.
$$

To be able to take expectations over $\theta$, we need to normalize this distribution.  This requires calculating $\int p(x | \theta) p(\theta) d\theta$.  Using the law of total probability, we see that this is $p(x)$: the the probability density associated with the evidence.  In general, calculating this integral is difficult.  For now, we will focus on the case where we can ignore the evidence, or when we can approximate this integral using quadrature.


## Working with multiple data points 

If we have multiple data points, $D = \{y_1, y_2, \ldots, y_n\}$, and we assume that the data points are independent and identically distributed (i.i.d.), we can build a new likelihood using the fact that probabilities of independent events multiply.  In this case, the likelihood is the product of the likelihoods of the individual data points:

$$
p(y_1, \ldots, y_n | \theta) = \prod_{i=1}^n p(y_i | \theta),
$$

This leads to a problem: we are multiplying many small numbers together, which can quickly become numerically unstable (i.e., the numbers become too small to be represented by the computer).  

### Working in log-space

To avoid this, we can work in log-space.  Rather than taking the product of the likelihoods, we take the sum of the log-likelihoods:

$$
\log p(y_1, \ldots, y_n | \theta) = \sum_{i=1}^n \log p(y_i | \theta).
$$

Note that the prior then becomes an additive term in the log-likelihood, and the evidence becomes an additive constant:

$$
\log p(\theta | y_1, \ldots, y_n) = \sum_{i=1}^n \log p(y_i | \theta).  + \log p(\theta)  + \text{constant}.
$$

This term also shows us how to balance the relative importance of the prior and the likelihood.  At small amounts of data, the prior can have a considerable effect.  However, as we keep adding data, the likelihood term will dominate.  This is a key feature of Bayesian Inference: the more data you have, the more the data will drive the posterior probability and the less the prior will matter.

## Bayesian Linear Regression
The true power of Bayesian Inference comes from its ability to handle complex models.  For example, consider the case of linear regression.  In linear regression, we assume that the data points $y_i$ are generated by a linear model from some known quantity $x_i$ with unknown slope $m$ and intercept $b$.  In this case, the likelihood is given by

$$
p(y_1, \ldots, y_n | m, b) = \prod_{i=1}^n p(y_i | m, b, x_i),
$$

where $p(y_i | m, b, x_i)$ is the probability density of the Gaussian distribution with mean $m x_i + b$ and standard deviation $\sigma$,

$$
p(y_i | m, b, x_i) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(y_i - m x_i - b)^2}{2\sigma^2}}.
$$

In this case, the hypothesis is specified by the parameters $m$ and $b$.  We can then use Bayes Rule to calculate the posterior probability of the parameters given the data.  This is a powerful tool, as it allows us to quantify the uncertainty in the parameters of the model.  For example, we can calculate the probability that the slope is positive, or the probability that the slope is within some range.

If we have a large amount of data, our model is dominated by the likelihood term.  Seeking the maximum a posteriori estimate is equivalent to maximizing the log-likelihood, which is in turn given by

$$
\log p(y_1, \ldots, y_n | m, b) = \sum_{i=1}^n \log p(y_i | m, b, x_i) = -\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - m x_i - b)^2 + \text{constant}.
$$

This is precisely the same as the least squares estimate for linear regression.  As such, our Bayesian estimate generalizes the least squares estimate, and provides a principled way to quantify its uncertainty.

## References
- [This Link](https://www.statlect.com/fundamentals-of-statistics/Bayesian-inference) contains much of the information in this lecture, presented in a slightly different way.
- I found a video on [Bayesian Statistics](https://www.youtube.com/watch?v=3jP4H0kjtng) that seems good, but I haven't watched it closely.  It does seem very meme heavy, so be prepared for that.