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

# Intro to Probability

## Learning Objectives

- Understand the basic concepts of probability theory and what a probability space is.
- Understand what a Random Variable is and how it is used to describe random phenomena.
- Be proficient in using and manipulating probability mass and density functions.
- Connect probability theory to the concept of free energy in statistical mechanics.


## Probability and Chemistry

Randomness is everywhere in chemistry.  Just to name a few examples:
- Microscopic systems are randomly kicked by thermal motion, causing fluctations in their state.
- Quantum mechanics is inherently probabilistic: when we measure a quantum system, it randomly collapses to one of the possible states.
- Measurements are subject due to random errors, and we can only estimate the true value of a quantity with some uncertainty.
Probability is the mathematical framework that allows us to reason about these random phenomena.

There are two main interpretations of probability:
- **Frequentist**: Probability is the long-run frequency of an event occurring.
- **Bayesian**: Probability is a measure of the degree of belief that an event will occur.
These interpretations do not change the mathematical rules of probability, but they do change what we might find interesting or useful to calculate.
For now, we will focus on how probability is constructed, and come back to the difference in these interpretations later.

### A (Non-Rigorous) Introduction to Probability

Mathematicians describe probability using the following terms.
For any random event, we can define a probability space $(\Omega, \mathcal{F}, P)$, where:
- $\Omega$ is the set of all possible outcomes of the event.
- $\mathcal{F}$ is a set of ``events'': all the subsets of reasonable outcomes that we might want to assign a probability to.  (Formally, $\mathcal{F}$ is a $\sigma$-algebra of subsets of $\Omega$.  This is far more technical than we need to get into here.)
- $P$ is a probability measure, which assigns a number between 0 and 1 to each event in $\mathcal{F}$, such that $P(\Omega) = 1$ and $P(\emptyset) = 0$.

#### Examples:

Consider rolling a six-sided, evenly weighted die.  The probability space is $(\Omega, \mathcal{F}, P)$, where:
- Each outcome in $\Omega$ is which face of the die comes up.
- $\mathcal{F}$ is the set of outcomes: for instance, if the face is a one or two.
<!-- - $\Omega = \{1, 2, 3, 4, 5, 6\}$. -->
<!-- - $\mathcal{F} = \{\emptyset, \{1\}, \{2\}, \{3\}, \{4\}, \{5\}, \{6\}, \{1, 2\}, \{1, 3\}, \ldots, \{1, 2, 3, 4, 5, 6\}\}$. -->
<!-- - $P(\{1\}) = P(\{2\}) = \ldots = P(\{6\}) = 1/6$.  Similarly, $P(\{1, 2\}) = P(\{1, 3\}) = \ldots = 1/3$, and so on. -->
- $P$ gives the probability of each event in $\mathcal{F}$.  For example, the probability of rolling a one is $1/6$, or the probability of rolling an even number is $1/2$.

Another example is in equilibrium statistical mechanics,
- Microstates are the possible outcomes of a random event, and are elements in $\Omega$.
- Macrostates, which are sets of possible microstates, are elements in $\mathcal{F}$.
- The Boltzmann distribution is a specific example of a probability measure, which assigns a probability to each macrostate.

### What is a Random Variable?

A **random variable** is a function that maps outcomes of a random event to real numbers.
<!-- It is *not* the same thing as an element in $\Omega$: observing  -->
Care must be taken to distinguish between random variables and elements in $\Omega$.  Coming back to the die example, the random variable $X$ might be the number that comes up when the die is rolled.  $X$ is a function that maps the event "a one comes up" to the number 1, and so on.  However, these are not the same thing: one is a physical event, whereas the other is a mathematical object.  Additionally, there are many other possible random variables we can define.  For instance, another random variable would be the parity of the number that comes up: 0 if the number is even, and 1 if the number is odd.  Yet another would be the square of the number on the face.

Note that since random variables map to numbers, there is a natural way to add them together, multiply them, and so on.  For instance, if we have two dice, $X_1$ is the random variable corresponding to the number on the first die, and $X_2$ is the random variable corresponding to the number on the second die, then $X_1 + X_2$ is a new random variable corresponding to the sum of the two dice.

We can trivially extend random variables to vector-valued functions.  For instance, if we have a die and a coin, we could define a random variable $X = (X_1, X_2)$, where $X_1$ is the number on the die and $X_2$ is the result of the coin flip.  


## Quantifying Probabilities

One of the most common ways of quantifying probabilities is to use a **probability mass function** (PMF) for discrete random variables, or a **probability density function** (PDF) for continuous random variables.

### Probability Distributions

Random variables are assigned probabilities using a **probability distribution**.  This is a function that assigns probabilities to the possible values of the random variable.  Again, consider the case where $X$ is the number on a six-sided die.  The probability distribution of $X$ assigns a probability of $1/6$ to the numbers 1, 2, 3, 4, 5, and 6.  Similarly, the probability of seeing $X = 1$ or $X = 2$ is $1/3$, and so on.

To evaluate probability distributions, we often use two additional functions: "probability mass functions" (PMFs) for discrete random variables, and "probability density functions" (PDFs) for continuous random variables.  

### Probability Mass Functions (PMFs)

A PMF is a function $p(x)$ that gives the probability that a discrete random variable $X$ takes on a specific value $x$.  The PMF must satisfy two properties:
1. $0 \leq p(x) \leq 1$ for all $x$.
2. $\sum_x p(x) = 1$.
For example, the probability mass function the number seen on a six-sided die is $p(x) = 1/6$ for $x = 1, 2, 3, 4, 5, 6$.
The probabilities of events can be calculated by summing the probabilities of the outcomes in the event.
For example, the probability of rolling an even number is $p(\{2, 4, 6\}) = p(2) + p(4) + p(6) = 1/6 + 1/6 + 1/6 = 1/2$.

Importantly, note that summing a PMF is **NOT** the same same as summing the random variable.  Again, consider the sum of two dice.  
The probability of the sum being 2 is $p(2) = 1/36$, the probability of the sum being 3 is $p(3) = 2/36$, and so on.  This is clearly not the same as the PMFs for the individual rolls. 

### Probability Density Functions (PDFs)

Probability density functions are the continuous analog of probability mass functions.  A PDF $p(x)$ gives the probability that a continuous random variable $X$ takes on a value in the interval $[a, b]$ as $\int_a^b p(x) dx$.  The PDF must satisfy two properties:
1. $p(x) \geq 0$ for all $x$.
2. $\int_{-\infty}^\infty p(x) dx = 1$.
Note that, unlike the PMF case, $p(x)$ can be greater than 1 for some values of $x$.
Moreover, the probability of a continuous random variable taking on a specific value is zero: after all, $\int_a^a p(x) dx = 0$.
The PDF doesn't give probabilities, it gives a *density* of probability.  Consequently, the units of the PDF are probability per unit length / area / volume / etc, depending on the dimensionality of the random variable.

A classic example of a PDF is the "bell curve" or Gaussian probability density.  For a one-dimensional random variable, this is given by

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right),
$$

where $\mu$ is the mean of the distribution and $\sigma$ is the standard deviation.  The PDF is centered at $\mu$ and has a width of $\sigma$.  The integral of the PDF over all space is 1, as required.

Note that the PDF (or PMF) is not the same thing as probability distribution: rather, they are tools we use to describe probability distributions.
For instance, we might say that a random variable $X$ obeys the Normal, or Gaussian, distribution.  In this case, we expect the probability density function to be a bell curve, as above.  Integrating over the PDF evaluates the probabilities specified by the distribution, e.g.

$$
P([0, \infty)) = \int_0^\infty \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) dx = \frac{1}{2}.
$$

### Changing Variables Changes the PDF

One of the most important properties of probability density functions is that they change under a change of variables.  For instance, consider a random variable $X$ with PDF $p(x)$ for a real number $x$.  Let $Y = f(X)$ be a new random variable, where $f$ is an monotonic (always increasing or always decreasing) and differentiable function.  The PDF of $Y$ is given by

$$
p_Y(y) = p_X(f^{-1}(y)) \left|\frac{d}{dy} f^{-1}(y)\right|,
$$
where $f^{-1}(y)$ is the inverse function of $f$.  This is a consequence of the fact that the probability of $Y$ being in an interval $[a, b]$ is the same as the probability of $X$ being in the interval $[f^{-1}(a), f^{-1}(b)]$.

### Sampling from a Probability Distribution

Given a probability mass or density function, how might we draw samples from it?  This is a fundamental, but also extremely nontrivial, question: finding ways to efficiently sample random variables
is a going to be a major theme in our immediate future.  For now, we will focus on the simplest case: sampling from a distribution with a known probability mass function.
We assume that we have access to a random number generator that can produce random numbers uniformly distributed between 0 and 1.  Then, we can sample from a discrete distribution as follows:

1. Generate a random number $r$ between 0 and 1.
2. For each possible value of the random variable, sum the probability of that value to a running total.  When the running total exceeds $r$, return that value.

For instance, consider the die example.  We can generate a random number $r$ between 0 and 1, and then return 1 if $r < 1/6$, 2 if $1/6 \leq r < 2/6$, and so on.  

Graphically, what we have done is built a "cumulative distribution function", which is the sum of the probability mass function up to a certain value.  We can generalize this to continuous random variables as well.  For a given probability density, we can integrate the PDF up to a certain value to get the cumulative distribution function.

$$
F(x) = \int_{-\infty}^x p(x') dx'.
$$

Note that this is a monotonically increasing function that goes from 0 to 1.
To sample from the distribution by generating a random number $r$ between 0 and 1, and returning the value of $x$ such that $F(x) = r$.  This is known as the "inverse transform method".

## Doing things with Random Variables

### Expected Values

Expected values are a way of summarizing the distribution of a random variable.  The expected value of a random variable $X$ is denoted $\langle X \rangle$ or $E[X]$, and is defined as the weighted average of the possible values of $X$, weighted by their probabilities.  For a discrete random variable with PMF $p(x)$, the expected value is given by $\langle X \rangle = \sum_x x p(x)$.  For a continuous random variable with PDF $p(x)$, the expected value is given by $\langle X \rangle = \int x p(x) dx$.

We can also take the expected value of functions of random variables.  For instance, the expected value of $X^2$ is known as the second moment of $X$, and is denoted $\langle X^2 \rangle$ or $E[X^2]$.  If $X$ is discete and we have a PMF the variance is given by $\langle X^2 \rangle = \sum_x x^2 p(x)$.
Similarly, if $X$ is continuous and we have a probability density, the variance is given by $\langle X^2 \rangle = \int x^2 p(x) dx$.  For an arbitrary function $f(X)$, the expected value is given by $\langle f(X) \rangle = \sum_x f(x) p(x)$ for discrete random variables, and $\langle f(X) \rangle = \int f(x) p(x) dx$ for continuous random variables.
In classical statistical mechanics, observables are functions of random variables, and the expected value of the observable is the average value we would expect to measure if we repeated the experiment many times.

(An enterprising reader might ask, isn't $Y = f(X)$ itself a random variable?  The answer is yes, and as it happens calculating the expected value of $Y$ over the probability distribution for $Y$  is equivalent to calculating the expected value of $f(X)$ over the probability distribution for $X$.  This is known as the "law of the unconscious statistician", or LOTUS.)

Finally, we note that we can also calculate probabilities as expectation values using an "indicator function" $I_A(x)$, which is 1 if $x \in A$ and 0 otherwise.  The probability of $X$ being in the set $A$ is given by $P(X \in A) = \langle I_A(X) \rangle$.


## Connection with Free Energy

The free energy is a central quantity in statistical mechanics and thermodynamics.   Typically, we write it as the log of the partition function, 

$$
F = - k_B T \log Z =  k_B T \log \int e^{-\beta H(x)} dx,
$$

where $H(x)$ is the Hamiltonian of the system, $T$ is the temperature, and $k_B$ is the Boltzmann constant.  
<!-- The partition function is a sum over all possible states of the system, weighted by the Boltzmann factor $e^{-\beta H(x)}$.  The free energy is a measure of the system's ability to do work, and is related to the probability of observing a particular state of the system.  In particular, the probability of observing a state $x$ is given by -->
Similarly, we might evaluate the free energy of a specific state $A$ by only integrating over that state:

$$
F_A = - k_B T \log \int_A e^{-\beta H(x)} dx.
$$

(As a concrete example, if $A$ is an interval in one dimension, the integral would be evaluated as $\int_a^b e^{-\beta H(x)} dx$.)
Now, let us consider a free energy difference between two states $A$ and $B$:

$$
\Delta F = F_A - F_B = - k_B T \log \frac{\int_A e^{-\beta H(x)} dx}{\int_B e^{-\beta H(x)} dx} 
$$

Dividing both the numerator and denominator by $\int e^{-\beta H(x)} dx$, and observing that $p(x) = e^{-\beta H(x)}/ \int e^{-\beta H(x)} dx$, we find

$$
\Delta F = - k_B T \log \frac{\int_A p(x) dx}{\int_B p(x) dx} = - k_B T \log \frac{P(A)}{P(B)}.
$$

In other words, the free energy difference between two states is proportional to the log of the ratio of the probabilities of observing those states.  
