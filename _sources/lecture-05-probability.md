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

## Doing things with Random Variables

### Expected Values

Expected values are a way of summarizing the distribution of a random variable.  The expected value of a random variable $X$ is denoted $\langle X \rangle$ or $E[X]$, and is defined as the weighted average of the possible values of $X$, weighted by their probabilities.  For a discrete random variable with PMF $p(x)$, the expected value is given by $\langle X \rangle = \sum_x x p(x)$.  For a continuous random variable with PDF $p(x)$, the expected value is given by $\langle X \rangle = \int x p(x) dx$.

We can also take the expected value of functions of random variables.  For instance, the expected value of $X^2$ is known as the second moment of $X$, and is denoted $\langle X^2 \rangle$ or $E[X^2]$.  If $X$ is discete and we have a PMF the variance is given by $\langle X^2 \rangle = \sum_x x^2 p(x)$.
Similarly, if $X$ is continuous and we have a probability density, the variance is given by $\langle X^2 \rangle = \int x^2 p(x) dx$.  For an arbitrary function $f(X)$, the expected value is given by $\langle f(X) \rangle = \sum_x f(x) p(x)$ for discrete random variables, and $\langle f(X) \rangle = \int f(x) p(x) dx$ for continuous random variables.
In classical statistical mechanics, observables are functions of random variables, and the expected value of the observable is the average value we would expect to measure if we repeated the experiment many times.

(An enterprising reader might ask, isn't $Y = f(X)$ itself a random variable?  The answer is yes, and as it happens calculating the expected value of $Y$ over the probability distribution for $Y$  is equivalent to calculating the expected value of $f(X)$ over the probability distribution for $X$.  This is known as the "law of the unconscious statistician", or LOTUS.)


### Conditional Probabilities:

Conditional probabilities are a way of quantifying the probability of an event given that another event has occurred.  The conditional probability of event $A$ given event $B$ is denoted $P(A|B)$.  This is the probability of $A$ occurring, given that $B$ has occurred.  For instance, the probability of rolling a two on a die, given that the number is even, is $P(\{2\}|\{2, 4, 6\}) = 1/3$.  The conditional probability of rolling a one on a die, given that the number is even, is $P(\{1\}|\{2, 4, 6\}) = 0$.

The conditional probability is, in general, given by 

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}.
$$

where $P(A \cap B)$ is the probability that both $A$ and $B$ occur, often called the "joint probability" of $A$ and $B$.

For continuous random variables.
