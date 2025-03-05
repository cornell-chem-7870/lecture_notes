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

# Conditional Probability and Bayes Rule

## Learning Objectives

- Understand the Concept of Conditional Probability.

### Conditional Probabilities:

Conditional probabilities are a way of quantifying the probability of an event given that  event has occurred.  The conditional probability of event $A$ given event $B$ is denoted $P(A|B)$.  This is the probability of $A$ occurring, given that $B$ has occurred.  For instance, the probability of rolling a two on a die, given that the number is even, is $P(\{2\}|\{2, 4, 6\}) = 1/3$.  The conditional probability of rolling a one on a die, given that the number is even, is $P(\{1\}|\{2, 4, 6\}) = 0$.

The conditional probability is, in general, given by 

$$
P(A|B) = \frac{P(A \cap B)}{P(B)}.
$$

where $P(A \cap B)$ is the probability that both $A$ and $B$ occur, often called the "joint probability" of $A$ and $B$.

One specific case is where we ask, "what is the probability mass function given that a condition is true"?  Effectively, we are asking for the probability of observing a specific value of a random variable given event.  This is known as the "conditional probability mass function."  Applying the definition of conditional probability, we have

$$
p(x | A) = \begin{cases}
\frac{p(x)}{\sum_{x' \in A} p(x)} & \text{if } x \in A, \\
0 & \text{otherwise}.
\end{cases}
$$

Similarly, we can write a "conditional probability density function" as

$$
p(x | A) = \begin{cases}
\frac{p(x)}{\int_A p(x) dx} & \text{if } x \in A, \\
0 & \text{otherwise}.
\end{cases}
$$