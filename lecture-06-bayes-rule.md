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
- Understand Bayes Rule, and how it is applied in inference.

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

Conditional expectations obey the ``law of total probability'' which states that for any event $A$ such that (a) the events $A_i$ are mutually exclusive and (b) at least one of the $A_i$ must occur, then

$$
E[X] = \sum_{i} E[X|A_i] P(A_i).
$$

or for a continuous random variable

$$
E[X] = \int E[X|A] p(A) dA.
$$

## Bayes Rule

The equations above  suggest a relationship between the conditional probabilities $P(A|B)$ and $P(B|A)$.  Indeed, this relationship is given by *Bayes Rule*:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}.
$$

Bayes Rule is crucial tool for inference and decision making.  A classic example is in medical diagnosis.  Suppose a patient has a rare disease that affects 1 in 10,000 people.  A test for the disease is 99% accurate, meaning that the probability of a false positive is 1%.  If a patient tests positive for the disease, what is the probability that they actually have the disease?  Let $D$ be the event that the patient has the disease, and $T$ be the event that the patient tests positive.  We are asked to find $P(D|T)$.  We can use Bayes Rule to write this as 

$$
P(D|T) = \frac{P(T|D)P(D)}{P(T)}.
$$

We know that $P(D) = 1/10000$, $P(T|D) = 0.99$, and $P(T|D^c) = 0.01$.  We can calculate $P(T)$ using the law of total probability:

$$
P(T) = P(T|D)P(D) + P(T|D^c)P(D^c) = 0.99 \times 0.0001 + 0.01 \times 0.9999.
$$ 

Plugging in the values, we find that $P(D|T) \approx 0.0098$, meaning that the patient has less than a 1\% chance of having the disease, even after testing positive.  This is because the disease is so rare: the vast majority of people we test do not have the disease, so even a very accurate test will produce many false positives. 

### Bayesian Inference

Bayesian inference is a method of statistical inference that uses Bayes Rule to update our beliefs about a hypothesis.  As discussed previously, in Bayesian statistics probability is a measure of our belief in which hypotheses are true.  Consequently, we begin by assigning a prior probability to each hypothesis, $P(H)$.   The prior encodes how strongly we believe a given hypothesis could be correct before we have seen any data.
Then, we are given a set of data $D$, and we want to update our beliefs about the hypothesis.  In other words, we want to calculate the posterior probability of the hypothesis given the data, $P(H|D)$.  Bayes Rule tells us that

$$
P(H|D) = \frac{P(D|H)P(H)}{P(D)}.
$$

The term $P(D|H)$ is called the *likelihood* of observing the data: it can be interpreted as the probability that, if a hypothetical experiment were to be performed where the hypothesis $H$ is true, the data $D$ would be observed.  The term $P(D)$ is the evidence, or the probability of observing the data under any hypothesis.  Applying this formala then gives us a new probability distribution over the hypotheses, called the *posterior* probability.  We can use the posterior to make decisions or predictions.  For instance, we can see how much more we belief one hypothesis over another, by calculating the ratio of the posterior probabilities, called the *Bayes factor*.  We can also calculate the expected value of a hypothesis given the data, called the *posterior expectation*, or calculate other averages over the space of hypotheses.

### References on Bayes Rule:

- ThreeBlueOneBrown, of whom I am a huge fan, has a [great video](https://www.youtube.com/watch?v=HZGCoVF3YvM) on Bayes Rule.
- Nature has an article on Bayesian statistics that is a little more involved, you can find it [here](https://doi.org/10.1038/s43586-020-00001-2).