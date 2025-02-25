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

# Linear Regression and Least Squares

## Learning Objectives

## Intro to Linear Regression

In linear regression, we are given a set of data points $(x_i, y_i)$ and we want to find a line that best fits the data.
We consider the case where $x_i$ is multi-dimensional, with $m$ features.
Finding the line that best fits the data corresponds to finding an $m$-dimensional vector $\mathbf{w}$ such that 

$$
y_i \approx w \cdot x_i + b
$$

To solve this problem, our first step is to write it as a matrix equation.
We introduce the matrix $X$ with entries

$$
X = \begin{bmatrix}
1 & x_{1,1} & x_{1,2} & \cdots & x_{1,m} \\
1 & x_{2,1} & x_{2,2} & \cdots & x_{2,m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & x_{n,2} & \cdots & x_{n,m} \\
\end{bmatrix}
$$

where $n$ is the number of data points.  We can then write the linear regression problem as

$$
\begin{bmatrix}
y_1 \\
y_2 \\
y_3 \\
\vdots \\
y_n
\end{bmatrix}
\approx
\begin{bmatrix}
1 & x_{1,1} & x_{1,2} & \cdots & x_{1,m} \\
1 & x_{2,1} & x_{2,2} & \cdots & x_{2,m} \\
1 & x_{3,1} & x_{3,2} & \cdots & x_{3,m} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & x_{n,2} & \cdots & x_{n,m} \\
\end{bmatrix}
\begin{bmatrix}
b \\
w_1 \\
w_2 \\
\vdots \\
w_m
\end{bmatrix}
$$

or more succinctly as

$$
Y \approx X W
$$
where $Y$ is the vector of $y_i$ values, $X$ is the matrix of $x_i$ values, and $W$ is the vector of $w_i$ values and $b$.

## Least Squares

To get the best approximation possible, we will attempt to minimize the length of the difference vector $Y - X W$.

$$
    W_{\text{LS}} = \text{argmin}_W \| X W - Y\|_2^2 = \text{argmin}_W \sum_{i=1}^n |x_i \cdot W - y_i|^2
$$

This strategy is called the method of least squares for pretty obvious reasons.
(The notation $\| \cdot \|_2$ denotes the Euclidean norm, which is the square root of the sum of the squares of the elements.
So far we have not discussed any other norms, so this notation is redundant, but it will be good for us to get used to it.)

To solve this, we will take the SVD of $X$ (here we are using the convention that $U$ and $V$ are square orthogonal matrices).

$$
\| X W - Y\|_2^2 = \| U \Sigma V^\dagger W - Y\|_2^2 = \| \Sigma V^\dagger W - U^\dagger Y\|_2^2
$$

Where the second equality follows that (a) $U$ is an orthogonal unitary and (b) orthogonal matrices preserve the Euclidean norm.
Moreover, for simplicity we can instead minimize over $Z = V^\dagger W$ rather than $W$ directly.

Writing the Euclidean norm out explicitly, we have that

$$
\| X W - Y\|_2^2 = \sum_{i=1}^r |\sigma_i z_i - (U^\dagger Y)_i|^2  + \sum_{i=r+1}^M |(U^\dagger Y)_i|^2
$$
where $r$ is the number of non-zero singular values of $X$.
We can clearly minimize this by setting $z_i = \sigma_i^{-1} (U^\dagger Y)_i$ for $i \leq r$ and making an arbitrary choice for $z_i$ for $i > r$.
For simplicity, we will set $z_i = 0$ for $i > r$.

Recalling that $Z = V^\dagger W$, implying that $V Z = W$, we can write the solution as

$$
W_{\text{LS}} = V \Sigma^{+} U^\dagger  Y
$$

where $\Sigma^{+}$ is the matrix with the reciprocals of the non-zero singular values of $\Sigma$ on the diagonal and zeros elsewhere.

Remark: The matrix $V \Sigma^{+} U^\dagger$ is called the Moore-Penrose pseudoinverse of $X$ and is denoted $X^+$.  This is a generalization of the inverse of a matrix to non-square matrices
or matrices that are not full rank, and obeys the following properties:
1. $X X^+ X = X$
2. $X^+ X X^+ = X^+$
3. $(X X^+)^\dagger = X X^+$
4. $(X^+ X)^\dagger = X^+ X$

### Polynomial and Basis Regression

The above discussion can be generalized to polynomial regression.
In this case, we have a set of data points $(x_i, y_i)$ and we want to find a polynomial of degree $d$ that best fits the data.
Rather than using the features $x_i$ directly, we will first take all monomials of degree $d$ or less of the features, and use them to 
build a larger $X$ matrix.
For example, if $d = 2$ and $m = 2$, we would have

$$
X = \begin{bmatrix}
1 & x_{1,1} & x_{1,2} & x_{1,1}^2 & x_{1,1} x_{1,2} & x_{1,2}^2 \\
1 & x_{2,1} & x_{2,2} & x_{2,1}^2 & x_{2,1} x_{2,2} & x_{2,2}^2 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
1 & x_{n,1} & x_{n,2} & x_{n,1}^2 & x_{n,1} x_{n,2} & x_{n,2}^2 \\
\end{bmatrix}
$$

We then proceed as before.  More generally, we can consider a basis of functions $f_i(x)$ and write the regression problem as

$$
Y \approx X W = \sum_{i=1}^m f_i(x) w_i
$$

where $f_i(x)$ are the basis functions and $w_i$ are the coefficients we want to find.
Here, our matrix $X$ is given by

$$
X = \begin{bmatrix}
f_1(x_1) & f_2(x_1) & \cdots & f_m(x_1) \\
f_1(x_2) & f_2(x_2) & \cdots & f_m(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
f_1(x_n) & f_2(x_n) & \cdots & f_m(x_n) \\
\end{bmatrix}
$$

### Application: Calculating the Committor Function

We consider a system with $N$ states, which hops from one state withthe following matrix of transition probabilities:

$$
T_{ij} = \mathbb{P}(X_{t+1} = j | X_t = i)
$$

where $X_i$ is the state of the system.  The "|" symbol is read as "given" and is used to denote conditional probabilities.
The committor function is a function that tells us the probability that a system will reach a certain set of states (set $B$) before another set of states (set $A$).

$$
q_i = \mathbb{P}(\text{reach } B \text{ before } A | X_i)
$$

This is a key quantity in the study of rare events, and is used to calculate the rate of transitions between states in a field known as transition path theory.
The committor function can be calculated by solving the linear regression problem using the transition matrix of the system, which is a matrix that tells us the probability of transitioning from one state to another.
Specifically, the committor function obeys $q_i = 0$ for $i \in A$ and $q_i = 1$ for $i \in B$, and satisfies the equation

$$
q_i = \sum_j T_{ij} q_j
$$

for all other states.  This is is because the probability of reaching either $B$ is the same as the probability of reaching $B$ from any of the states that can be reached from $i$, weighted by the probability of reaching those states.

To write this as a linear regression problem, denote by $D$ the set of states that are not in $A$ or $B$.  We can rewrite our sum as

$$
q_i = \sum_{j \in D} T_{ij} q_j + \sum_{j \in A} T_{ij} * 0 + \sum_{j \in B} T_{ij} * 1
$$

implying

$$
\sum_{j \in D} ( \delta_{ij} - T_{ij}) q_j + \sum_{j \in B} T_{ij}
$$

where $\delta_{ij}$ is the Kronecker delta function, which is 1 if $i = j$ and 0 otherwise.
We can then write this as a matrix equation

$$
(I - T^D) q = b
$$

where $I$ is the identity matrix, $T^D$ is the entries of the transition matrik where both $i$ and $j$ are in $D$, and $b$ is a vector with entries $\sum_{j \in B} T_{ij}$ for all $i$ in $D$.

## Tikhonov Regularization

In practice, we often have to deal with noisy data, which can lead to overfitting.
We also might also have many more parameters than data, or have a matrix with small singular values, leading to numerical instability. 
To deal with these issues, we use *regularization*.  In regularization, we modify the problem to make it more stable or to prevent overfitting,
at the cost of introducing some bias.

One simple way of regularizing a problem is to simply drop singular values that are below a certain threshold.   This is certainly not a bad approach, but maybe we want something a little more gradual that we can tune.

One common approach is to add a penalty term to the least squares problem.  This is called Tikhonov regularization.  Rather than solving our standard least squares problem, we instead solve the problem

$$
W_{\text{reg}} = \text{argmin}_W \| X W - Y\|_2^2 + \alpha \| W\|_2^2
$$

where $\alpha$ is a parameter that we can tune to control the amount of regularization.
This problem can be solved by taking the SVD of $X$ and solving the following equation

$$
W_{\text{reg}} = V (\Sigma^2 + \alpha I)^{-1} \Sigma U^\dagger Y
$$

where $\Sigma^2$ is the matrix with the square of the singular values of $\Sigma$ on the diagonal and zeros elsewhere.
As we increase the $\alpha$, we slowly reduce noise sensitivity, at the cost of introducing more bias.
To choose the best value of $\alpha$, one very simple approach is to use the L-curve method. 
In this method, we plot the norm of the residual $\| X W - Y\|_2^2$ as a function of $\| W\|_2^2$ for different values of $\alpha$.
on a log-log plot.  We then look for the point where the curve has the sharpest bend: this is a  good value of $\alpha$ to choose.

### Deriving Tikhonov regularization

To derive this expression, we again take the SVD of $X$.  
For simplicity, this time we assume that $U$, $\Sigma$, $V$ are real.
We can then write the least squares problem as
  <!-- and write the quantity we are minimizing as  -->

$$
\| X W - Y\|_2^2 + \alpha \| W\|_2^2 
= \| U \Sigma V^t W - Y\|_2^2 + \alpha \| W\|_2^2 
= \| \Sigma V^t W - U^t Y\|_2^2 + \alpha \| V^t W\|_2^2
$$

We have again used the fact that both $U$ and $V$ are orthogonal matrices.
Again, defining $Z = V^t W$, we can write this as 

$$
\| \Sigma Z - U^t Y\|_2^2 + \alpha \| Z\|_2^2
= \sum_{i=1}^M (\sigma_i z_i - (U^t Y)_i)^2 + \alpha  z_i^2
= \sum_{i=1}^M \sigma_i^2 z_i^2 - 2 \sigma_i z_i (U^t Y)_i
 + (U^t Y)_i^2 
$$

To calculate the minimum, we take the derivative with respect to $z_i$ and set it to zero.  This gives

$$
2 (\sigma_i^2 + \alpha) z_i = 2 \sigma_i (U^t Y)_i
$$

or

$$
z_i = \frac{\sigma_i (U^t Y)_i}{\sigma_i^2 + \alpha}
$$

Writing this as a matrix equation, we have

$$
Z = \left( \Sigma^2 + \alpha I \right)^{-1} \Sigma U^t Y
$$

This gives us the solution for $Z$.  To get the solution for $W$, we use the fact that $Z = V^t W$, and write

$$
W_{\text{reg}} = V Z = V \left( \Sigma^2 + \alpha I \right)^{-1} \Sigma U^t Y
$$













