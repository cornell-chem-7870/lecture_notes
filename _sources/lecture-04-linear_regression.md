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

### Least Squares

To get the best approximation possible, we will attempt to minimize the length of the difference vector $Y - X W$.

$$
    W_{\text{LS}} = \text{argmin}_W \| X W - Y\|_2^2
$$

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


