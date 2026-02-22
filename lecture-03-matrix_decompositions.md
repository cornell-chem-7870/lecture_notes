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

# Eigenvectors, Unitary Matrices, and Matrix Decompositions

## Learning Objectives

- Map Quantum systems to matrix equations.
- Understand the concept of eigenvectors and eigenvalues, in particular the specific case of Hermitian matrices.
- Understand the concept of a unitary matrix and its properties.
- Understand the concept of matrix decompositions, most notably the Singular Value Decomposition (SVD).

## Matrices and quantum mechanics

We have previously discussed an interpretation of wavefunctions in terms of vectors.
Here, we are going to make this more explicit by considering the matrix representation of quantum mechanical operators.
We will see that many of the concepts we have discussed in the context of linear algebra have direct analogs in quantum mechanics.

First, we restrict our attention to finite-dimensional quantum systems.  Then, each wavefunction can be represented as a sum of a set of $n$ basis functions:

$$
\psi(x) = \sum_{i=1}^n c_i \phi_i(x)
$$

where $\phi_i(x)$ are the basis functions, and $c_i$ are the coefficients of the expansion.
For now, we assume our basis functions are orthonormal: that is, $\int \phi_i(x) \phi_j(x) dx = \delta_{ij}$. 
Infinite dimensional systems are also possible, but require a much more careful treatment: the development of functional analysis in the 20th century was largely driven by the need to understand these systems.

Applying an operator $\hat{A}$ to a wavefunction $\psi(x)$ gives a new wavefunction $\psi'(x)$, with it's own expansion coefficients.
In fact, this is true for each $\phi_i(x)$:

$$
\hat{A} \phi_i(x) = \sum_{j=1}^n a_{ij} \phi_j(x)
$$

Since quantum mechanical operators are linear, we can rewrite the application of $\hat{A}$ to $\psi(x)$ as

$$
\hat{A} \psi(x) = \sum_{i=1}^n c_i \hat{A} \phi_i(x) = \sum_{i=1}^n c_i \sum_{j=1}^n a_{ij} \phi_j(x)
$$

Effectively, we have a matrix equation: applying the operator $\hat{A}$ to the wavefunction $\psi(x)$ is equivalent to multiplying the vector of coefficients $c_i$ by the matrix $A_{ij} = a_{ij}$.  This matrix is called the matrix representation of the operator $\hat{A}$ in the basis $\phi_i(x)$.
Consequently, we can even stop thinking about the wavefunction as a function, and instead think of it as a vector of coefficients.



## Eigenvectors and Eigenvalues

Equipped with this new perspective, we will revisit many of the math underlying quantum mechanics from our new, matrix-based perspective.
We begin with eigenvectors and eigenvalues.

### Definition

Given a matrix $A$, a vector $v$ is an eigenvector of $A$ if $Av = \lambda v$ for some scalar $\lambda$.
The scalar $\lambda$ is called the eigenvalue corresponding to the eigenvector $v$.
Immediately, a few things should be clear:
* For a matrix $A$ to be an eigenvector, it must be square.
* The eigenvector $v$ is not unique: any scalar multiple of $v$ is also an eigenvector.
* Two eigenvectors with the same eigenvalue can be summed to form a new eigenvector:
if $Av_1 = \lambda v_1$ and $Av_2 = \lambda v_2$, then $A(v_1 + v_2) = \lambda(v_1 + v_2)$.
   * We call the space of formed by all possible linear combinations of eigenvectors corresponding to a particular eigenvalue the eigenspace of that eigenvalue.
   * The dimension of the eigenspace is called the geometric multiplicity of the eigenvalue.


If we allow our eigenvalues and eigenvalues to be complex, then every square matrix has $n$ eigenvalues, counted with multiplicity.
These eigenvalues don't have to nonzero though:
consider the matrix

$$
A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
$$

This matrix has only one eigenvalue, 0, with a single eigenvector $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

## Eigendecomposition

We say a matrix $A$ is diagonalizable if it can be written as $A = V \Lambda V^{-1}$, where $V$ is a matrix whose columns are the eigenvectors of $A$, and $\Lambda$ is a diagonal matrix whose diagonal entries are the eigenvalues of $A$.
Immediately, it should be clear that not all matrices are diagonalizable.  If we again consider the matrix

$$
A = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
$$

any possible eigendecomposition would have take the form

$$
A = V \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix} V^{-1}
$$

which is clearly an impossibility.  Matrices that are not diagonalizable are called *defective*.  (This always seemed to me like a very mean thing to call a matrix.)

### Application of Eigendecomposition: Exponential of a Matrix

One application of Eigendecompositions is to compute the exponential of a matrix.
Given a matrix $A$ whose eigendecomposition exists, we can write $A = V \Lambda V^{-1}$.
Then, 

$$
e^A = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \ldots
$$

Substituting in the eigendecomposition of $A$, we have

$$
e^A = V \left( I + \Lambda + \frac{\Lambda^2}{2!} + \frac{\Lambda^3}{3!} + \ldots \right) V^{-1}
$$

Since $\Lambda$ is diagonal, we can further write

$$
e^A = V \begin{bmatrix} e^{\lambda_1} & 0 & 0 & \ldots \\ 0 & e^{\lambda_2} & 0 & \ldots \\ 0 & 0 & e^{\lambda_3} & \ldots \\ \vdots & \vdots & \vdots & \ddots \end{bmatrix} V^{-1}
$$

This is a very useful result: it allows us to compute the exponential of a matrix by computing the exponential of its eigenvalues.
For instance, consider a matrix of first-order rate constants in a chemical system:

$$
K = \begin{bmatrix} -k_{12} & k_{21} & 0 \\ k_{12} & -k_{21} - k_{23} & k_{32} \\ 0 & k_{23} & -k_{32} \end{bmatrix}
$$

If the concentration vector of the system at time $t$ is $c(t)$, then the time evolution of the system is given by the equation

$$
\frac{dc}{dt} = K c
$$

The solution to this equation is $c(t) = e^{Kt} c(0)$, where $c(0)$ is the initial concentration vector.  (To prove this, sustitute the definition of the matrix integral into the differential equation, differentiate by $t$, and simplify.)
We can compute $e^{Kt}$ by computing the eigendecomposition of $K$, and then computing the exponential of the eigenvalues.


### Hermitian Matrices  

One specific case of interest, is when a matrix is *Hermitian*: that is, when $A = A^\dagger$.
A famous theorem in linear algebra, the spectral theorem for Hermitian matrices,
states that all Hermitian matrices are diagonalizable.
Moreover,  it immediately follows that 
eigenvalues are real, since

$$
   (x^\dagger x) \lambda = x^\dagger A x =  (A x)^\dagger x = \lambda^* (x^\dagger x)
$$

Moreover, eigenvectors corresponding to different eigenvalues are orthogonal.  Denoting the eigenvectors corresponding to different eigenvalues as $v_i$ and $v_j$, we have

$$
\lambda_j v_i^\dagger v_j = v_i^\dagger A v_j =  
(A v_i)^\dagger v_j = \lambda_i^* v_i^\dagger v_j
$$

which implies that $v_i^\dagger v_j = 0$.



This has an important consequence: we can write our eigenvectors in a matrix $V$, such that $V^\dagger V = I$:
orthogonality implies that

$$
(V^\dagger V)_{ij} =  v_i^\dagger v_j = \delta_{ij}
$$

Consequently, if $A$ is a Hermitian matrix, there exists an eigendecomposition
such that

$$
A = V \Lambda V^\dagger
$$

The properties of Hermitian matrices form the basis for a lot of quantum mechanics:
* Since the eigenvalues of a Hermitian matrix are real, the observed values of quantum mechanical observables are real.
* Since the eigenvectors of a Hermitian matrix are orthogonal, we do not see transitions between different states in quantum mechanics
when observing a system in a state in which we are in an eigenstate of the observable.

## Unitary Matrices

The property of obeying $A^\dagger A = I$ is true for a specific class of matrices: the unitary matrices.
A matrix $U$ is unitary if its conjugate transpose is its inverse: $U^\dagger = U^{-1}$: this is equivalent to $U^\dagger U = UU^\dagger = I$.

Unitary matrices have many nice properties.
* They preserve the length of vectors: $\|Ux\| = \|x\|$.  
s is because $\|Ux\|^2 = (Ux)^\dagger Ux = x^\dagger U^\dagger U x = x^\dagger x = \|x\|^2$.
e generally, they preserve the inner product of vectors: $\langle Ux, Uy \rangle = \langle x, y \rangle$.
* The columns of a unitary matrix are orthonormal.
* The rows of a unitary matrix are orthonormal.
* Geometrically, unitary matrices correspond to rotations and reflections: the geometric properties that preserve length.
* The eigenvalues of a unitary matrix have absolute value 1. 



### Examples of Unitary Matrices

* The identity matrix is unitary.
* The matrix 

$$
R(\theta) = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}
$$

for any $\theta$ is unitary: this is a two-dimensional rotation matrix.
* The matrix

$$
F = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$

Consequently, unitary matrices are not useful just useful in quantum mechanics:
they are useful any time we are interested in preserving length.
For instance, the action of rotation and reflections on chemical systems are often encoded using unitary matrices.
In addition to the rotation and reflection matrices above, Wigner-D matrices, which are used to describe the rotation of spherical harmonics, are unitary.


## The Singular Value Decomposition

We have already seen one matrix decomposition: the eigendecomposition.  However, the eigendecomposition does not exist for all matrices.
The Singular Value Decomposition (SVD) is a more general matrix decomposition that does.

Given a matrix $A$, the SVD of $A$ is a factorization of the form

$$
A = U \Sigma V^\dagger
$$

where $U$ and $V$ are unitary matrices, and $\Sigma$ with non-negative real numbers on the diagonal and zeros elsewhere.
The non-zero elements of $\Sigma$ are called the singular values of $A$.  Some things to note:
ver, * The SVD always exists for any matrix.  However, it is NOT unique.
* If $A$ is diagonalizable (i.e., it has an eigendecomposition), then the singular values are the absolute values of the eigenvalues.
* The singular values are the eigenvalues of $A^\dagger A$ (or $AA^\dagger$) raised to the power of 1/2.
* The columns of $U$ are the eigenvectors of $AA^\dagger$, and the columns of $V$ are the eigenvectors of $A^\dagger A$.

### Application of the SVD:

#### Data Compression

One of the most important applications of the SVD is in data compression.  In general, storing an $m \times n$ matrix requires $mn$ elements.
However, in many applications many of the singular values are small.  Consequently, we can neglect them and store only the largest singular values and their corresponding columns of $U$ and $V$.

#### Principal Component Analysis

This concept of 'low-rank approximation' is also the basis of the Principal Component Analysis (PCA) algorithm, which is used to reduce the dimensionality of data. Assume we have a data matrix of size $m \times n$: each row is a different data point, and each column is a different feature.
In PCA, we first 'center' each column (i.e., subtract the mean of each column from each element of the column), so that each column is zero-mean.
Then, we compute the SVD of the data matrix, and then keep only the largest singular values and their corresponding columns in $U$ and $V$.
We can then interpret the columns of $V$ as the principal components of the data: the directions in which the data varies the most.
The column of $U$ then tells us how much each data point varies in each of these directions.
This allows us to build a lower-dimensional representation of the data that captures the most important features.  

Note that since the PCA is the SVD of the data matrix, it is equivalent to diagonalizing the covariance matrix of the data.
Then, the covariance matrix of the data is $C = \frac{1}{n} X^\dagger X$. The eigenvectors of the covariance matrix are the principal components of the data.

$$
C = \frac{1}{n} X^\dagger X = \frac{1}{n} (U \Sigma V^\dagger)^\dagger U \Sigma V^\dagger = \frac{1}{n} V \Sigma U^\dagger U \Sigma V^\dagger = V \frac{1}{n} \Sigma^2 V^\dagger
$$

Consequently, the eigenvectors of the covariance matrix are the same as the columns of $V$ in the SVD of the data matrix.

#### Least Squares Regression

Another application of the SVD is in solving least squares regression problems.

Say we have a matrix $A$ and a vector $b$, and we want to find the vector $x$ that gets as close to solving the equation $Ax = b$ as possible.
The least squares solution is the vector $x$ that minimizes $\|Ax - b\|^2$.  After some mathematics (we will not go into herFe), it can be shown that the least squares solution is $x = V \Sigma^{-1} U^\dagger b$, where $U$, $\Sigma$, and $V$ are the SVD of $A$.  Moreover, we can also control the number of singular values we use in the solution: by damping the singular values, we can make solutions that may have higher error but are less sensitive to noise. 


#### Orthogonalizing Vectors

Another application of the SVD is to orthogonalize a set of vectors.
Say we have a set of vectors $v_1, v_2, \ldots, v_n$, and want to find an orthogonal set of vectors $u_1, u_2, \ldots, u_n$ that span the same space
(i.e., any vector that can be written as a linear combination of the $v_i$ can also be written as a linear combination of the $u_i$).
We can do this by computing the SVD of the matrix whose columns are the $v_i$, and the columns of $U$ will be the $u_i$.
