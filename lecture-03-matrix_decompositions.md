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

This matrix has only one eigenvalue, 0, with eigenvectors $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

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

One specific case of interest, however, is when a matrix is *Hermitian*: that is, when $A = A^\dagger$.
A famous theorem in linear algebra, the spectral theorem for Hermitian matrices,
states that all Hermitian matrices are diagonalizable.
Moreover,   it immediately follows that 
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

These two observations form the basis for a lot of quantum mechanics:
* Since the eigenvalues of a Hermitian matrix are real, the observed values of quantum mechanical observables are real.
* Since the eigenvectors of a Hermitian matrix are orthogonal, we do not see transitions between different states in quantum mechanics
when observing a system in a state in which we are in an eigenstate of the observable.

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
* The eigenvalues of a unitary matrix have absolute value 1.  (If you are not familiar with eigenvalues, keep on reading!)



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


## Matrix Decompositions
TBD