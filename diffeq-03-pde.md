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

# From ODE to PDE

## Learning Objectives

- Understand the difference between ODEs and PDEs
- Understand the basic structure of a PDE
- Know how to solve a PDE using finite difference methods
- Be able to implement the Crank-Nicolson method for solving PDEs

## What is a PDE?

Partial differential equations generalize ordinary differential equations (ODEs) to functions of multiple variables.

The first part of a PDE is an equation that relating the partial derivatives of a function to each other.
For instance,
- The wave equation describes the propagation of waves in a medium and is given by:

  $$
  \frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u(x, t)
  $$
  
  where \(u\) is the wave function, \(t\) is time, and \(\nabla^2\) is the Laplacian operator.
- The Schrodinger equation describes the evolution of quantum states and is given by:
    
  $$
  i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi(x) + V(x) \psi(x)
  $$
  
  where \(\psi\) is the wave function, \(V\) is the potential energy, and \(m\) is the mass of the particle.

- The Convection-Diffusion equation describes the transport of substances in a medium.  For an incompressible flow,  it is given by:

  $$
  \frac{\partial u}{\partial t} + v \cdot \nabla u = D \nabla^2 u
  $$

  where \(u\) is the concentration of the substance, \(v\) is the velocity field, and \(D\) is the diffusion coefficient.

- The Fokker-Planck equation describes the time evolution of the probability density function of a stochastic process.  It is given by:

  $$
  \frac{\partial P(x, t)}{\partial t} = -\nabla \cdot (v(x) P(x, t)) + D \nabla^2 P(x, t)
  $$
  
  where \(P\) is the probability density function, \(v(x)\) is the drift velocity, and \(D\) is the diffusion coefficient.

While these are time-dependent PDEs, we can also have time-independent PDEs.  (Often, these result from looking for steady states for time-dependent equaitons.)  For instance, the Poisson equation describes the potential field in electrostatics and is given by:

$$
\nabla^2 u(x) = -\frac{\rho(x)}{\epsilon_0}
$$

where \(u\) is the potential, \(\rho(x)\) is the charge density, and \(\epsilon_0\) is the permittivity of free space.


The *second* part of a PDE is the boundary conditions and initial conditions, which specify how the function behaves at spatial and temporal boundaries respectively.
The initial conditions specify the state of the system at time \(t=0\), while the boundary conditions specify the behavior of the system at the spatial boundaries.
These boundary conditions can be of different types.  Two of note are:
- **Dirichlet boundary conditions** specify the value of the function at the boundary.  (The wavefunction going to 0 at the endges of an infinite potential energy wall is an example of this.)
- **Neumann boundary conditions** specify the value of the derivative of the function at the boundary.
In other cases we might specify a combination of the two, or even a more complex condition.

```{note}
A PDE is not complete with out its boundary and initial conditions!  They are essential to solving the PDE. If they change, the solution may change drastically.
```

### Recap on Prior PDE Solvers We've Discussed

We have already written down a PDE solver without discussing it in February!
In a previous homework, we attempted to solve the Schrodinger equation by
1. Introducing a Basis Set expansion of the wavefunction
2. Writing down a matrix representation of the Hamiltonian that gives the time-evolution of the coefficients.
3. Diagonalizing the Hamiltonian to find the eigenvalues and eigenvectors and use them to time-evolve the system.

This is an example of an approach known as a spectral method: we decompose the solution into a linear combination of basis functions and then solve for the coefficients of these basis functions.
Note that it is necessary that the basis set obeys the boundary conditions of the problem.  For instance, if we are solving the Schrodinger equation with Dirichlet boundary conditions, we need to use a basis set that goes to zero at the boundaries.  (The Fourier basis does not do this, but the sine basis does.)

Spectral methods are often very powerful approaches, but they have their limitations.  For instance, it is not clear how to apply this to a PDE with non-linear terms.  (The Schrodinger equation is linear in the wavefunction, but other PDEs,
such as the Kuramoto-Sivashinsky equation $\partial_t u = -\partial_x^2 u - \partial_x^4 u - \partial_x (u^2)$ are not.)
Moreover, the basis set expansion is not always the most efficient way to represent the solution: a bad basis set may converge slowly, or lead to ringing artifacts.
Finally, expanding the size of the basis set can lead to a large dense matrix: increasing computational expense.

## Finite Difference Methods

Here, we discuss a different approach to solving PDEs, known as finite difference methods.
In Finite difference methods, we evaluate the function an a grid of points.
We then approximate the derivatives in the PDE using finite differences,
similar to the approach we took when discussing ODE solvers.
For instance, we can approximate the first derivative of a function \(u(x)\) at point \(x\) using a forward difference:

$$
\frac{\partial u}{\partial x} \approx \frac{u(x + \Delta x) - u(x)}{\Delta x}
$$

a backward difference:

$$
\frac{\partial u}{\partial x} \approx \frac{u(x) - u(x - \Delta x)}{\Delta x}
$$

or a central difference:

$$
\frac{\partial u}{\partial x} \approx \frac{u(x + \Delta x) - u(x - \Delta x)}{2 \Delta x}
$$

Similarly, we can approximate the second derivative of a function \(u(x)\) at point \(x\) using a central difference:

$$
\frac{\partial^2 u}{\partial x^2} \approx \frac{u(x + \Delta x) - 2 u(x) + u(x - \Delta x)}{\Delta x^2}
$$

These operators can be written as matrices as well.  For instance, the second derivative operator can be written as:

$$
D_2 = \frac{1}{\Delta x^2}
\begin{pmatrix}
 \ddots & \ddots &\ddots & \ddots & \ddots & \ddots & \ddots \\
 \ddots & 1      &-2     & 1      & 0      & 0      & \ddots \\
 \ddots & 0      &1      & -2     & 1      & 0      & \ddots \\
 \ddots & 0      &0      & 1      & -2     & 1      & \ddots \\
 \ddots & \ddots &\ddots & \ddots & \ddots & \ddots & \ddots
\end{pmatrix}
$$

(The values in the corners depend on the boundary conditions.)  While this is a large matrix, it is sparse: most of the entries are zero.  This can often lead to much faster computations.

Once we have these finite difference approximations, we can substitute them into the PDE to get a set of ODEs that give the time-evolution of the function at each point in space.
For instance, consider the simple diffusion equation

$$
\frac{\partial u}{\partial t} =  \partial_x^2 u(x, t)
$$

Applying the finite difference approximation to the second derivative, we get

$$
\frac{\partial u_{i}}{\partial t}(t) = \frac{u_{i+1}(t) - 2 u_{i}(t) + u_{i-1}(t)}{\Delta x^2}
$$

where \(u_i\) is the value of the function at point \(i\) in space. 


### The Crank-Nicolson Method

One of the most common finite difference methods is the Crank-Nicolson method.  In Crank-Nicolson, we
1. Use central difference to approximate spacial derivatives in the PDE
2. Integrate the resulting ODEs using the trapezoidal rule.

This is a very stable method, and is commonly used for PDEs that model diffusions or other dissipative processes.
For other PDEs it is not necessarily the most efficient method, but it is a reasonably ``safe'' option.

If we attempt to solve the simple diffusion equation above using the Crank-Nicolson method, we get

$$
\begin{split}
    u_i(t + \Delta t) = u_i(t) + \frac{\Delta t}{2} \left( \frac{u_{i+1}(t) - 2 u_{i}(t) + u_{i-1}(t)}{\Delta x^2} + \frac{u_{i+1}(t + \Delta t) - 2 u_{i}(t + \Delta t) + u_{i-1}(t + \Delta t)}{\Delta x^2} \right)
\end{split}
$$

or using the $D_2$ matrix we defined above:

$$
    u(t + \Delta t) = u(t) + \frac{\Delta t}{2} \left( D_2 u(t) + D_2 u(t + \Delta t) \right)
$$

Note that for this (and for many other PDEs) we can solve for the function at $u(t + \Delta t)$ using linear algebra:

$$
    \left( I - \frac{\Delta t}{2} D_2 \right) u(t + \Delta t) = \left( I + \frac{\Delta t}{2} D_2 \right) u(t)
$$

implying 

$$
    u(t + \Delta t) = \left( I - \frac{\Delta t}{2} D_2 \right)^{-1} \left( I + \frac{\Delta t}{2} D_2 \right) u(t)
$$

(A proof that this matrix is invertible is beyond the scope of this class.)  

```{note}
In practice, you don't actually want to invert the matrix since this very computationally expensive.  Instead, it is better to solve the previous equation,
observing that it takes the form

$$
A x = b
$$

where $A = \left( I - \frac{\Delta t}{2} D_2 \right)$, $x = u(t + \Delta t)$, and $b = \left( I + \frac{\Delta t}{2} D_2 \right) u(t)$,
and you can leverage extensive work put into solving sparse linear algebra equations, e.g. the methods in `scipy.sparse.linalg`.
```