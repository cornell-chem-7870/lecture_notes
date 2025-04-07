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

# Intro to Numerical Integration

## Learning Objectives

- Discuss some Examples of ODEs in chemistry.
- Connect numerical integration of functions with numerical integration of ODEs.
- Derive the Forward and Backward Euler methods for ODEs.

## What is an ODE

An ODE, or ordinary differential equation, is an equation that relates a function to its derivatives.  Denoting the function as \(y(t)\), an ODE can be expressed in the form:

$$
\frac{dy(t)}{dt} = f(t, y(t))
$$

where \(f\) is a function of time \(t\) and the value of the function at that time, \(y(t)\).
Note that $y$ may be a vector, in
Our goal is to find the function \(y(t)\) that satisfies this equation.
In many cases, we can attempt to solve this equation analytically, by a variety of methods, including separation of variables, integrating factors, and so forth.
But in general, this is not the case.  Then, we need to solve the ODE *numerically*: using a computer, we attempt to construct an approximate solution.


### Examples of ODEs in Chemistry

ODES show up in multiple areas of chemistry.  Two common examples are:

- **Chemical Kinetics**: The rate of change of concentration of reactants and products in a chemical reaction can be described by ODEs. For example, the rate of a first-order reaction can be expressed as:

$$ \frac{d[A]}{dt} = -k[A] $$

where \([A]\) is the concentration of reactant A and \(k\) is the rate constant.
For higher-order reactions, the rate equations can be more complex, involving multiple reactants and products.  For instance, consider the Iodine clock reaction, with rate equations given below.

$$ 
\begin{split}
    \frac{d[I_3^{-}]}{dt} &= k_1[I^{-}]^3 - k_2[I_3^{-}][S_2O_3^{2-}]^2 \\
    \frac{d[I^{-}]}{dt} &= - k_1 [I^{-}]^3 + 3 k_2 [I_3^{-}][S_2O_3^{2-}]^2 \\
    \frac{d[S_2O_3^{2-}]}{dt} &= - k_2[I_3^{-}][S_2O_3^{2-}]^2 S.
\end{split}
$$

- **Hamiltonian Dynamics**: The motion of particles in a system can be described by Hamilton's equations, which are a set of first-order ODEs.  In classical mechanics, the equations of motion for a particle can be expressed as:

$$
\begin{split}
    \frac{dx}{dt} &= \frac{\partial H}{\partial p} \\
    \frac{dp}{dt} &= -\frac{\partial H}{\partial x}
\end{split}
$$

where \(H\) is the Hamiltonian function, \(x\) is the position, and \(p\) is the momentum of the particle.  For example, consider a simple harmonic oscillator with Hamiltonian given by: 

$$ H(x, p) = \frac{p^2}{2m} + \frac{1}{2} k x^2 $$

where \(m\) is the mass of the particle and \(k\) is the spring constant.

## Solving ODEs using Forwards and Backwards Euler 

To solve these equations, we first consider a simpler case: when $f$ is only a function of $t$.

$$
\frac{dy(t)}{dt} = f(t)
$$

Given the value of $y$ at time $0$, we wish to find the value of $y$ at a time $T$ in the future.
A simple way to do this is to divide the time interval from $0$ to $T$ into $n$ equal intervals, each of length $\Delta t = \frac{T}{n}$,
and then approximate the value of $y$ at each time step using a left Riemann sum.

$$
y(T) = y(0) + \sum_{i=0}^{n-1} f(i \Delta t) \Delta t
$$

Note that we can also write this as an iterative equation:

$$
y((i+1) \Delta t) = y(i \Delta t) + f(i \Delta t) \Delta t
$$

Repeatedly substituting this equation gives us the value of $y$ at any time $T$ in the future.

We are going to use this approach to solve ODEs in general.  Now, considering the case where $f$ can also depend on $y$, we can write the equation as:

$$
y((i+1) \Delta t) = y(i \Delta t) + f(i \Delta t, y(i \Delta t)) \Delta t
$$

This is known as the Forward Euler method.  The algorithm  proceeds according to the following pseudocode:

1. Calculate the time step $\Delta t = \frac{T}{n}$.
2. Set $y(0) = y_0$.
3. For $i = 0$ to $n-1$:
    - Use the equation above to calculate $y((i+1) \Delta t)$.
4. Return $y(T)$.


### Limitations of Forward Euler
For instance, it can cause the solute to "blow up" if the time step is too large or if we are trying to integrate over very long periods of time.  Consider solving a simple harmonic oscillator using Forward Euler, with $k = m = 1$.  The equation of motion is given by:

$$
\begin{split}
    \frac{dx}{dt} &= v \\
    \frac{dv}{dt} &= - x
\end{split}
$$

We start with $x(0) = 1$ and $v(0) = 0$: initially, our energy is given by $E = \frac{1}{2} 1^2 + \frac{1}{2} 0^2 = \frac{1}{2}$.
Now, let's take a step of $\Delta t = 0.1$ and see what happens to the energy of the system.
Our Forward Euler algorithm  gives us

$$
\begin{split}
    x(0.1) &= x(0) + v(0) \Delta t = 1 + 0 \cdot 0.1 = 1 \\
    v(0.1) &= v(0) - x(0) \Delta t = 0 - 1 \cdot 0.1 = -0.1
\end{split}
$$
Our new energy is given is given by $ E = \frac{1}{2} 1^2 + \frac{1}{2} (-0.1)^2 = \frac{1}{2} + 0.005 = 0.505$.
Now, let's take another step of $\Delta t = 0.1$:

$$
\begin{split}
    x(0.2) &= x(0.1) + v(0.1) \Delta t = 1 - 0.1 \cdot 0.1 = 0.99 \\
    v(0.2) &= v(0.1) - x(0.1) \Delta t = -0.1 - 1 \cdot 0.1 = -0.2
\end{split}
$$
Our new energy is given is given by $ E = \frac{1}{2} (0.99)^2 + \frac{1}{2} (-0.2)^2 = \frac{1}{2} (0.9801) + 0.02 = 0.49005 + 0.02 = 0.51005$.
This is a problem: the energy of the system is increasing, and it will continue to increase until the system blows up.  
Graphically, if we plot the position and velocity of the system as a function of time, we see that the system is oscillating, but the amplitude of the oscillation is increasing and gradually spiralling out.

The speed at which the energy increases with both the time step and the force constant.  This is a specific example of a more general trend:
- Large values of the derivative make our Forward Euler method unstable: we say an ODE with large derivatives is *stiff*.
- Large time steps make our Forward Euler method unstable.

Consequently, if our system is very stiff, we often need to use very small time steps to get a reasonable answer.


### Backward Euler Method

Our unhappiness with Forwards Euler motivates us to look for a different approach.
Fortunately, left Riemann sums are not the only way to approximate integrals.  
<!-- We can also use right Riemann sums, or trapezoidal sums, or Simpson's rule, etc. -->
Let's try a different approach: using the right Riemann sum instead of the left Riemann sum.
If we do this, the iteration becomes

$$
y((i+1) \Delta t) = y(i \Delta t) + f((i+1) \Delta t, y((i+1) \Delta t)) \Delta t
$$

This is known as the Backward Euler method.
Unfortunately, we have paid a price: we need to solve this equation for $y((i+1) \Delta t)$, in each step of the equation.
A simple approach might be to use a fixed-point iteration.  We start with an initial guess for $y((i+1) \Delta t)$, and then repeatedly substitute this value into the right-hand side of the equation until we converge to a solution.
More generally, we might use a root-finding algorithm, such as gradient descent or Newton's method, to find the solution.
(Root finding will be the subject of a future lectures.)
Methods where we need to solve an equation for self-consistency are called *implicit* methods, in contrast to the *explicit* methods such as Forward Euler.

If we apply the Backward Euler method to the simple harmonic oscillator, instead of seeing the energy continuously increase we see it continuously *decrease*
and both $x$ and $v$ will spiral inwards to 0.
This is... better.  At least, the energy is not blowing up.
In fact, backwards Euler is much more stable than forwards Euler: if you need to solve an ODE and are running into issues with it being stiff, Backwards Euler can be a good choice.
The drawback is that we need to solve an equation at each time step, which can require evaluating $f$ multiple times per step and be computationally expensive.

## Summary

We have found two methods for solving ODEs: Forward Euler and Backward Euler.
One is explicit and easy to implement, but can be unstable for stiff equations.
The other is implicit and more stable, but requires solving an equation at each time step.
Neither of them preserve the energy of the system, which is a problem if we are trying to do physical simulations.
Our next goal is to find a method that is both explicit and preserves the system's energy.