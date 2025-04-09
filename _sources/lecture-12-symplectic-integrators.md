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

# More on Numerical ODEs

## Learning Objectives

- Introduce time-symmetric integrators that preserve the energy.
- Derive the equations for the leapfrog integrator, as commonly used in molecular dynamics.
- Discuss the concept of a ``symplectic integrator``.
- Analyse the error in an ODE integrator.

## Time-symmetric integrators

So far, we have introduced two approaches to solving a generic ODE

$$
\frac{dy(t)}{dt} = f(t, y(t))
$$

where \(f\) is a function of time \(t\) and the value of the function at that time, \(y(t)\).  We have also discussed how to solve this equation numerically, using the forward and backward Euler methods, given respectively by

$$
\begin{split}
    y(t + \Delta t) &= y(t) + \Delta t f(t, y(t)) \\
    y(t + \Delta t) &= y(t) + \Delta t f(t+ \Delta t, y(t + \Delta t ))
\end{split}
$$

Unfortunately, we are not quite happy with either solution.  Applying these to a simple harmonic oscillator, we saw that the energy either increases to infinity (forward Euler) or decreases to zero (backward Euler).  
To address this, we will try to introduce integrators that obey the symmetries of a physical system.
One classic one is time-reversal symmetry: if we run the dynamics up to time $T$, reverse the arrow of time, and then run the dynamics back to time $0$, we should end up where we started.  
For inspiration, we again two turn Riemann integration, and find two integrators that obey this symmetry.  The first is the trapezoidal rule: numerically integrating a function using trapezoid rule gives the same answer if we march left or right.
If we generalize this to the case of ODEs, we can write down a time-symmetric integrator as follows:

$$
\begin{split}
    y(t + \Delta t) &= y(t) + \frac{\Delta t}{2} \left( f(t, y(t)) + f(t + \Delta t, y(t + \Delta t)) \right) \\
    &= y(t) + \frac{\Delta t}{2} \left( f(t, y(t)) + f(t + \Delta t, y(t)) \right)
\end{split}     
$$

A quick calculation shows that this is symmetric: integrating backwards in time from $y(t + \Delta t)$  puts us back at $y(t)$.
Note that this is an implicit integrator: we need to know the value of $y(t + \Delta t)$ in order to evaluate the function at that point.  
As with backward Euler, we will need to find a way to solve this equation.  However, the stability and accuracy of this integrator makes this often well worth it.


The second time-symmetric integrator is inspired by the midpoint rule.  To write this, we actually take a step from two steps back, $y(t - \Delta t)$, to $y(t + \Delta t)$, and then evaluate the function at the midpoint, $y(t)$:

$$
    y(t + \Delta t) = y(t ) + \Delta t f(t + \Delta t / 2, y(t + \Delta t / 2)) 
$$

This leaves us with a problem: how do we evaluate the function at the midpoint?  A simple way is we use the same update for the midpoints as well: we combine this update with an update on the midpoint steps:

$$
    y(t + \Delta t / 2) = y(t - \Delta t / 2) + \Delta t f(t , y(t )) 
$$

(Equivalently, we could have doubled our timestep to $2 \Delta t$.)
Unlike the trapezoidal rule, this is an explicit integrator: we can evaluate the function at the midpoint without needing to know the value of $y(t + \Delta t)$.

## The Leapfrog Integrator

This integrator based on the midpoint rule is particularly convenient when working with Newtonian mechanics.  In this case, we can write the equations of motion as

$$
\begin{split}
    \frac{dx}{dt} &= v \\
    \frac{dv}{dt} &= \frac{1}{M} F(t, x)
\end{split}
$$

where $F$ is the force.  (Recall that for Hamiltonian systems, the force is given by $F = - \partial U / \partial x$.)  We can then write the leapfrog integrator as follows:

$$
\begin{split}
x(t + \Delta t) = x(t) + \Delta t v(t + \Delta t / 2)
v(t + \Delta t / 2) = v(t - \Delta t / 2) + \frac{\Delta t}{M} F(t, x(t))
\end{split}
$$

Note that we never need to evaluate the force at the midpoint, $F(t + \Delta t / 2, x(t + \Delta t / 2))$.  We still need to take only a single evaluation of the force, as we would if we were to use forwards or backwards Euler.
However, this integrator is time-symmetric: we have achieved an important physical symmetry.

### Symplectic Integrators

In fact, both the the trapezoidal rule and the leapfrog integrators obey a special property when applied to Hamiltonian dynamics (i.e., where the force is given by $F = - \partial U / \partial x$).  
This property is symplecticity.  While an in-depth treatment of symplecticity is beyond the scope of this course, in words symplecity means that phase space volume is preserved as the dynamics progresses.
Symplecticity is true for the Hamiltonian equations of motion, and is deeply related to the conservation of energy and angular momentum.  
The trapezoidal rule and leapfrog integrators are both symplectic integrators.  As such, they preserve an invariant that is close to, but not exactly, the system's energy.

## Evaluating error in ODE integrators.

So how big of an error do these integrators make?  One way to evaluate this is the error in integrating from time $0$ to time $T$.  As a simple example, let's consider the forwards Euler integrator, reproduced below for your converience.

$$
    y(t + \Delta t) = y(t) + \Delta t f(t, y(t)) 
$$

We will consider the simplified case where $f$ is only a function of $y$, and not $t$.  

We first ask, "how big is the error in the first step?"  To do this, we denote the *true* solution at time $t + \Delta t$ as $y^*(t + \Delta t)$.  Taylor expanding around $y(t)$, we have

$$
y^*(t + \Delta t) = y(t) + \Delta t f(y(t)) + \frac{1}{2} \Delta t^2 \frac{\partial f}{\partial y} f(y(t))  + O(\Delta t^3)
$$

where we have used the fact that $dy/dt = f(y)$ in the chain rule.
Subtracting this from the forward Euler prediction, we find that the error in the first step is given by

$$
\begin{split}
    \epsilon_1 &= y^*(t + \Delta t) - y(t + \Delta t) \\
    &= \frac{1}{2} \Delta t^2 \frac{\partial f}{\partial y} f(y(t)) + O(\Delta t^3)
\end{split}
$$

implyiing that the error scales as $\Delta t^2$.
Now, to go from $0$ to $T$, we need to take $N = T / \Delta t$ steps. If at each step the error we accumulate scalers as $\Delta t^2$, we expect our total error to be $O(\Delta t)$.
We consequently say that the forward Euler integrator is accurate to "first order" in $\Delta t$.
By very similar logic, we can show that the backward Euler integrator is also first order in $\Delta t$.

The trapezoidal rule, on the other hand, is *second order* in $\Delta t$.  
For simplicity, we consider the case where $f$ is only a function of $y$, and not $t$.  

We first expand the value of $f(y(t + \Delta t))$ around $y(t)$:

$$
f(y(t + \Delta t)) =
f(t, y(t)) + \Delta t \frac{\partial f}{\partial t} (y(t)) f(y(t))  + O(\Delta t^2) 
$$

Substituting this into the trapezoidal rule, we find that the trapezoidal rule prediction obeys

$$
\begin{split}
y(t) =& y(t) + \frac{\Delta t}{2}  f(t, y(t))   + \frac{\Delta t}{2} \left( f(t, y(t)) + \Delta t \frac{\partial f}{\partial t} (y(t)) f(y(t))  + O(\Delta t^2) \right) + O(\Delta t^3) \\
 =& y(t) + \Delta t f(t, y(t)) + \frac{1}{2} \Delta t^2 \frac{\partial f}{\partial t} (y(t)) f(y(t))  + O(\Delta t^3)
\end{split}
$$

This is, to *third* order, the same as the true solution. Consequently, we expect the total error from integrating from $0$ to $T$ to be $O(\Delta t^2)$, and we say that the trapezoidal rule is a *second order* integrator.
The leapfrog integrator is also a second order integrator.  What this means is that, in general, we expect the leapfrog and trapezoidal rule to be more accurate than the forward and backward Euler integrators.



## References

- The best reference on integrators that I know of for applications to chemical systems is **Molecular Dynamics** by Leimkuhler and Matthews. 
