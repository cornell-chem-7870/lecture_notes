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

# Optimization

## Learning Objectives
- Briefly discuss optimization and the utility of the gradient
- Discuss gradient descent and Newton's method.


## An Intro to Optimization

So far we have had many equations where we needed to minimize a quantity.  When we were solving least squares problems, we attempted to find the vector $m$ that minimized the quantity $||Ax - b||^2$.  
When we were attempting to solve ODEs and PDEs, we needed to solve implicit updates such as backward Euler, which required finding $x_{t+ \Delta t}$ that minimized the difference between $f(x_{t+\Delta t})$  and 
$x_t + \Delta t g(x_{t+\Delta t}) $.  In all of these cases so far, we have told you "just call this numpy / scipy method" to solve the problem.
However, for many applications you might want to write your own optimizer, or at least to debug the optimizer you are using.  Consequently, in this section we will drill down on optimizers and how they work.

Say you have a function $f(x)$ that you want to minimize.  The basic recipe for most optimization algorithms are as follows:
1. Move in a direction that you (hope) decreases the value of the function.
2. Do it again.

### Gradient Based Optimization

To get this very subtle and complicated algorithm to work, it helps to have a notion of what directions decrease the function.  For this reason, we focus on *gradient-based optimization* algorithms: algorithms that use the gradient of the function to determine the direction to move in.
Nowadays, it is reasonably easy to access the gradient of functions using automatic differentiation: in the mid-to-late 2010's, there was incredible work put into developing software that can automatically compute the gradient of a function using a procedure known as *automatic differentiation*: examples include PyTorch, Jax, and Autograd.
In brief, these software packages keep track of a sequence of operations and use the chain rule to compute the gradient of the resulting function.

## Gradient Descent

The most basic gradient-based optimization algorithm is *gradient descent*.  The idea is to take a step in the direction of the negative gradient of the function.  This is because the gradient points in the direction of steepest ascent, so moving in the opposite direction should decrease the function value.  The update rule for gradient descent is given by:

$$
x_{t+1} = x_t - \alpha \nabla f(x_t)
$$

where $\alpha$ is a parameter known as the *step size* (also called the *learning rate* in machine learning circles).  Note that we can compare this to solving the following ODE

$$
\frac{dx}{dt} = -\nabla f(x)
$$

using Forward Euler's method and setting the timestep to $\alpha$.  This gives us some intuition for the effect of the step size.  If $\alpha$ is too small, we will take a long time to converge to the minimum.  
If $\alpha$ is too large, our solution may diverge.

For instance, consider minimizing the function $f(x) = x^2 + y^2$, starting at $x=1, y=1$.  The gradient is given by $\nabla f(x) = (2x, 2y)$.  If we set $\alpha = 0.2$, then our first step is given by:

$$
x_{t+1} = (1, 1) - 0.2 (2, 2) = (0.6, 0.6)
$$

and our second step is given by:

$$
x_{t+2} = (0.6, 0.6) - 0.2 (1.2, 1.2) = (0.36, 0.36)
$$

Clearly, we are making progress.  Moreover, our optimization algorithm "slows down" as we get closer to the minimum, so we don't overshoot.  
Now, let's try a larger step size.  If we set $\alpha = 0.4$, then our first two steps are given by

$$
\begin{split}
x_{t+1} =& (1, 1) - 0.4 (2, 2) = (0.2, 0.2) \\
x_{t+2} =& (0.2, 0.2) - 0.4 (0.4, 0.4) = (0.04, 0.04)
\end{split}
$$

We are making better progress!  Let's try a really bigg step size and see what happens and set $\alpha = 2.0$.  Bigger is better, right?

$$
\begin{split}
x_{t+1} =& (1, 1) - 2.0 (2, 2) = (-3, -3) \\
x_{t+2} =& (-3, -3) - 2.0 (-6, -6) = (9, 9)
\end{split}
$$

Now we are diverging: we see an increase, rather than a  decrease, in the function value.  This leads to a piece of practical advice: if you are using a gradient-based optimizer and you don't see steady progress, the first step is to consider dropping your step size.


## Newton's Method

Gradient descent is a first-order method: it only uses the first derivative of the function to determine the direction to move in.  
This can lead to problems, particularly if our problem is badly scaled.
For instance, consider optimizing the function $f(x) = x^2 + 0.001 y^2$ using gradient descent, starting from $x = 1, y = 1$, with $\alpha = 0.2$. The gradient is given by $\nabla f(x) = (2x, 0.002y)$, so our first step is given by:

$$
x_{t+1} = (1, 1) - 0.2 (2, 0.002) = (0.6, 0.9996)
$$

While we are making progress in $x$, we are making very little progress in $y$.  We could attempt to increase the step size in $y$, but this would lead to divergence in $x$. 

There is nothing that fundamentally different from the previous example, we have just scaled our $y$ coordinate in a different way.  Newton's method addresses this deficiency by using the second derivative of the function, as well as its first.
The idea is to use the second derivative to determine the curvature of the function, and then use this information to determine the step size.  The update rule for Newton's method is given by:

$$
x_{t+1} = x_t - \alpha H_f(x_t)^{-1} \nabla f(x_t)
$$

where $\nabla^2 f(x_t)$ is the Hessian of the function at $x_t$.  The Hessian is a matrix that contains all of the second derivatives of the function.  For instance, for a function $f(x,y)$, the Hessian is given by:

$$
H_f(x,y) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{pmatrix}
$$

Considering our previous quadratic example, the Hessian is given by:

$$
H_f(x,y) = \begin{pmatrix}
2 & 0 \\
0 & 0.002
\end{pmatrix}
$$

The inverse of the Hessian is then given by

$$
H_f(x,y)^{-1} = \begin{pmatrix}
\frac{1}{2} & 0 \\
0 & 500
\end{pmatrix}
$$

Setting $\alpha=1$, our0.2update rule is given by
$$
x_{t+1} = \begin{pmatrix}1 \\ 1 \end{pmatrix} - 0.2 \begin{pmatrix}
0.5 & 0 \\
0 & 500
\end{pmatrix} \begin{pmatrix}
2  \\
0.002
\end{pmatrix} = (0.8, 0.8)
$$

This a much more reasonable step size.  In fact, with $\alpha=1$, Newton's method converges in one step for all quadratic functions, no matter the scaling.