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

# Intro to JaX

## Working in JaX

JaX is a library that provides automatic differentiation and GPU/TPU support. It is designed to be used with NumPy-like arrays and functions.  In fact, part of JaX is a drop-in replacement for NumPy.  Instead of importing NumPy, you import jax.numpy.

```{code-cell} python3
import numpy as np
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])

c = a + b
print(c)
print(c.shape)
print(type(c))
```

## Gradients in JaX

JaX adds additional functionality to NumPy.  For example, JaX can automatically differentiate functions that are defined using JaX operations.  This is done using the `grad` function, which takes a function as input and returns a new function that computes the gradient of the input function.

```{code-cell} python3
from jax import grad

def f(x):
    return x ** 2 + 3 * x + 5

x = jnp.array(2.0)

df = grad(f)  # df is a function that computes the gradient of f

print(df(x))  # prints the gradient of f at x = 2.0
```

The `value_and_grad` function is a convenience function that computes both the value and the gradient of a function at a given point.  

```{code-cell} python3
from jax import value_and_grad
def f(x):
    return x ** 2 + 3 * x + 5
x = jnp.array(2.0)
value, grad_value = value_and_grad(f)(x)  # value is f(x), grad_value is df(x)
print(value)  # prints the value of f at x = 2.0
print(grad_value)  # prints the gradient of f at x = 2.0
```

You can specify which variables to differentiate with respect to by using the `argnums` argument.  By default, `argnums` is set to 0, which means that the first argument of the function is differentiated.  You can set `argnums` to a tuple of integers to differentiate with respect to multiple arguments.

```{code-cell} python3
from jax import grad
def f(x, y):
    return x ** 2 + y ** 2 + 3 * x * y
x = jnp.array(2.0)
y = jnp.array(3.0)
df_dx = grad(f, argnums=0)  # df_dx is a function that computes the gradient of f with respect to x
df_dy = grad(f, argnums=1)  # df_dy is a function that computes the gradient of f with respect to y
print(df_dx(x, y))  # prints the gradient of f with respect to x at x = 2.0, y = 3.0
print(df_dy(x, y))  # prints the gradient of f with respect to y at x = 2.0, y = 3.0
```


The `grad` function can also be used to compute higher-order derivatives.  For example, the second derivative of a function can be computed by taking the gradient of the gradient.

```{code-cell} python3  
d2f = grad(grad(f))  # d2f is a function that computes the second derivative of f
print(d2f(x))  # prints the second derivative of f at x = 2.0
```

JaX also has functions that calculate more complicated derivatives, such as Jacobians and Hessians.  For intance, the Hessian of a function (second-derivative matrix) can be computed using the `hessian` function.

```{code-cell} python3
from jax import hessian

def f(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

x = jnp.array([1.0, 2.0, 3.0])
hess = hessian(f)  # hess is a function that computes the Hessian of f
print(hess(x))  # prints the Hessian of f at x = [1.0, 2.0, 3.0]
```

For multivalued functions, JaX provides the `jacfwd` and `jacrev` functions to compute the Jacobian matrix.  The `jacfwd` function computes the Jacobian using forward-mode differentiation, while the `jacrev` function computes the Jacobian using reverse-mode differentiation.  
The difference between these methods is somewhat beyond our scope.  
However, that forward-mode differentiation is more efficient for functions with a small number of inputs and a large number of outputs, while reverse-mode differentiation is more efficient for functions with a large number of inputs and a small number of outputs.

```{code-cell} python3
from jax import jacfwd, jacrev

def f(x):
    return jnp.array([x[0] ** 2 + x[1], x[1] ** 2 + x[0]])

x = jnp.array([1.0, 2.0])
jac = jacfwd(f)  # jac is a function that computes the Jacobian of f using forward-mode differentiation
print(jac(x))  # prints the Jacobian of f at x = [1.0, 2.0]
jac = jacrev(f)  # jac is a function that computes the Jacobian of f using reverse-mode differentiation
print(jac(x))  # prints the Jacobian of f at x = [1.0, 2.0]
```

### Best Practices for Gradients in JaX

1. Use JaX operations instead of NumPy operations whenever possible.  While you can call NumPy operations on JaX arrays, this breaks the autograd functionality.   
2. Keep functions ``pure''.  This means that the functions should not have side effects, such as modifying global variables or printing to the console.  

## Speeding up JaX with JIT and vmap 

JaX also provides a Just-In-Time (JIT) compilation feature that can significantly speed up the execution of functions.  By default, python code is "interpreted".  This means that the code is executed line by line.  JIT compilation, on the other hand, introspects the code and converts into a set of instructions that can be executed more efficiently.  (If you are familiar with C, C++, or Fortran, these languages compile everything.)  To use JIT compilation, you can use the `jit` function, which takes a function as input and returns a new function that is JIT-compiled.  The first time the JIT-compiled function is called, it will take longer to execute, but subsequent calls will be much faster.

```{code-cell} python3
from jax import jit

def f(x):
    return x ** 2 + 3 * x + 5

x = jnp.array(2.0)
jit_f = jit(f)  # jit_f is a JIT-compiled version of f
print(jit_f(x))  # prints the value of f at x = 2.0
```

```{note}
JIT compilation is not always faster.  For small functions, the overhead of JIT compilation may outweigh the benefits.  However, for large functions or functions that are called many times, JIT compilation can provide a significant speedup.
```

If you want to define a function that is immediately JIT-compiled, you can use the `@jit` decorator.  This is a convenient way to JIT-compile a function without having to call the `jit` function explicitly.

```{code-cell} python3
from jax import jit

@jit
def f(x):
    return x ** 2 + 3 * x + 5

x = jnp.array(2.0)
print(f(x))  # prints the value of f at x = 2.0
```

When used correctly, JIT compilation can provide a significant speedup.  However, not every function can be JIT compiled.  Functions whose behavior is not well-defined or that have side effects cannot be JIT compiled.  This includes functions that use random number generation, functions that affect global variables, functions that initialize arrays of size that cannot be determined at compile time, and many other examples.   
Much of the difficulties with JIT compilations are why JaX has a reputation of being very complicated.  However, if you are careful, you can use JIT compilation to speed up your code significantly.

### Vmap

JaX also provides a `vmap` function that allows you to vectorize functions: take a function that operates on a single input and convert it into a function that operates on a batch of inputs.  When done correctly, vmap can help broadcast operations over multiple inputs without the need for slow `for` loops in python, or complicated numpy array indexing tricks.

```{code-cell} python3
from jax import vmap

def f(x):
    return x ** 2 + 3 * x + 5

x = jnp.array([1.0, 2.0, 3.0])
vmap_f = vmap(f)  # vmap_f is a vectorized version of f

print(vmap_f(x))  # prints the value of f at x = [1.0, 2.0, 3.0]
```

```{note}
vmap is not the same as JIT compilation.  JIT compilation is used to speed up the execution of a single function, while vmap is used to vectorize a function so that it can operate on a batch of inputs.  However, you can use both JIT compilation and vmap together to get the best of both worlds.
```

To specify which arguments to vectorize over, you can use the `in_axes` argument.  By default, `in_axes` is set to 0, which means that the first argument of the function is vectorized.  You can set `in_axes` to a tuple of integers to vectorize over multiple arguments.

```{code-cell} python3
from jax import vmap

def f(x, y):
    return x ** 2 + y ** 2 + 3 * x * y

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([4.0, 5.0, 6.0])

vmap_f = vmap(f, in_axes=(0, 0))  # vmap_f is a vectorized version of f
print(vmap_f(x, y))  # prints the value of f at x = [1.0, 2.0, 3.0] and y = [4.0, 5.0, 6.0]
```

## Further Reading

If you are interested in learning more about JaX, I recommend the official JaX documentation, which is available [here](https://docs.jax.dev/en/latest/).
