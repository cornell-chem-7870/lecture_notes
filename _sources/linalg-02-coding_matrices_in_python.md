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

# Linear Algebra in Python with Numpy

## Learning Objectives

- Be able to use the `numpy` library to create and manipulate matrices.
- Using `numpy.linalg` to perform more advanced matrix operations.
- Understand the difference between element-wise operations and matrix operations.

## Introduction to  Numpy

Numpy is a python library for numerical computing. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
The core data structure in numpy is the `ndarray`, which is a multidimensional array of elements of the same type. To initialize a numpy array, you can use the `np.array()` function.

```python
import numpy as np

# Create a 1D numpy array
a = np.array([1, 2, 3, 4, 5])
print(a)

# Create a 2D numpy array
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
```

1D numpy arrays are treated vectors, with little distinction between row and column vectors. 2D numpy arrays are treated as matrices, with rows and columns.
(One can make an explicit row or column vector by thinking of it as a one-row or one-column matrix,
and making a 2D array that is shape `(1, n)` or `(n, 1)` respectively.)
A 3D numpy array can be thought of as a collection of matrices, or a 3-tensor.

Numpy also has support for arrays of complex numbers: in python, complex numbers are represented as `j` rather than `i`.

```python
import numpy as np

# Create a 1D numpy array of complex numbers
a = np.array([1 + 2j, 3 + 4j, 5 + 6j])
print(a)
```
 

### Accessing Elements of a Numpy Array

You can access elements of a numpy array using indexing. Numpy arrays are zero-indexed, meaning the first element has an index of 0.
Subsequent elements have indices 1, 2, 3, and so on. You can also use negative indices to access elements from the end of the array.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# Access the first element
print(a[0])

# Access the last element
print(a[-1])
```

You can also access ranges of elements using slicing. Slicing allows you to access a subset of elements from the array by specifying a start index, end index, and step size.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# Access the first three elements
print(a[:3]) 

# Access the second, third, and fourth elements
print(a[1:4])
```
The indexing is *inclusive* on the left and *exclusive* on the right: `a[start:end]` will return elements `start`, `start+1`, ..., `end-1`.
One can also specify a step size, which is the number of elements to skip between each element.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Access every other element
print(a[::2])

# Access elements in reverse order
print(a[::-1])

# Access every second element between the 3rd and 8th elements
print(a[2:8:2])
```

#### Accessing Elements of a 2D Numpy Array

For 2D numpy arrays, you can access elements using two indices: the row index and the column index. The row index comes first, followed by the column index.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access the element in the first row and first column
print(a[0, 0])

# Access the element in the second row and third column
print(a[1, 2])
```

You can also use slicing to access ranges of elements in a 2D numpy array.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access the first row
print(a[0, :])

# Access the second column
print(a[:, 1])

# Access a submatrix
print(a[:2, :2])
```
Note that here a colon `:` is used to indicate that all elements along that axis should be included.

### Basic Operations on Numpy Arrays

Most operations on numpy arrays are element-wise. This means that the operation is applied to each element of the array individually.

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])

# Add 1 to each element
b = a + 1
print(b)

# Square each element: exponentiation is double asterisk NOT caret
c = a**2
print(c)
```

Many additional mathematical operations are available in numpy, such as `np.sin()`, `np.cos()`, `np.exp()`, and `np.log()`.
Again, these operations are applied element-wise to the array.

```python
import numpy as np
a = np.array([[1, 2], [3, 4]])

# Compute the sine of each element
b = np.sin(np.pi * a)
print(b)
```
In general, element-wise operations such as adding, subtracting, etc, typically require that the two arrays have the same shape.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

# Add the two arrays element-wise
c = a + b
print(c)
```
However, there is one exception: numpy will tile along dimensions of size 1 to make the shapes match.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3
b = np.array([[7, 8, 9]]) # 1x3

# Add the two arrays element-wise
c = a + b
print(c) # b is added to each row of a

d = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
e = np.array([[7], [8], [9]]) # 3x1

# Add the two arrays element-wise
f = d + e
print(f) # e is added to each column of d
```
Moreover, if one of the arrays has less dimensions than the other, numpy will add dimensions of size 1 to the smaller array to try and make the shapes match.

```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]]) # 2x3
b = np.array([7, 8, 9]) # this time a 1D array of length 3

# Add the two arrays element-wise
c = a + b
print(c) # b is added to each row of a
```
Note that these element-wise operations could also be performed manually by writing loops that iterate over the elements of the array.
However, using numpy's built-in functions is generally much faster and more efficient.

## Linear Algebra with Numpy

Importantly, the multiplication operator `*` performs element-wise multiplication, *not* matrix multiplication.
Matrix multiplication is performed using the `@` operator.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Element-wise multiplication
c = a * b
print(c)

# Matrix multiplication
d = a @ b
print(d)
```
Numpy also provides a function `np.dot()` to perform matrix multiplication and matrix-vector multiplication.
```python
f = np.dot(a, b)
print(f)
```
In most cases `@` and `np.dot()` will give the same result, but there are some edge cases where they differ.
In general, it is recommended to use `@` for matrix multiplication.  However, always check on a small example to make sure you understand the behavior!

Matrix-vector products can also be performed using `@` or `np.dot()`.
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])

# Matrix-vector multiplication: b is interpreted as a column vector
c = a @ b
print(c)

# Matrix-vector multiplication: b is interpreted as a row vector
d = b @ a
```
If one makes an explicit column vector as an Nx1 matrix, then former multiplication will work as expected,
but the latter will raise an error.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5], [6]]) 

c = a @ b
print(c)

# d = b @ a  # This will raise an error
d = b.T @ a  # This will work as we are multiplying a row vector by a matrix
print(d)
```

Numpy also provides some attributes for operations on arrays, such as the ability to take
the transpose of a matrix can be computed using `.T`.
```python
import numpy as np

a = np.array
([[1, 2, 3],
  [4, 5, 6]])

# Transpose the matrix
b = a.T
print(b)
```



Numpy provides a module called `numpy.linalg` that contains a variety of linear algebra functions. This module can be used to perform operations such as matrix inversion, determinant calculation, eigenvalue computation, and more.

For instance, to take the inverse of a matrix, you can use the `np.linalg.inv()` function.

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])

# Compute the inverse of the matrix
b = np.linalg.inv(a)
print(b)
```

Numpy also has the matrix equivalent of many elementwise functions: if you want to take the matrix exponential instead of an elementwise exponential, you can use `np.linalg.expm()`, for instance.
In general, spending some time exploring the `numpy.linalg` module can be very helpful for understanding the capabilities of numpy,
and the [corresponding documentation](https://numpy.org/doc/2.1/reference/routines.linalg.html) is a great use of time.