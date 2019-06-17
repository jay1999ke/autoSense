# autoDiff

## A minimal reverse accumulation mode differentiation library

### Basic concept

We introduce autoTensor, a boxed pytorch tensor capable of backpropagation using automatic differntiation.

### Working documentation

Every function returns an autoTensor that has a backpropagation channel to compute the gradients with respect to dependent autoTensor.

#### Example

![Example](https://github.com/jay1999ke/autodiff/raw/master/autodiff.jpeg)

Consider a function F3,

F3 = (X @ Y) * (Y + Z)


