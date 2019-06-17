# autoDiff

## A minimal reverse accumulation mode differentiation library

### Basic concept

We introduce autoTensor, a boxed pytorch tensor capable of backpropagation using automatic differntiation.

### Working documentation

Every function returns an autoTensor that has a backpropagation channel to compute the gradients with respect to dependent autoTensor.

#### Example

![github-small](autofiff.jpeg)

Consider a function F3,

F3 = (X @ Y) * (Y + Z)


