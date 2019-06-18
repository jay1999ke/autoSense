# autoDiff

## A minimal reverse accumulation mode differentiation library

### Basic concept

We introduce autoTensor, a boxed pytorch tensor capable of backpropagation using automatic differntiation.

### Working documentation

Every function is an autoTensor that has a backpropagation channel to compute the gradients with respect to dependent autoTensor.

#### Example

![Example](https://github.com/jay1999ke/autodiff/raw/master/autodiff.jpeg)

Consider a function F3,

F3 = (X @ Y) * (Y + Z)

X, Y autoTensors require gradients whereas Z does not require gradient.

F3 can be composed of primitive functions as below:

F1 = X @ Y
F2 = Y + Z
F3 = F1 * F2

If a autoTensor, that requires a gradient, is used to compose a child autoTensor then the child autoTensor also requires a gradient. In our example, F1, F2 and F3 all requires gradients by this property. For all functions that are composed of autoTensors that require gradients, a vjp is assigned to the reverse computational graph node. This vjp with the gradient of child autoTensor is used to obtain the gradient of a given function with respect to parent autoTensor.


