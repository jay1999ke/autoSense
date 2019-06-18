"""this file is to be scapped later"""

import torch
from autodiff import autoTensor, MatMul, Add, Multiply
import numpy as np


"""
# t1 is (3, 2)
t1=    autoTensor(value = torch.Tensor( [[1, 2], [3, 4], [5, 6]] ),requires_grad=True)

# t2 is a (2, 1)
t2 = autoTensor(value=torch.Tensor([[10], [20]]),requires_grad=False)
t3 = MatMul(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[-1], [-2], [-3]]))
t3.backprop(grad)

print(t1.grad,"\n")
print( torch.mm(grad.value,t2.value.transpose(1,0)),"\n")

print(t2.grad,"\n")
print( torch.mm(t1.value.transpose(1,0),grad.value),"\n" )
""

t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = Add(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
""" 
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = Multiply(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
