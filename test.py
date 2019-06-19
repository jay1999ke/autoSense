"""this file is to be scapped later"""

import torch
from autodiff import autoTensor, MatMul, Add, Multiply
import numpy as np


"""
# matmul test
t1=    autoTensor(value = torch.Tensor( [[1, 2], [3, 4], [5, 6]] ),requires_grad=True)

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
#add test
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = Add(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
""
#mul test
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = Multiply(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
""
# sub tet
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = t1-t2

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
""
#power test
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([2, 2, 2]), requires_grad = True)               # (1, 3)

t3 = t1**t2

print(t3)

grad = autoTensor(torch.Tensor([[1, 2, 1], [1, 1, 2]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
""
#div test
t1 = autoTensor(torch.Tensor([[1, 2, 3], [4, 5, 6]]), requires_grad = True)    # (2, 3)
t2 = autoTensor(torch.Tensor([7, 8, 9]), requires_grad = True)               # (1, 3)

t3 = t1/t2

print(t3)

grad = autoTensor(torch.Tensor([[1, 1, 1], [1, 1, 1]]))

t3.backprop(grad)

print(t1.grad)

print(t2.grad)
"""
# sum test

t1 = autoTensor(torch.Tensor([[1, 2, 3],[1, 2, 3]]), requires_grad=True)
t2 = t1.sum(axis=1)
print(t2)

grad = autoTensor(torch.Tensor([3]))
t2.backprop(grad)

print(t1.grad)