"""this file is to be scapped later"""

import torch # emit
from autotensor import autoTensor, autodiffNode
import numpy as np
from functional import MatMul


# t1 is (3, 2)
t1=    autoTensor(value = torch.Tensor( [[1, 2], [3, 4], [5, 6]] ),requires_grad=True)

# t2 is a (2, 1)
t2 = autoTensor(value=torch.Tensor([[10], [20]]),requires_grad=True)
t3 = MatMul(t1,t2)

print(t3)

grad = autoTensor(torch.Tensor([[-1], [-2], [-3]]))
t3.backward(grad)

print(t1.grad.value,"\n")
print( torch.mm(grad.value,t2.value.transpose(1,0)),"\n")

print(t2.grad.value,"\n")
print( torch.mm(t1.value.transpose(1,0),grad.value),"\n" )