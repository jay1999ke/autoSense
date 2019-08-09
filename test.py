import torch
import numpy as np
from autosense.autodiff import autoTensor as tensor
import autosense.autodiff.functional as F

a = tensor(torch.ones(3,3))
a.requires_grad=True

x = F.Transpose(a,1,0)
z = x@a

z.backprop( tensor([[1,2,3],[1,2,3],[1,2,3]]) )

print(a.grad,x.grad)

print("####################################")

a = torch.ones(3,3)
a.requires_grad=True

x = a.transpose(1,0)
z = x@a

z.backward( torch.tensor([[1,2,3],[1,2,3],[1,2,3]]).type(torch.FloatTensor))

print(a.grad,x.grad)