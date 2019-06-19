# pylint: disable=no-name-in-module
import torch
import numpy as np

def make_autoTensor(tensor):
    t_type = type(tensor)
    if t_type == int or t_type == float or t_type == bool:
        return autoTensor(value=torch.Tensor([tensor]))
    elif isinstance(tensor,autoTensor):
        return tensor
    elif isinstance(tensor,torch.Tensor):
        return autoTensor(value=tensor)
    elif isinstance(tensor,np.ndarray):
        return autoTensor(value=torch.Tensor(tensor))
    else:
        return autoTensor(value=torch.Tensor(tensor))

def make_torchTensor(tensor):
    t_type = type(tensor)
    if t_type == int or t_type == float or t_type == bool:
        return torch.Tensor([tensor])
    elif isinstance(tensor,autoTensor):
        return tensor.value
    elif isinstance(tensor,torch.Tensor):
        return tensor
    elif isinstance(tensor,np.ndarray):
        return torch.Tensor(tensor)
    else:
        return torch.Tensor(tensor)

class autoTensor:
    
    def __init__(self, value, dependents=None, requires_grad: bool = False):
        self.value = make_torchTensor(value).type(torch.FloatTensor)
        self.requires_grad = requires_grad
        self.grad = None

        if dependents is None:
            self.dependencies = []
        else:
            self.dependencies = dependents
    

    def size(self):
        return f"autoTensor({self.value.size()})"

    def __repr__(self):
        return f"autoTensor(\n{self.value})\n"

    def grad_zeros(self):
        if self.grad != None:
            self.grad.value = self.grad.value * 0

    def backprop(self, gradient):
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if self.grad == None:
            self.grad = make_autoTensor(torch.zeros(self.value.size()))
        if gradient is None:
            if self.size() == torch.rand([]).size():
                gradient = autoTensor(value=torch.ones([]))
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.value = self.grad.value + gradient.value  # type: ignore

        Node.dfs(dependencies = self.dependencies, gradient = gradient)
    
    def grad_sweep(self):
        self.grad_zeros()
        Node.dfs_grad(self.dependencies)

    def numpy(self):
        return self.value.cpu().detach().numpy()

    def MatMul(self,other):
        return MatMul(self,make_autoTensor(other))
        
    def mm(self,other):
        return MatMul(self,make_autoTensor(other))

    def __matmul__(self, other):
        return MatMul(self,make_autoTensor(other))

    def add(self,other):
        return Add(self,make_autoTensor(other))

    def Add(self,other):
        return Add(self,make_autoTensor(other))

    def __add__(self, other):
        return Add(self,make_autoTensor(other))

    def __radd__(self, other):
        return Add(self,make_autoTensor(other))

    def __iadd__(self, other):
        self.value = self.value + make_autoTensor(other).value
        return self

    def __sub__(self, other):
        return Substract(self,make_autoTensor(other))

    def __rsub__(self, other):
        return Substract(make_autoTensor(other),self)

    def __isub__(self, other):
        self.value = self.value - make_autoTensor(other).value
        return self

    def __neg__(self):
        return Negate(self)

    def sub(self,other):
        return Substract(self,make_autoTensor(other))
    
    def Substract(self,other):
        return Substract(self,make_autoTensor(other))

    def Multiply(self,other):
        return Multiply(self,make_autoTensor(other))
    
    def mul(self,other):
        return Multiply(self,make_autoTensor(other))

    def __imul__(self, other):
        self.value = self.value * make_autoTensor(other).value
        return self

    def __mul__(self, other):
        return Multiply(self,make_autoTensor(other))

    def __rmul__(self, other):
        return Multiply(self,make_autoTensor(other))

    def __pow__(self,power):
        return Power(self,make_autoTensor(power))

    def __truediv__(self,other):
        return Divide(self,make_autoTensor(other))

    def __rtruediv__(self,other):
        return Divide(make_autoTensor(other),self)

    def __itruediv__(self,other):
        self.value = self.value / make_autoTensor(other).value
        return self

    def sum(self,axis=0):
        return Sum(self,axis)

class Node:   
    """Node for a reverse computation graph"""

    def __init__(self, autoVariable, vjp):
        """A node holds a dependent variable and a vjp"""
        assert type(vjp) == type(self.__init__), "None-Callable generated"

        self.autoVariable = autoVariable
        self.vjp = vjp

    @staticmethod
    def dfs(dependencies,gradient):
        """This is where the magic happens"""
        for dependency in dependencies:
            if dependency.autoVariable.requires_grad:
                back_gradient = dependency.vjp(gradient)
                dependency.autoVariable.backprop(back_gradient)

    @staticmethod
    def dfs_grad(dependencies):
        """This is where the magic happens"""
        for dependency in dependencies:
            if dependency.autoVariable.requires_grad:
                dependency.autoVariable.grad_sweep()

# Dealing with circular imports
from autodiff.functional import Add, MatMul, Multiply, Negate, Substract, Power, Divide, Sum