# pylint: disable=no-name-in-module
import torch
import numpy as np

def make_autoTensor(tensor):
    """Ensures that incoming object is an autoTensor,
    otherwise it converts object to tensor if possibe"""

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
    """Ensures that incoming object is an torch.Tensor,
    otherwise it converts object to tensor if possibe"""

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

class autoTensor(object):
    """autoTensor is the basic building block for the automatic diffentiation system"""
    
    def __init__(self, value, channels=None, requires_grad: bool = False):
        self.value = make_torchTensor(value).type(torch.FloatTensor)
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_saved = None

        if channels is None:
            self.channels = []
        else:
            self.channels = channels    

    def size(self):
        return self.value.size()

    def __repr__(self):
        return f"autoTensor(\n{self.value})\n"

    def grad_zeros(self):
        if self.grad != None:
            self.grad.value = self.grad.value * 0

    def backprop(self, gradient):
        """Propagates appropriate gradient to local reverse computational sub-graph"""

        if not self.requires_grad:
            raise RuntimeError("Gradient called on a non-differntiable variable")
        
        if self.grad == None:
            self.grad = make_autoTensor(torch.zeros(self.value.size()))

        if gradient is None:
            if self.size() == torch.rand([]).size():
                gradient = autoTensor(value=torch.ones([]))
            else:
                raise RuntimeError("Gradient not provided")

        self.grad.value = self.grad.value + gradient.value 

        Node.dfs(channels = self.channels, gradient = gradient)
    
    def grad_sweep(self):
        """Clears gradient values to zero in the local reverse computational sub-graph"""
        
        self.grad_zeros()
        Node.dfs_grad(self.channels)

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

    def exp(self):
        return Exp(self)

class Node(object):   
    """Node for a reverse computation graph"""

    def __init__(self, autoVariable, vjp):
        """A node holds a back_channel variable and a vjp"""
        assert type(vjp) == type(self.__init__), "None-Callable generated"

        self.autoVariable = autoVariable
        self.vjp = vjp

    @staticmethod
    def dfs(channels,gradient):
        """This is where the magic happens"""
        for back_channel in channels:
            if back_channel.autoVariable.requires_grad:
                back_gradient = back_channel.vjp(gradient)
                back_channel.autoVariable.backprop(back_gradient)

    @staticmethod
    def dfs_grad(channels):
        for back_channel in channels:
            if back_channel.autoVariable.requires_grad:
                back_channel.autoVariable.grad_sweep()

# Dealing with circular imports
from autodiff.functional import Add, MatMul, Multiply, Negate, Substract, Power, Divide, Sum, Exp
from neural import Weight