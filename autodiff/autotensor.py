# pylint: disable=no-name-in-module
import torch
import numpy as np
import typing


class autoTensor:
    
    def __init__(self, value, dependents=None, requires_grad: bool = False):
        self.value = value.type(torch.FloatTensor)
        self.requires_grad = requires_grad
        self.grad = None

        if dependents is None:
            self.dependencies = []
        else:
            self.dependencies = dependents

    def size(self):
        return f"autoTensor({self.value.size()})"

    def __repr__(self):
        return f"autoTensor({self.value})"

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

    def numpy(self):
        return self.value.cpu().detach().numpy()

    def MatMul(self,other):
        from functional import MatMul 
        return MatMul(self,other)
        
    def mm(self,other):
        from functional import MatMul 
        return MatMul(self,other)

    def matmul(self,other):
        from functional import MatMul 
        return MatMul(self,other)

class Node:    
    def __init__(self, autoVariable, compute_gradient):
        assert type(compute_gradient) == type(self.__init__), "None-Callable generated"

        self.autoVariable = autoVariable
        self.compute_gradient = compute_gradient

    @staticmethod
    def dfs(dependencies,gradient):
        for dependency in dependencies:
            if dependency.autoVariable.requires_grad:
                back_gradient = dependency.compute_gradient(gradient)
                dependency.autoVariable.backprop(back_gradient)

def make_autoTensor(tensor):
    if isinstance(tensor,autoTensor):
        return tensor
    elif isinstance(tensor,torch.Tensor):
        return autoTensor(value=tensor)
    elif isinstance(tensor,np.ndarray):
        return autoTensor(value=torch.Tensor(tensor))



