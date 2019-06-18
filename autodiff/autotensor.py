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

    def backward(self, gradient):
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if self.grad == None:
            self.grad = autoTensor(value=torch.zeros(self.value.size()))
        if gradient is None:
            if self.size() == torch.rand([]).size():
                gradient = autoTensor(torch.rand([]))
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")
        self.grad.value = self.grad.value + gradient.value  # type: ignore

        for dependency in self.dependencies:
            backward_grad = dependency.compute_gradient(gradient)
            dependency.autoVariable.backward(backward_grad)

class autodiffNode:    
    def __init__(self, autoVariable, compute_gradient):
        assert type(compute_gradient) == type(self.__init__), "None-callable function generated"

        self.autoVariable = autoVariable
        self.compute_gradient = compute_gradient
