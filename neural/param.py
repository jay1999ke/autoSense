from autodiff import autoTensor
import torch

class Weight(autoTensor):
    """The datastructure that holds parameters of a Model"""

    def __init__(self, value, channels=None, requires_grad: bool = False):
        super(Weight,self).__init__(value=value, channels=None, requires_grad=requires_grad)

    def update_weights(self,learning_rate):
        self.value = self.value - learning_rate*self.grad.value

# TODO:
# Implementation of all kinds of initialization techniques

class Initializer(object):
    pass