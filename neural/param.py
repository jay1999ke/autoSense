from autodiff.autotensor import autoTensor
import torch
import torch.nn.init as torchInit

class Weight(autoTensor):
    """The datastructure that holds parameters of a Model"""

    def __init__(self, shape, channels=None, requires_grad= True,initializer=None):
        if initializer is not None:
            super(Weight,self).__init__(value=initializer.createParam(shape), channels=None, requires_grad=requires_grad)
        else:
            super(Weight,self).__init__(value=Initializer.createParamStatic(shape),channels=None, requires_grad=requires_grad)


    def update_weights(self,learning_rate):
        self.value = self.value - learning_rate*self.grad.value

# TODO:
# Implementation of all kinds of initialization techniques

class Initializer(object):
    """Weights initializer"""
    
    def __init__(self,init_type):
        self.init_type = init_type.lower()

    def createParam(self,shape):
        if self.init_type == "random":
            return torch.rand(shape)
        elif self.init_type == "xavier":
            return torchInit.xavier_uniform_(torch.empty(shape))
        elif self.init_type == "he":
            return torchInit.kaiming_uniform_(torch.empty(shape))

    @staticmethod
    def createParamStatic(shape):
        return torch.rand(shape) * 0.09
