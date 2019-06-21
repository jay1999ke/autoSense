from autodiff.autotensor import autoTensor, Node
import torch

#for implemantation of all types of losess

class Loss(autoTensor):
    def __init__(self,value, channels=None, requires_grad = True):
        super(Loss,self).__init__(value=value, channels=None, requires_grad=requires_grad)

    def backward(self):
        gradient = autoTensor(torch.ones(self.value.size()))
        self.backprop(gradient)

class SquareError(Loss):
    """Produces Square error loss of Model"""

    def __init__(self,y_pred,y_target):
        super(SquareError,self).__init__(value=0.5*((y_pred.value-y_target.value)**2))
        self.y_pred = y_pred
        self.y_target = y_target

        back_channel = Node(autoVariable=y_pred,vjp=self.der)
        self.channels.append(back_channel)

    def der(self,gradient):
        value = self.y_pred.value - self.y_target.value
        return autoTensor(value=value)

    def __repr__(self):
        value = torch.sum(self.value)/self.value.size()[0]
        return f"Cost : autoTensor({value})"



# TODO: 
# implemantation of all types of losess
