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


class MeanAbsoluteError(Loss):
    """Produces MAE loss of Model"""

    def __init__(self,y_pred,y_target):
        super(MeanAbsoluteError,self).__init__(value = torch.abs(y_pred.value-y_target.value))
        self.y_pred = y_pred
        self.y_target = y_target

        back_channel = Node(autoVariable = y_pred,vjp = self.der)
        self.channels.append(back_channel)

    def der(self,gradient):
        value = self.y_pred.value - self.y_target.value
        value = value/torch.abs(value)  
        return autoTensor(value=value)

    def __repr__(self):
        value = torch.sum(self.value)/self.value.size()[0]
        return f"Cost : autoTensor({value})"


class BinaryCrossEntropy(Loss):
    """Produces BinaryCrossEntropy loss of a model"""
    def __init__(self, y_pred, y_target):
        super(BinaryCrossEntropy, self).__init__(value = -(y_target.value*torch.log(y_pred.value) + (1 - y_target.value)*torch.log(1 - y_pred.value)))
        self.y_pred = y_pred
        self.y_target = y_target

        back_channel = Node(autoVariable = y_pred, vjp = self.der)
        self.channels.append(back_channel)

    def der(self, gradient):
        value = -self.y_target.value/self.y_pred.value + (1 - self.y_target.value)/(1 - self.y_pred.value)
        return autoTensor(value = value)

    def single(self):
        value = torch.sum(self.value)/self.value.size()[0]
        return value
        
    def __repr__(self):
        value = torch.sum(self.value)/self.value.size()[0]
        return f"Cost : autoTensor({value})"

class LogLikelihood(Loss):
    """Minus Log Likelihood Function is similar to multiclass cross entropy Loss"""

    def __init__(self, y_pred, y_target):
        super(LogLikelihood, self).__init__(value = -(y_target.value*torch.log(y_pred.value)))
        self.y_pred = y_pred
        self.y_target = y_target

        back_channel = Node(autoVariable = y_pred, vjp = self.der)
        self.channels.append(back_channel)

    def der(self, gradient):
        value = self.y_target.value/self.y_pred.value
        return autoTensor(value = -value)

    def __repr__(self):
        value = -torch.sum(self.value)/self.value.size()[0]
        return f"Cost : autoTensor({value})"

