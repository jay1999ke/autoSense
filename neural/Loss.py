from autodiff import autoTensor, Node

#for implemantation of all types of losess

class Loss(autoTensor):
    def __init__(self,value, channels=None, requires_grad = True):
        super(Loss,self).__init__(value=value, channels=None, requires_grad=requires_grad)

class SquareError(Loss):
    """Produces Square error loss of Model"""

    def __init__(self,y_pred,y_target):
        super(SquareError,self).__init__(value=0.5*((y_pred.value-y_target)**2))
        self.y_pred = y_pred
        self.y_target = y_target

        back_channel = Node(autoVariable=y_pred,vjp=self.der)
        self.channels.append(back_channel)

    def der(self,gradient):
        value = self.y_pred.value - self.y_target.value
        return autoTensor(value=value)



# TODO: 
# implemantation of all types of losess
