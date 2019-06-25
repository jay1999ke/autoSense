from autodiff.autotensor import autoTensor, Node
from neural.param import Weight


class Optimizer(object):
    
    def __init__(self,type,loss,learning_rate,beta=0.9,beta2=0.999):
        self.type = type.lower()
        self.loss = loss
        self.learning_rate = learning_rate
        self.beta = beta
        self.beta2 = beta2

    def step(self):
        if self.type == "sgd":
            Node.dfs_update_param(self.loss.channels,self.learning_rate)
        elif self.type == "gdmomentum":
            #standard gd with momentum
            Node.dfs_grad_copy(self.loss.channels)

            #TODO: complete this

        #TODO : complete other optim methods

        