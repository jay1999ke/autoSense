from autosense.autodiff.autotensor import autoTensor, Node
from autosense.neural.param import Weight
import autosense.neural as neural

class optimNode(Node):
    
    @staticmethod
    def dfs_update_param(channels,learning_rate):
        for back_channel in channels:
            if isinstance(back_channel.autoVariable,neural.param.Weight):
                back_channel.autoVariable.update_weights(learning_rate)
            optimNode.dfs_update_param(back_channel.autoVariable.channels,learning_rate)

    @staticmethod
    def dfs_update_gdmomentum(channels,learning_rate,beta):
        for back_channel in channels:
            if isinstance(back_channel.autoVariable,neural.param.Weight):
                if back_channel.autoVariable.grad_saved == None:
                    back_channel.autoVariable.grad_saved = autoTensor(value = back_channel.autoVariable.grad.value)

                back_channel.autoVariable.gdmomentum(learning_rate,beta)
            optimNode.dfs_update_gdmomentum(back_channel.autoVariable.channels,learning_rate,beta)      

class Optimizer(object):
    
    def __init__(self,type,loss,learning_rate,beta=0.9,beta2=0.999):
        self.type = type.lower()
        self.loss = loss
        self.learning_rate = learning_rate
        self.beta = beta
        self.beta2 = beta2

    def step(self):
        if self.type == "sgd":
            optimNode.dfs_update_param(self.loss.channels,self.learning_rate)
        elif self.type == "gdmomentum":
            #standard gd with momentum
            optimNode.dfs_update_gdmomentum(self.loss.channels,self.learning_rate,self.beta)

        #TODO : complete other optim methods

        