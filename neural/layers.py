from autodiff.autotensor import autoTensor, Node
import torch
from neural import Weight, Initializer
import autodiff.functional as F

class Layer(object):
    """Abstract class that is inherited by all types of layers""" 

    def __call__(self):
        raise  NotImplementedError
    
class Linear(Layer):

    def __init__(self, input_dim, output_dim,initializer=None,bias=True):
        self.weight = Weight(shape=(input_dim,output_dim),initializer=initializer)
        if bias:
            self.bias = Weight(shape=(1,output_dim),initializer=initializer)
        self.bias_present = bias

    def __call__(self,inputs):
        if self.bias_present:
            return F.MatMul(inputs,self.weight) + self.bias
        else:
            return F.MatMul(inputs,self.weight) 
