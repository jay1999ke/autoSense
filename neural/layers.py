from autodiff.autotensor import autoTensor, Node
import torch

class Layer(object):
    """Abstract class that is inherited by all types of layers""" 

    def __call__(self):
        raise  NotImplementedError("Should have implemented this")
    
class Linear(Layer):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim =  output_dim

    # TODO:
    def __call__(self):
        raise  NotImplementedError("Should have implemented this")
