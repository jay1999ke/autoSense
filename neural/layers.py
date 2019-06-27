from autodiff.autotensor import autoTensor, Node
import torch
from neural import Weight, Initializer
import autodiff.functional as F
import torch.nn.init as torchInit

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

class Conv2D(Layer):

    def __init__(self,filter_shape,padding=0,stride=1,initializer=None):
        """
        input – input tensor of shape (minibatch,in_channels,iH,iW) \n
        weight – filters of shape (out_channels,in_channels,kH,kW) \n
        bias – bias tensor of shape (out_channels). """
        self.padding = padding
        self.stride = stride
        self.filter = Weight(filter_shape,initializer=initializer)
        self.bias = Weight(shape = filter_shape[0])

    def __call__(self,inputs):
        return F.Conv2d(image_block=inputs,
                        filters=self.filter,
                        bias=self.bias,
                        padding=self.padding,
                        stride=self.stride
                        )

class Dropout(Layer):

    def __init__(self,input_shape,keep_prob=0.8):

        self.input_shape = input_shape
        self.keep_prob = keep_prob

    def __call__(self,inputs):
        mask = torchInit.uniform_(torch.rand(self.input_shape)).type(inputs.value.type())
        mask[mask < self.keep_prob] = 1
        mask[mask != 1 ] = 0
        return F.Dpout(inputs,mask)
