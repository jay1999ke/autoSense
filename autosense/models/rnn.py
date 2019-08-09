# analysis file
# TODO:approriate changes required post-analysis

from autosense.autodiff.autotensor import autoTensor, Node
import torch
from autosense.neural.param import Weight, Initializer
from autosense.neural.layers import Linear2
import autosense.autodiff.functional as F
import torch.nn.init as torchInit

class LSTMnode(object):

    def __init__(self,input_size,hidden_size,output_size,initializer=None,):
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.f = Linear2(input_size,hidden_size,output_size,initializer)
        self.i = Linear2(input_size,hidden_size,output_size,initializer)
        self.c_ = Linear2(input_size,hidden_size,output_size,initializer)
        self.o = Linear2(input_size,hidden_size,output_size,initializer)

        self.c = autoTensor(torch.zeros(output_size,1))
        self.f.weight.name = "f1"
        self.f.weight2.name = "f2"

        self.i.weight.name = "i1"
        self.i.weight2.name = "i2"

        self.c_.weight.name = "c_1"
        self.c_.weight2.name = "c_2"

        self.o.weight.name = "o1"
        self.o.weight2.name = "o2"


    def forward(self,h,x,it):
        it = str(it)
        f = F.sigmoid(self.f(h,x))
        i = F.sigmoid(self.i(h,x))
        c_ = F.tanh(self.c_(h,x))
        o = F.sigmoid(self.o(h,x))
        self.c = f*self.c + i*c_
        h.requires_grad=False
        h = o*F.tanh(self.c)
        f.name = "g_f"+it
        i.name = "g_i"+it
        c_.name = "g_c_"+it
        o.name = "g_o"+it

        return h


    def backprop(self,gradient):

        """
        input: 4 grads f,i,c_,o


        """
        pass


    """
    def dfs(channels,gradient):
        This is where the magic happens
        for back_channel in channels:
            if back_channel.autoVariable.requires_grad:
                back_gradient = back_channel.vjp(gradient)
                back_channel.autoVariable.backprop(back_gradient)
    """