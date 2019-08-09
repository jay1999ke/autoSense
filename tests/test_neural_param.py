import unittest
import pytest


#imports below
from autosense.neural.param import Initializer, Weight
from autosense.autodiff.autotensor import autoTensor
import torch

class test_class_Weight(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        weight= Weight((5,6))
        assert isinstance(weight,autoTensor)
        assert weight.value.size() == torch.empty((5,6)).size()

        initer = Initializer("xavier")
        weight= Weight((5,6),initializer=initer)

        assert isinstance(weight,autoTensor)
        assert weight.value.size() == torch.empty((5,6)).size()

    def test_updade_weights(self):
        
        weight= Weight((5,6))
        pre = weight.value.cpu().detach().clone()
        weight.grad =autoTensor( torch.rand((5,6)))

        weight.update_weights(learning_rate=0.01)

        assert torch.equal(weight.value,pre - 0.01*weight.grad.value)


        
class test_class_Initializer(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_createParam(self):
        initer = Initializer("xavier")

        weight = initer.createParam((5,6))

        assert isinstance(weight,torch.Tensor)
        assert weight.size() == torch.empty((5,6)).size()

    def test_createParamStatic(self):
        weight = Initializer.createParamStatic((5,6))        

        assert isinstance(weight,torch.Tensor)
        assert weight.size() == torch.empty((5,6)).size()