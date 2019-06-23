import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.autotensor import autoTensor
from neural.layers import Linear
import torch

class test_class_Linear(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        layer = Linear(6,1)
        
        assert layer.weight.size() == torch.empty(6,1).size()
        assert layer.bias.size() == torch.empty(1,1).size()

    def test_call(self):
        layer = Linear(6,1)
        X = autoTensor(torch.rand(5,6))

        l_out = layer(X)

        assert torch.equal(l_out.value, torch.mm(X.value,layer.weight.value) + layer.bias.value)
