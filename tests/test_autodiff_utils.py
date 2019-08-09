import unittest
import pytest

#imports below
from autosense.autodiff.functional import Add, Substract, MatMul, Multiply, Power, Sum, Divide, Negate, tanh, sigmoid, relu
from autosense.autodiff.autotensor import autoTensor, make_autoTensor,Node
import torch

class test_func_reverse_broadcast(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3]])
        obj1.requires_grad = True
        obj = Add(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3]])
        obj2.requires_grad = True
        obj = Add(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj2.grad.value - make_autoTensor([[2,2]]).value) == 0
