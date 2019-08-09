import unittest
import pytest

#imports below
from autosense.autodiff.autotensor import autoTensor, make_autoTensor
from autosense.neural.loss import SquareError, MeanAbsoluteError, BinaryCrossEntropy, LogLikelihood
import torch

class test_class_SquareError(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):

        y = make_autoTensor(torch.ones(5,1))
        y_pred = make_autoTensor(torch.rand(5,1))

        loss = SquareError(y_pred=y_pred,y_target=y)

        assert loss.channels[0].autoVariable == y_pred

    def test_der(self):
        y = make_autoTensor(torch.ones(5,1))
        y_pred = make_autoTensor(torch.rand(5,1))
        y_pred.requires_grad = True
        loss = SquareError(y_pred,y)

        loss.backward()
    
        assert torch.equal(y_pred.grad.value,y_pred.value - y.value)

class test_class_MeanAbsoluteError(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_BinaryCrossEntropy(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_LogLikelihood(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO