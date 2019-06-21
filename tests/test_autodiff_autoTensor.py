import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.autotensor import make_autoTensor,make_torchTensor,autoTensor,Node
import torch

class test_class_autoTensor(unittest.TestCase):
    pass
    #TODO

class test_class_Node(unittest.TestCase):
    pass
    #TODO

class test_func_make_autoTensor(unittest.TestCase):
    def test_make_autoTensor(self):
        obj = make_autoTensor(0)
        assert isinstance(obj,autoTensor)

        obj = make_autoTensor(True)
        assert isinstance(obj,autoTensor)

        obj = make_autoTensor(torch.rand(1))
        assert isinstance(obj,autoTensor)

        obj = make_autoTensor(obj)
        assert isinstance(obj,autoTensor)

class test_func_make_torchTensor(unittest.TestCase):

    def test_make_torchTensor(self):
        obj = make_torchTensor(0)
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(True)
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(torch.rand(7,1))
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(torch.rand(7,1))
        assert isinstance(obj,torch.Tensor)
