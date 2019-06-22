import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.autotensor import make_autoTensor,make_torchTensor,autoTensor,Node
import torch

class test_class_Node(unittest.TestCase):
    pass
    #TODO

class test_class_autoTensor(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_grad_zeros(self):
        obj = make_autoTensor([[-3,3],[4,5]])

        obj.grad = make_autoTensor([[-3,3],[4,5]])
        obj.grad_zeros()
        assert torch.sum(obj.grad.value) == 0

    def test_backprop(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj3 = make_autoTensor([[-3,3],[4,5]])
        obj4 = make_autoTensor([[1,1],[1,1]])
        obj1.requires_grad = True
        obj2.requires_grad = False
        obj3.requires_grad = True
        obj4.requires_grad = True
        obj5 = (obj1 + obj2) + (obj3 * obj4)

        obj5.backprop(make_autoTensor([[-3,3],[4,5]]))

        assert torch.sum(obj1.grad.value - obj1.value) == 0        
        assert torch.sum(obj3.grad.value - make_autoTensor([[-3,3],[4,5]]).value) == 0
        assert torch.sum(obj4.grad.value == obj3.value) == 0     

    def test_grad_sweep(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj3 = make_autoTensor([[-3,3],[4,5]])
        obj4 = make_autoTensor([[1,1],[1,1]])
        obj1.requires_grad = True
        obj2.requires_grad = False
        obj3.requires_grad = True
        obj4.requires_grad = True
        obj5 = (obj1 + obj2) + (obj3 * obj4)

        obj5.backprop(make_autoTensor([[-3,3],[4,5]]))
        obj5.grad_sweep()
        assert torch.sum(obj1.grad.value) == 0
        assert obj2.grad == None               
        assert torch.sum(obj3.grad.value) == 0
        assert torch.sum(obj4.grad.value) == 0             
    
class test_func_make_autoTensor(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

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
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_make_torchTensor(self):
        obj = make_torchTensor(0)
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(True)
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(torch.rand(7,1))
        assert isinstance(obj,torch.Tensor)

        obj = make_torchTensor(torch.rand(7,1))
        assert isinstance(obj,torch.Tensor)
