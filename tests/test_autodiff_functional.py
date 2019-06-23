import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.functional import Add, Substract, MatMul, Multiply, Power, Sum, Divide, Negate, tanh, sigmoid, relu, Exp
from autodiff.autotensor import autoTensor, make_autoTensor,Node
import torch

class test_class_MatMul(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3,7],[4,5,9]])
        obj2 = make_autoTensor([[-3,3],[4,5],[0,0]])
        obj1.requires_grad = True
        obj = MatMul(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3,7],[4,5,9]])
        obj2 = make_autoTensor([[-3,3],[4,5],[0,0]])
        obj1.requires_grad = True
        obj = MatMul(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj1.grad.value - torch.mm(make_autoTensor([[1,1],[1,1]]).value,obj2.value.transpose(1,0))) == 0

class test_class_Add(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Add(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Add(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj1.grad.value - make_autoTensor([[1,1],[1,1]]).value) == 0

class test_class_Multiply(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Multiply(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Multiply(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj1.grad.value - obj2.value) == 0

class test_class_Substract(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Substract(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj2.requires_grad = True
        obj = Substract(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj2.grad.value + make_autoTensor([[1,1],[1,1]]).value) == 0

class test_class_Divide(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Divide(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj2.requires_grad = True
        obj = Divide(obj1,obj2)
        obj.backprop(make_autoTensor([[1,1],[1,1]]))

        assert torch.sum(obj1.grad.value - 1/obj1.value) == 0
        assert torch.sum(obj2.grad.value + obj1.value/obj2.value**2) == 0



class test_class_Sum(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Sum(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Sum(obj1)

        obj.backprop(make_autoTensor([1,1]))
    
        assert torch.sum(obj1.grad.value - make_autoTensor([1,1]).value * torch.ones(2,2) ) == 0

class test_class_Power(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Power(obj1,3)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Power(obj1,3)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value - 3 * make_autoTensor([[1,1],[1,1]]).value * obj1.value**2 ) == 0

class test_class_Negate(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Negate(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Negate(obj1)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value + make_autoTensor([[1,1],[1,1]]).value) == 0

class test_class_tanh(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = tanh(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = tanh(obj1)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value - make_autoTensor([[1,1],[1,1]]).value* (1-obj.value**2)) == 0

class test_class_sigmoid(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    
    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = sigmoid(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = sigmoid(obj1)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value - make_autoTensor([[1,1],[1,1]]).value*obj.value *(1-obj.value)) == 0

class test_class_relu(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj =relu(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = relu(obj1)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value - make_autoTensor([[0,1],[1,1]]).value) == 0

class test_class_Exp(unittest.TestCase):

    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj =Exp(obj1)

        assert obj.channels[0].autoVariable == obj1

    def test_der(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Exp(obj1)

        obj.backprop(make_autoTensor([[1,1],[1,1]]))
    
        assert torch.sum(obj1.grad.value - obj.value) == 0
