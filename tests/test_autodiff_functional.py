import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.functional import Add, Substract, MatMul, Multiply, Power, Sum, Divide, Negate, tanh, sigmoid, relu
from autodiff.autotensor import autoTensor, make_autoTensor,Node

class test_class_MatMul(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Add(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        obj1 = make_autoTensor([[-3,3],[4,5]])
        obj2 = make_autoTensor([[-3,3],[4,5]])
        obj1.requires_grad = True
        obj = Add(obj1,obj2)

        assert obj.channels[0].autoVariable == obj1

class test_class_Substract(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Multiply(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Divide(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Sum(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Power(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_Negate(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_tanh(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_sigmoid(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

class test_class_relu(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO