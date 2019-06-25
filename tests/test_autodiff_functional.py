import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.functional import Flatten2d, Conv2d, Add, Substract, MatMul, Multiply, Power, Sum, Divide, Negate, tanh, sigmoid, relu, Exp
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

class test_class_Conv2d(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        image.requires_grad = True
        filters.requires_grad = True
        bias.requires_grad = True

        obj = Conv2d(image,filters,bias)

        assert obj.channels[0].autoVariable == image
        assert obj.channels[1].autoVariable == filters
        assert obj.channels[2].autoVariable == bias
        assert obj.size() == torch.empty(3,6,16,16).size()

        obj1 = Conv2d(image,filters,bias,padding=1)
        assert obj1.size() == torch.empty(3,6,18,18).size()

        obj2 = Conv2d(image,filters,bias,stride=3)
        assert obj2.size() == torch.empty(3,6,6,6).size()

        obj3 = Conv2d(image,filters,bias,padding=3,stride=4)
        assert obj3.size() == torch.empty(3,6,6,6).size()


    def test_der_image(self):
        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        image.requires_grad = True

        obj = Conv2d(image,filters,bias)
        obj.backprop(autoTensor(torch.rand(3,6,16,16)))

        assert image.grad.size() == image.size()

        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        image.requires_grad = True

        obj = Conv2d(image,filters,bias,padding=1)
        obj.backprop(autoTensor(torch.rand(3,6,18,18)))

        assert image.grad.size() == image.size()

        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        image.requires_grad = True

        obj = Conv2d(image,filters,bias,stride=3)
        obj.backprop(autoTensor(torch.rand(3,6,6,6)))

        assert image.grad.size() == image.size()

        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        image.requires_grad = True

        obj = Conv2d(image,filters,bias,padding=3,stride=3)
        obj.backprop(autoTensor(torch.rand(3,6,8,8)))

        assert image.grad.size() == image.size()


    def test_der_filter(self):
        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        filters.requires_grad = True

        obj = Conv2d(image,filters,bias)
        obj.backprop(autoTensor(torch.rand(3,6,16,16)))

        assert filters.grad.size() == filters.size()

        image1 = autoTensor(torch.rand(3,3,20,20))
        filters1 = autoTensor(torch.rand(6,3,5,5))
        bias1 = autoTensor(torch.rand([6]))
        filters1.requires_grad = True

        obj1 = Conv2d(image1,filters1,bias1,padding=1)
        obj1.backprop(autoTensor(torch.rand(3,6,18,18)))

        assert filters1.grad.size() == filters1.size()

        image2 = autoTensor(torch.rand(3,3,20,20))
        filters2 = autoTensor(torch.rand(6,3,5,5))
        bias2 = autoTensor(torch.rand([6]))
        filters2.requires_grad = True

        obj2 = Conv2d(image2,filters2,bias2,stride=4)
        obj2.backprop(autoTensor(torch.rand(3,6,4,4)))

        assert filters2.grad.size() == filters2.size()

        image2 = autoTensor(torch.rand(3,3,20,20))
        filters2 = autoTensor(torch.rand(6,3,5,5))
        bias2 = autoTensor(torch.rand([6]))
        filters2.requires_grad = True

        obj2 = Conv2d(image2,filters2,bias2,stride=3)
        obj2.backprop(autoTensor(torch.rand(3,6,6,6)))

        assert filters2.grad.size() == filters2.size()


    def test_der_bias(self):
        image = autoTensor(torch.rand(3,3,20,20))
        filters = autoTensor(torch.rand(6,3,5,5))
        bias = autoTensor(torch.rand([6]))
        bias.requires_grad = True

        obj = Conv2d(image,filters,bias)
        obj.backprop(autoTensor(torch.rand(3,6,16,16)))

        assert bias.grad.size() == bias.size()

class test_class_Flatten2d(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))

    def test_init(self):
        image = autoTensor(torch.rand(3,3,20,20))
        image.requires_grad = True

        obj = Flatten2d(image)

        assert obj.size() == torch.empty(3,1200).size()
        assert obj.channels[0].autoVariable == image

    def test_der(self):
        image = autoTensor(torch.rand(3,3,20,20))
        image.requires_grad = True

        obj = Flatten2d(image)
        obj.backprop(autoTensor(obj.value))

        assert image.value.size() == image.grad.value.size()
        assert torch.equal(image.value,image.grad.value)
