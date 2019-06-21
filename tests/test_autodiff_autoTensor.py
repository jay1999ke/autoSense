import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below
from autodiff.autotensor import make_autoTensor,make_torchTensor,autoTensor,Node
import torch

class test_autoTensor(unittest.TestCase):
    pass
    #TODO

class test_Node(unittest.TestCase):
    pass
    #TODO

class test_func_make_autoTensor(unittest.TestCase):
    def test_make_autoTensor(self):
        obj = make_autoTensor(0)
        obj = make_autoTensor(True)
        obj = make_autoTensor(9)
        obj = make_autoTensor(obj)

class test_func_make_torchTensor(unittest.TestCase):

    def test_make_torchTensor(self):
        obj = make_torchTensor(0)
        obj = make_torchTensor(True)
        obj = make_torchTensor(9)
        obj = make_torchTensor(make_autoTensor(obj))
