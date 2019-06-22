import unittest
import pytest
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

#imports below


class test_class_SquareError(unittest.TestCase):
    def setup_method(self, method):
        print("\n%s:%s" % (type(self).__name__, method.__name__))
    pass
    #TODO

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