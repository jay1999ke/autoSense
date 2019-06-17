import torch


class autoTensor(object):
    
    def __init__(self,value,reqiures_grad=False):
        self.value = value
        self.reqiures_grad = reqiures_grad