import torch # emit
from autotensor import autoTensor, Node
import numpy as np

class MatMul(autoTensor):
    def __init__(self, at1, at2):
        super(MatMul,self).__init__(torch.mm(at1.value,at2.value))
        self.requires_grad = at1.requires_grad or at2.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.at2 = at2
            self.dependencies.append(depend)
        if at2.requires_grad:
            depend = Node(at2, self.der_pos2)
            self.at1 = at1
            self.dependencies.append(depend)


    def der_pos1(self, gradient):
        value = torch.mm(gradient.value,self.at2.value.transpose(1,0))
        return autoTensor(value=value)

    def der_pos2(self, gradient):
        value = torch.mm(self.at1.value.transpose(1,0),gradient.value)
        return autoTensor(value=value)
