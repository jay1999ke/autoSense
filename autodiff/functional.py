import torch
from autodiff.autotensor import autoTensor, Node
from autodiff.utils import reverse_broadcast

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

class Add(autoTensor):
    def __init__(self, at1, at2):
        super(Add,self).__init__(at1.value+at2.value)
        self.requires_grad = at1.requires_grad or at2.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.at1 = at1
            self.dependencies.append(depend)
        if at2.requires_grad:
            depend = Node(at2, self.der_pos2)
            self.at2 = at2
            self.dependencies.append(depend)

    def der_pos1(self, gradient):
        tensor = self.at1
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

    def der_pos2(self, gradient):
        tensor = self.at2
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

class Multiply(autoTensor):
    def __init__(self, at1, at2):
        super(Multiply,self).__init__(at1.value*at2.value)
        self.requires_grad = at1.requires_grad or at2.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.at1 = at1
            self.at2 = at2
            self.dependencies.append(depend)
        if at2.requires_grad:
            depend = Node(at2, self.der_pos2)
            self.at2 = at2
            self.at1 = at1
            self.dependencies.append(depend)

    def der_pos1(self, gradient):
        tensor = self.at1
        gradient = autoTensor( gradient.value * self.at2.value)
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)

    def der_pos2(self, gradient):
        tensor = self.at2
        gradient = autoTensor(gradient.value * self.at1.value)
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)

class Negate(autoTensor):
    def __init__(self, at1):
        super(Negate,self).__init__(-at1.value)
        self.requires_grad = at1.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.dependencies.append(depend)

    def der_pos(self,gradient):
        gradient = autoTensor(value= -gradient.value)
        return gradient

class Substract(autoTensor):
    def __init__(self, at1, at2):
        super(Substract,self).__init__(at1.value-at2.value)
        self.requires_grad = at1.requires_grad or at2.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.at1 = at1
            self.dependencies.append(depend)
        if at2.requires_grad:
            depend = Node(at2, self.der_pos2)
            self.at2 = at2
            self.dependencies.append(depend)

    def der_pos1(self, gradient):
        tensor = self.at1
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

    def der_pos2(self, gradient):
        tensor = self.at2
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value= -gradient.value)

class Power(autoTensor):
    def __init__(self, at1, pow_val):
        super(Power,self).__init__(at1.value**pow_val.value)

        self.requires_grad = at1.requires_grad

        if at1.requires_grad:
            depend = Node(at1, self.der_pos1)
            self.at1 = at1
            self.pow_val = pow_val
            self.dependencies.append(depend)

    def der_pos1(self, gradient):
        tensor = self.at1
        pow_val = self.pow_val
        gradient = reverse_broadcast(gradient,tensor)
        back_grad = pow_val.value * gradient.value * (tensor.value**(pow_val.value-1))
        return autoTensor(value=back_grad)
