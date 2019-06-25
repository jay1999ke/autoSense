import torch
from autodiff.autotensor import autoTensor, Node
from autodiff.utils import reverse_broadcast
import torch.nn.functional as F

class MatMul(autoTensor):
    """Matrix multiplication "autoTensor" function"""

    def __init__(self, tensor1, tensor2):
        super(MatMul,self).__init__(torch.mm(tensor1.value,tensor2.value))
        self.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor2 = tensor2
            self.channels.append(back_channel)
        if tensor2.requires_grad:
            back_channel = Node(tensor2, self.der_pos2)
            self.tensor1 = tensor1
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        value = torch.mm(gradient.value,self.tensor2.value.transpose(1,0))
        return autoTensor(value=value)

    def der_pos2(self, gradient):
        value = torch.mm(self.tensor1.value.transpose(1,0),gradient.value)
        return autoTensor(value=value)

class Add(autoTensor):
    """Addition "autoTensor" function"""

    def __init__(self, tensor1, tensor2):
        super(Add,self).__init__(tensor1.value+tensor2.value)
        self.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.channels.append(back_channel)
        if tensor2.requires_grad:
            back_channel = Node(tensor2, self.der_pos2)
            self.tensor2 = tensor2
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        tensor = self.tensor1
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

    def der_pos2(self, gradient):
        tensor = self.tensor2
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

class Multiply(autoTensor):
    """Multiplication "autoTensor" function"""

    def __init__(self, tensor1, tensor2):
        super(Multiply,self).__init__(tensor1.value*tensor2.value)
        self.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            self.channels.append(back_channel)
        if tensor2.requires_grad:
            back_channel = Node(tensor2, self.der_pos2)
            self.tensor2 = tensor2
            self.tensor1 = tensor1
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        tensor = self.tensor1
        gradient = autoTensor( gradient.value * self.tensor2.value)
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)

    def der_pos2(self, gradient):
        tensor = self.tensor2
        gradient = autoTensor(gradient.value * self.tensor1.value)
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)

class Divide(autoTensor):
    """Division "autoTensor" function"""

    def __init__(self, tensor1, tensor2):
        super(Divide,self).__init__(tensor1.value/tensor2.value)
        self.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            self.channels.append(back_channel)
        if tensor2.requires_grad:
            back_channel = Node(tensor2, self.der_pos2)
            self.tensor2 = tensor2
            self.tensor1 = tensor1
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        tensor = self.tensor1
        gradient = autoTensor( gradient.value * (1/self.tensor2.value))
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)

    def der_pos2(self, gradient):
        tensor = self.tensor2
        gradient = autoTensor(gradient.value * (-self.tensor1.value/(self.tensor2.value**2)))
        gradient = reverse_broadcast(gradient,tensor).value 
        return autoTensor(value=gradient)


class Negate(autoTensor):
    """Negative "autoTensor" function"""

    def __init__(self, tensor1):
        super(Negate,self).__init__(-tensor1.value)
        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos)
            self.channels.append(back_channel)

    def der_pos(self,gradient):
        gradient = autoTensor(value= -gradient.value)
        return gradient

class Substract(autoTensor):
    """Substraction "autoTensor" function"""

    def __init__(self, tensor1, tensor2):
        super(Substract,self).__init__(tensor1.value-tensor2.value)
        self.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.channels.append(back_channel)
        if tensor2.requires_grad:
            back_channel = Node(tensor2, self.der_pos2)
            self.tensor2 = tensor2
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        tensor = self.tensor1
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value=gradient.value)

    def der_pos2(self, gradient):
        tensor = self.tensor2
        gradient = reverse_broadcast(gradient,tensor)
        return autoTensor(value= -gradient.value)

class Power(autoTensor):
    """Power "autoTensor" function"""

    def __init__(self, tensor1, pow_val):
        super(Power,self).__init__(tensor1.value**pow_val)

        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.pow_val = pow_val
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        tensor = self.tensor1
        pow_val = self.pow_val
        gradient = reverse_broadcast(gradient,tensor)
        back_grad = pow_val * gradient.value * (tensor.value**(pow_val-1))
        return autoTensor(value=back_grad)

class Sum(autoTensor):
    """Elements sum "autoTensor" function"""

    def __init__(self, tensor1, axis=0):
        super(Sum,self).__init__(tensor1.value.sum(dim=axis,keepdim=True))

        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.shape = tensor1.value.size()
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        back_grad = gradient.value * torch.ones(self.shape)
        return autoTensor(value=back_grad)


class tanh(autoTensor):
    """tanh"autoTensor" function"""

    def __init__(self,tensor1):
        super(tanh,self).__init__(torch.tanh(tensor1.value))

        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        assert self.value.size() == gradient.value.size()    
        back_grad = gradient.value * (1-self.value**2)
        return autoTensor(value=back_grad)

class sigmoid(autoTensor):
    """sigmoid "autoTensor" function"""

    def __init__(self,tensor1):
        super(sigmoid,self).__init__(torch.sigmoid(tensor1.value))

        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        assert self.value.size() == gradient.value.size()
        back_grad = gradient.value * (self.value*(1-self.value))
        return autoTensor(value=back_grad)

class relu(autoTensor):
    """ReLU "autoTensor" function"""

    def __init__(self,tensor1):
        super(relu,self).__init__(torch.relu(tensor1.value))

        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        assert self.value.size() == gradient.value.size()
        sub_grad = self.value.clone()
        sub_grad[sub_grad > 0] = 1
        back_grad = gradient.value * sub_grad
        return autoTensor(value=back_grad)

class Exp(autoTensor):
    """exp "autoTensor" function"""

    def __init__(self, tensor1):
        super(Exp,self).__init__(torch.exp(tensor1.value))
        self.requires_grad = tensor1.requires_grad

        if tensor1.requires_grad:
            back_channel = Node(tensor1, self.der_pos1)
            self.tensor1 = tensor1
            self.channels.append(back_channel)

    def der_pos1(self, gradient):
        back_grad = gradient.value * self.value
        return autoTensor(value=back_grad)


class Conv2d(autoTensor):
    """Convolution operation"""
    
    def __init__(self,image_block,filters,bias,padding=0,stride=1):
        conv = F.conv2d(input=image_block.value,
            weight=filters.value,
            bias=bias.value,
            stride=(stride,stride),
            padding = (padding,padding)
        )
        super(Conv2d,self).__init__(value=conv)

        self.requires_grad = image_block.requires_grad or filters.requires_grad or bias.requires_grad
        self.padding = padding
        self.stride = stride
        self.image_block = image_block
        self.filters = filters
        self.bias = bias

        if image_block.requires_grad:
            back_channel = Node(image_block,self.der_image)
            self.channels.append(back_channel)
        if filters.requires_grad:
            back_channel = Node(filters,self.der_filters)
            self.channels.append(back_channel)
        if bias.requires_grad:
            back_channel = Node(bias,self.der_bias)
            self.channels.append(back_channel)

    def der_image(self,gradient):
        back_grad = F.conv_transpose2d(input = gradient.value,
            weight = self.filters.value,
            stride=(self.stride,self.stride),
            padding = (self.padding,self.padding)
            ) 
        return autoTensor(value=back_grad)

    def der_filters(self,gradient):
        back_grad = F.conv2d(input = self.image_block.value.permute(1,0,2,3),
                weight = gradient.value.permute(1,0,2,3),
                stride = (self.stride,self.stride),
                padding = (self.padding,self.padding)
            ).permute(1,0,2,3)
        return autoTensor(value=back_grad)

    def der_bias(self,gradient):
        back_grad = gradient.value.mean(0).mean(1).mean(1).view(self.filters.value.size()[0])
        return autoTensor(value=back_grad)