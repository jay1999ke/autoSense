import torch
from autodiff.autotensor import autoTensor
import numpy as np

def reverse_broadcast(gradient,tensor):
    grad_np = gradient.numpy()

    #when tensor was broadcasted by extending dimenstions
    number_of_added_dimentions = len(gradient.value.size()) - len(tensor.value.size() )
    for _ in range(number_of_added_dimentions):
        grad_np = grad_np.sum(axis=0)

    # undo simple broadcasting
    for i,dimention in enumerate(tensor.value.size()):
        if dimention == 1:
            grad_np = grad_np.sum(axis=i,keepdims=True)
    gradient.value = torch.Tensor(grad_np)

    assert gradient.size() == tensor.size()
    return gradient


