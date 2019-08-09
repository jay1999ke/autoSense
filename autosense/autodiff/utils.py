import torch
from autosense.autodiff.autotensor import autoTensor
from copy import deepcopy

def reverse_broadcast(gradient,tensor):
    grad_np = deepcopy(gradient.value)

    #when tensor was broadcasted by extending dimenstions
    number_of_added_dimentions = len(gradient.value.size()) - len(tensor.value.size() )
    for _ in range(number_of_added_dimentions):
        grad_np = grad_np.sum(dim=0)

    # undo simple broadcasting
    for i,dimention in enumerate(tensor.value.size()):
        if dimention == 1:
            grad_np = grad_np.sum(dim=i,keepdim=True)
    gradient = autoTensor(value= grad_np)

    assert gradient.size() == tensor.size()
    return gradient


