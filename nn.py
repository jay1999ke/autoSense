import torch
import numpy as np
import scipy.io as mat
from autodiff import autoTensor
from neural import Loss, Weight, Initializer, Linear, Optimizer, optimNode
import autodiff.functional as F

def get_accuracy_value(pred, y):
    return abs(torch.sum(y.value.max(dim=1)[1] == pred.value.max(dim=1)[1]).item()/y.size()[0])

if __name__ == "__main__":
    raw_data = mat.loadmat("testdata/mnist_reduced.mat")
    X = raw_data['X'].astype(np.float64)
    y = raw_data['y'].ravel()
    c_y=torch.Tensor(y).type(dtype=torch.FloatTensor)
    Y = np.zeros((5000, 10), dtype='uint8')
    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1
    y = Y.astype(np.float64)

    X = autoTensor(X)
    y = autoTensor(y)

    initer = Initializer("xavier")

    layer1 = Linear(400,25,initializer=initer)
    layer2 = Linear(25,25,initializer=initer)
    layer3 = Linear(25,10,initializer=initer)

    lr = 0.0001

    for x in range(1000):

        l1 = layer1(X)
        l2 = F.relu(layer2(l1))
        l3 = F.sigmoid(layer3(l2))
        loss = Loss.SquareError(l3,y)
        loss.backward()
        SGD = Optimizer("gdmomentum",loss,lr,beta=0.2)
        SGD.step()


        if x%50 == 0:
            print(x,loss,get_accuracy_value(l3,y))
        loss.grad_sweep()


