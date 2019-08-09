import torch
import numpy as np
import scipy.io as mat
from autosense.autodiff import autoTensor
from autosense.neural import Loss, Weight, Initializer, Linear, Optimizer, optimNode, Conv2D, Dropout
import autosense.autodiff.functional as F
import torch.nn.init as torchInit
import matplotlib.pyplot as plt
import gc

def get_accuracy_value(pred, y):
    return abs(torch.sum(y.value.max(dim=1)[1] == pred.value.max(dim=1)[1]).item()/y.size()[0])


if __name__ == "__main__":
    raw_data = mat.loadmat("testdata/mnist_reduced.mat")
    X = raw_data['X'].astype(np.float64)
    y = raw_data['y'].ravel()
    Y = np.zeros((5000, 10), dtype='uint8')
    for i, row in enumerate(Y):
        Y[i, y[i] - 1] = 1
    y = Y.astype(np.float64)

    X = autoTensor(X)
    y = autoTensor(y)
    X.value = X.value.view(X.size()[0],1,20,20)
    X.value = X.value.permute(0,1,3,2)

    """
    fig, ax = plt.subplots(nrows=2, ncols=5)
    c=0
    for row in ax:
        for col in row:
            col.imshow(X.value[c][0])
            c+=500
    plt.show()"""
    
    
    initer = Initializer("xavier")
    layer1 = Conv2D((3,1,7,7),initializer=initer)
    layer2 = Conv2D((6,3,7,7),initializer=initer)
    layer3 = Linear(384,10,initializer=initer)

    lr = 0.000007

    for x in range(500):

        l1 = F.relu(layer1(X))
        l2 = F.Flatten2d(F.relu(layer2(l1)))
        l3 = F.sigmoid(layer3(l2))
        loss = Loss.BinaryCrossEntropy(l3,y)
        loss.backward()

        SGD = Optimizer("sgd",loss,lr)
        SGD.step()

        print(x,loss,get_accuracy_value(l3,y))
        loss.grad_sweep()

        if x%100 == 0:
            gc.collect()
