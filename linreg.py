import numpy as np
from autodiff import autoTensor
from neural import Loss, Weight, Initializer, Linear
import matplotlib.pyplot as plt
import torch
import time

if __name__ == "__main__":

    data = np.matrix(np.genfromtxt("testdata/ex1data1.txt", delimiter=','))
    yth = np.array(data.shape)[1]-1
    X = np.array(data[:,yth-1])
    y = np.array(data[:, yth])
    X_c = X
    y_c = y
    m=y.size
    X2 = X**2
    X2_c = X2

    #plt.scatter(X[:, 0], y)
    #plt.show()

    X = autoTensor(X)
    X2 = autoTensor(X2)    
    y = autoTensor(y)

    initer = Initializer("he")

    l1 = Linear(1,1,initer)
    l2 = Linear(1,1,initer,bias=False)   

    lr = 0.000001
    s = time.time()
    for x in range(60000):

        h = l1(X) + l2(X2)
        loss = Loss.SquareError(h,y)
        loss.backward()

        l1.weight.value -= lr* l1.weight.grad.value
        l2.weight.value -= lr* l2.weight.grad.value
        l1.bias.value -= lr* l1.bias.grad.value

        if x%3000 == 0:
            print(x,"\t","loss: ",loss,time.time()-s)
            s = time.time()
        loss.grad_sweep()

    print(x,"loss: ",loss)

    w = l1.weight.numpy()[0]
    w2 = l2.weight.numpy()[0]
    b = l1.bias.numpy()[0]
    print(w,b)

    e=[]
    X__c = X_c*w
    X2_c = X2_c*w2
    X = X__c +X2_c + b
    for x in range(m):
        e.append(X[x])
    plt.scatter(X_c[:, 0], y_c)
    plt.scatter(X_c[:, 0], e)
    plt.show()


    