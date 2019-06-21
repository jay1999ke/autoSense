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

    q = []
    e=[]
    for x in range(4,25):
        q.append(x)
        e.append(b + w*x + w2*(x**2))
    plt.scatter(X_c[:, 0], y_c)
    plt.plot(e)
    plt.show()

    pred = h.numpy()
    for x in range(m):
        #print(pred[x],y_c[x],pred[x]-y_c[x])
        pass

    