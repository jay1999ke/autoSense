import numpy as np
from autodiff import autoTensor
import matplotlib.pyplot as plt
import torch

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

    X = autoTensor(torch.Tensor(X).type(torch.FloatTensor))
    X2 = autoTensor(torch.Tensor(X2).type(torch.FloatTensor))    
    y = autoTensor(torch.Tensor(y).type(torch.FloatTensor))

    w = autoTensor(torch.rand(1,1) *0.09,requires_grad=True) 
    w2 = autoTensor(torch.rand(1,1) *0.09,requires_grad=True) 
    b = autoTensor(torch.rand(1,1)* 0.09,requires_grad=True)


    h = (X * w + X2*w2) + b
    grad = h - y
    grad.requires_grad = False
    h.backprop(grad)
    z = h

    lr = 0.000001

    for x in range(60000):

        h = (X * w + X2*w2) + b
        grad = h - y
        grad.requires_grad = False
        h.backprop(grad)

        w.value -= lr* w.grad.value
        b.value -= lr* b.grad.value
        w2.value -= lr*w2.grad.value

        if x%3000 == 0:
            loss = 0.5*torch.sum((h.value - y.value)**2)
            print(x,"\t","loss: ",loss)
        grad.grad_sweep()

    print(x,"loss: ",loss)

    w = w.numpy()[0]
    w2 = w2.numpy()[0]
    b = b.numpy()[0]
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

    