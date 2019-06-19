import numpy as np
from autodiff import autoTensor
import matplotlib.pyplot as plt
import torch




if __name__ == "__main__":

    data = np.matrix(np.genfromtxt("testdata/ex1data1.txt", delimiter=','))
    yth = np.array(data.shape)[1]-1
    X = np.array(data[:,yth-1])
    y = np.array(data[:, yth])
    X,y = y,X
    X_c = X
    y_c = y
    m=y.size

    #plt.scatter(X[:, 0], y)
    #plt.show()

    X = autoTensor(torch.Tensor(X).type(torch.FloatTensor))
    y = autoTensor(torch.Tensor(y).type(torch.FloatTensor))

    w = autoTensor(torch.rand(1,1) *0.09,requires_grad=True) 
    b = autoTensor(torch.rand(1,1)* 0.09,requires_grad=True)


    h = (X * w) + b
    grad = h -y
    grad.requires_grad = False
    h.backprop(grad)

    lr = 0.0001

    for x in range(2000):

        h = (X * w) + b
        grad = h - y
        grad.requires_grad = False
        h.backprop(grad)

        w.value -= lr* w.grad.value
        b.value -= lr* b.grad.value

        grad.grad_sweep()

        if x%200 == 0:
            loss = 0.5*torch.sum((h.value - y.value)**2)
            print(x,"\t","loss: ",loss)
    print(x,"loss: ",loss)

    w = w.numpy()[0]
    b = b.numpy()[0]
    print(w,b)

    q = []
    e=[]
    for x in range(4,25):
        q.append(x)
        e.append(b + w*x)
    plt.scatter(X_c[:, 0], y_c)
    plt.plot(e)
    plt.show()

    pred = h.numpy()
    for x in range(m):
        #print(pred[x],y_c[x],pred[x]-y_c[x])
        pass

    