import torch
import string
import numpy as np
import scipy.io as mat
from autodiff import autoTensor
from neural import Loss, Weight, Initializer, Linear, Optimizer, Linear2
import autodiff.functional as F
import torch.nn.init as torchInit
import matplotlib.pyplot as plt
import gc
from time import time


def getXY(file_name):

    file_data = open(file_name,"r",encoding="utf-8")
    raw_data = file_data.read().split("\n")

    y = np.zeros((len(raw_data)-1,1))
    X=[]
    c=0
    for ins in raw_data:
        data = ins.split("\t")
        X.append(data[0])
        try:
            y[c]=int(data[1])
        except:
            pass
        c+=1
    X=X[:1000]

    X_=[]
    for sent in X:
        l=sent.lower().strip(" \t\n\r,;.)'(!").translate(str.maketrans('', '', string.punctuation)).split(" ")
        X_.append(l)

    c=0
    for x in X_:
        x.append(y[c])
        c+=1

    X = sorted(X_,key=len)

    for c,x in enumerate(X):
        y[c] = x[-1]
        del x[-1]
    
    return X,y

def loadGloveModel(filename):
    print("Loading Glove Model...")
    
    glove_vocab = []
    glove_embed=[]
    
    file = open(filename,'r',encoding='UTF-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        glove_embed.append(embed_vector)
    
    print('Loaded GLOVE')
    file.close()
    glove_embed[38522].insert(0,0)
    return glove_vocab,glove_embed

#######################################################################################

class lstm(object):

    def __init__(self):
        init = Initializer("xavier")

        self.f = Linear2(25,25,25,init)
        self.i = Linear2(25,25,25,init)
        self.c_ = Linear2(25,25,25,init)
        self.o = Linear2(25,25,25,init)

        self.out = Linear(25,1,init)

        self.c = autoTensor(torch.zeros(1,25))

    def forward(self,X,y,glove_vocab,glove_embed):
        
        for i,sentence in enumerate(X):
            self.c=autoTensor(torch.zeros(1,25))
            y_t = autoTensor(y[i])
            h = autoTensor(torch.zeros(1,25),requires_grad=True)
            print(i,end=" ")
            for word in sentence:
                try:
                    i = glove_vocab.index(word)
                    x = autoTensor(glove_embed[i].view(1,25))
                except :
                    x = autoTensor(glove_embed[0].view(1,25)*0)
                f = F.sigmoid(self.f(h,x))
                i = F.sigmoid(self.i(h,x))
                c_ = F.tanh(self.c_(h,x))
                o = F.sigmoid(self.o(h,x))
                self.c = f*self.c + i*c_
                h = o*F.tanh(self.c)
            s=time()
            z = F.sigmoid(self.out(h))
            loss = Loss.BinaryCrossEntropy(z,y_t)
            print(loss,z.value.item(),y_t.value.item())
            loss.backward()
            op = Optimizer("sgd",loss,0.0001)
            op.step()
            gc.collect()
            print(time()-s,end=" ")

if __name__ == "__main__":
    
    file_name = "testdata/amzn.txt"
    X,y = getXY(file_name)

    print(X[50],y[50],y.shape,len(X))
    glove_vocab,glove_embed = loadGloveModel("testdata/glove.twitter.27B.25d.txt")

    print(torch.Tensor(glove_embed).size())

    l = lstm()
    l.forward(X,y,glove_vocab,torch.Tensor(glove_embed))
