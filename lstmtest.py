import torch
import string
import numpy as np
import scipy.io as mat
from autosense.autodiff import autoTensor
from autosense.neural import Loss, Weight, Initializer, Linear, Optimizer, Linear2
import autosense.autodiff.functional as F
import torch.nn.init as torchInit
import matplotlib.pyplot as plt
import gc
from time import time
from autosense.models.rnn import LSTMnode

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
        self.node = LSTMnode(25,25,25,init)

        self.out = Linear(25,1,init)


    def forward(self,X,y,glove_vocab,glove_embed):
        
        for i,sentence in enumerate(X):
            self.node.c=autoTensor(torch.zeros(1,25))
            y_t = autoTensor(y[i])
            h = autoTensor(torch.zeros(1,25),requires_grad=True)
            print(i,end=" ")
            for word in sentence:
                try:
                    i = glove_vocab.index(word)
                    x = autoTensor(glove_embed[i].view(1,25))
                except :
                    x = autoTensor(glove_embed[0].view(1,25)*0)
                h=self.node.forward(h,x)
            s=time()
            z = F.sigmoid(self.out(h))
            loss = Loss.BinaryCrossEntropy(z,y_t)
            print(loss,z.value.item(),y_t.value.item())
            loss.backward()
            op = Optimizer("sgd",loss,0.0001)
            op.step()
            gc.collect()
            print(time()-s,end=" ")

    def test(self,glove_vocab,glove_embed):
        test1 = ["its","a","good","product"]
        test2 = ["This","is","bad","indeed"]

        print("\n\nTest 1: Its a good product",end=": ")
        #test 1
        self.node.c=autoTensor(torch.zeros(1,25))
        h = autoTensor(torch.zeros(1,25))
        for word in test1:
            try:
                index = glove_vocab.index(word)
                sub_x = glove_embed[index].view(1,25)
            except :
                sub_x = glove_embed[0].view(1,25)*0
            x = autoTensor(sub_x)
            h=self.node.forward(h,x,j)
        z = F.sigmoid(self.out(h))
        print(z)

        print("\n\nTest 2: [This,is,bad,indeed]",end=": ")
        #test 1
        self.node.c=autoTensor(torch.zeros(1,25))
        h = autoTensor(torch.zeros(1,25))
        for word in test2:
            try:
                index = glove_vocab.index(word)
                sub_x = glove_embed[index].view(1,25)
            except :
                sub_x = glove_embed[0].view(1,25)*0
            x = autoTensor(sub_x)
            h=self.node.forward(h,x,j)
        z = F.sigmoid(self.out(h))
        print(z)

    def run(self,X,y,glove_vocab,glove_embed):
        for epoch in range(115):
            print("epoch: ",epoch)
            epoch_loss = 0
            acc = 0
            for i in range(len(X)):
                if i == 6:
                    break
                self.node.c=autoTensor(torch.zeros(1,25))
                y_t = autoTensor(y[i])
                h = autoTensor(torch.zeros(1,25),requires_grad=True)

                batch = X[i]
                print("\t",i,len(batch[0]),end=" ")

                t = time()
                for j in range(len(batch[0])):
                    for k in range(len(batch)):
                        try:
                            word = str(batch[k][j])
                            index = glove_vocab.index(word)
                            sub_x = glove_embed[index].view(1,25)
                        except :
                            sub_x = glove_embed[0].view(1,25)*0
                        if k == 0:
                            x = sub_x
                        else:
                            x = torch.cat((x,sub_x))

                    x = autoTensor(x)
                    
                    h=self.node.forward(h,x,j)
                print("t:",round(time()-t,4),end="\t")

                s=time()
                z = F.sigmoid(self.out(h))
                loss = Loss.BinaryCrossEntropy(z,y_t)
                print(loss,(get_accuracy_value(z,y_t),loss.value.size()[0]),end=" ")
                
                #loss.trace_backprop()
                #loss.reset_count()
                loss.backward()
                op = Optimizer("sgd",loss,0.00005)
                op.step()
                gc.collect()
                print(round(time()-s,4))
                epoch_loss+= loss.single()
                acc += get_accuracy_value(z,y_t)
            print("\nepoch summary: loss: ",epoch_loss/5," ACC: ",acc/316)
                
def get_accuracy_value(pred, y):
    a = pred.value.clone()
    a[a >= 0.5 ]=1
    a[a!=1] = 0
    return abs(torch.sum(y.value == a).item())



if __name__ == "__main__":
    
    file_name = "testdata/amzn.txt"
    X,y = getXY(file_name)

    print(X[50],y[50],y.shape,len(X))
    glove_vocab,glove_embed = loadGloveModel("testdata/ignore/glove.twitter.27B.25d.txt")
    glove_embed = torch.Tensor(glove_embed).cuda()
    print(glove_embed.size())

    X_list = []
    y_list = []
    xlen = 0
    subl=[]
    subly=[]
    for i,x in enumerate(X):
        if len(x) == xlen:
            subl.append(x)
            subly.append(y[i])
        else:
            if len(subl) != 0:
                X_list.append(subl)
                y_list.append(subly)
            subl = []
            subl.append(x)
            subly.append(y[i])
            xlen = len(x)

    suby=[]
    for i,x in enumerate(X_list):
        y_s = np.zeros((len(x),1))
        for j,ins in enumerate(x):
            y_s[j] = y_list[i][j]

        suby.append(y_s)

    l = lstm()
    l.run(X_list,suby,glove_vocab,glove_embed)
    l.test(glove_vocab,glove_embed)
