import torch
import string
import numpy as np
import scipy.io as mat
from autodiff import autoTensor
from neural import Loss, Weight, Initializer, Linear, Optimizer
import autodiff.functional as F
import torch.nn.init as torchInit
import matplotlib.pyplot as plt
import gc


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
    embedding_dict = {}
    
    file = open(filename,'r',encoding='UTF-8')
    
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab_word = row[0]
        glove_vocab.append(vocab_word)
        embed_vector = [float(i) for i in row[1:]] # convert to list of float
        embedding_dict[vocab_word]=embed_vector
        glove_embed.append(embed_vector)
    
    print('Loaded GLOVE')
    file.close()

    return glove_vocab,glove_embed,embedding_dict

#######################################################################################

class lstm(object):

    def __init__(self):
        init = Initializer("xavier")
        self.fh = Linear(25,25,init)
        self.ih = Linear(25,25,init)
        self.ch_ = Linear(25,25,init)
        self.oh = Linear(25,25,init)
        self.fx = Linear(25,25,init,bias=False)
        self.ix = Linear(25,25,init,bias=False)
        self.cx_ = Linear(25,25,init,bias=False)
        self.ox = Linear(25,25,init,bias=False) 

if __name__ == "__main__":
    
    file_name = "testdata/amzn.txt"
    X,y = getXY(file_name)

    print(X[50],y[50],y.shape,len(X))

    glove_vocab,glove_embed,embedding_dict = loadGloveModel("testdata/glove.twitter.27B.25d.txt")

    print(embedding_dict["im"])
