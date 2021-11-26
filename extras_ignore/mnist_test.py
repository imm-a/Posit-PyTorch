

from pathlib import Path
import requests
import numpy as np
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
        
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
        
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

import torch
torch.ops.load_library("pos_mul/build/libpos_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#print(x_train, y_train)
print(x_train.shape)
#print(y_train.min(), y_train.max())


import math

weights = torch.randn(784, 10) / math.sqrt(784)
x_train.requires_grad_()
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)
def model(xb,dim1,dim2,dim3):
    mul = torch.ops.my_ops.pos_mul(xb, weights,dim1,dim2,dim3)
    add_bias = mul + bias #change to posit addition
    return torch.sigmoid(add_bias)
bs = 64
#xb = x_train[0:bs]  # a mini-batch from x
#preds = model(xb,bs,784,10)  # predictions
#preds[0], preds.shape

#print(preds[0], preds.shape)
def error(a,y):
    return (a-y)
def relu(x):
    return (x>0)*x
def backward(a,x):
    dSig = (x>0)
    dW = torch.matmul(x,dSig)
    db = torch.sum(dSig,0)
    return dW,db

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

#yb = y_train[0:bs]
#print(loss_func(preds, yb))

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
    
    
#print(accuracy(preds, yb))


#from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 1  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb,bs,784,10)
        loss = loss_func(pred, yb)
        print('pred:',pred[0])
        #print(loss)
        #print(pred)
        #weights.retain_grad()
        dW,db = backward(pred,torch.transpose(xb,0,1))
        weights = weights - lr*dW
        bias = bias - lr*db
print('done')
print(loss_func(model(xb,bs,784,10), yb), accuracy(model(xb,bs,784,10), yb))
