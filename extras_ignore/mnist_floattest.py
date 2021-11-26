

from pathlib import Path
import requests

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
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)
def model(xb):
    m = torch.nn.ReLU()
    return torch.sigmoid(xb @ weights + bias)
bs = 64
#xb = x_train[0:bs]  # a mini-batch from x
#preds = model(xb,bs,784,10)  # predictions
#preds[0], preds.shape
def model_pos(xb):
    m = torch.nn.ReLU()
    mul = torch.ops.my_ops.pos_mul(xb,weights,bs,784,10)
    #print("mul: ",mul[0])
    add_bias = mul + bias #change to posit addition
    return torch.sigmoid(add_bias)
#print(preds[0], preds.shape)
def error(a,y):
    return (a-y)
def relu(x):
    y = x.detach.numpy()
    return (y>0)*x
def drelu(x):
    if x>0:
       return 1
    else:
       return 0
def backward(a,x,y):
    dSig = drelu(x)
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
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        print(yb.shape)
        #print('pred:',pred)
        loss = loss_func(pred, yb)
        #print(loss)
        weights.retain_grad()
        loss.backward()
        #print(weights.grad)
        with torch.no_grad():
     
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
print('done')
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
print(weights)
print("float: ",loss_func(model(x_valid[0:bs]), y_valid[0:bs]), accuracy(model(x_valid[0:bs]), y_valid[0:bs]))
x_ = model_pos(x_valid[0:bs])
print("posit: ",loss_func(x_, y_valid[0:bs]), accuracy(x_, y_valid[0:bs]))
#print("flmul:", (x_valid[0:bs] @ weights)[0])
print(model(x_valid[0:bs])[10])
print(x_[10])
