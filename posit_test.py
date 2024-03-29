import torch
import torch.nn as nn
import math
import numpy as np
torch.manual_seed(0)
from NLayerNet import *
from plot_distrib import plot_pos
from finderror import find_error

torch.ops.load_library("distrib/build/libdistrib.so") #disable when not in use

N = 64 #Batch size

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    actual = torch.argmax(yb, dim=1)
    return (preds == actual).float().mean()

layers = [784,100,64,10] #Layer sizes
activ = ['relu','linear','logsoftmax'] #Layer Activations

#DATASET - MNIST
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

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
y_train = torch.nn.functional.one_hot(y_train).to(torch.float32)
y_valid = torch.nn.functional.one_hot(y_valid).to(torch.float32)
n, c = x_train.shape


model = NLayerNet(layers,activ,False)
print(model)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-3)
n = x_train.shape[0]



for t in range((n - 1) // N + 1):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train[t*N:t*N+N])
    loss = criterion(y_pred, y_train[t*N:t*N+N])
    if t % 100 == 99:
        print(t, loss.item())
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print('Training Accuracy: ',accuracy(model(x_train),y_train))
print('Validation Accuracy: ',accuracy(model(x_valid[0:128*N]),y_valid[0:128*N]))

#Get layer weights
weights = []
weights_ = [] #for error calc
bias = []
#bias_posit = []
for i in range(len(layers)-1):
    w = model.linear[i].weight.t()
    weights.append(model.linear[i].weight.t())
    weights_.append(torch.reshape(w,(1,w.shape[0]*w.shape[1])))
    b = model.linear[i].bias.t()
    bias.append(torch.reshape(b,(1,b.shape[0])))
   
#POSIT INFERENCE
from datetime import datetime
start=datetime.now()

test_model = PositLayerNet(layers,activ,weights,bias)

posit_accuracy = accuracy(test_model(x_valid[0:128*N]),y_valid[0:128*N]) #Make sure this is a multiple of N
print('Accuracy of Posit: ',posit_accuracy)
print('time elapsed: ',(datetime.now()-start).seconds)

#CONVERT WEIGHTS & BIAS TO POSIT
bias_posit = []
weights_posit = []
all_fp = []
all_posit = []
bias_posit.append(torch.ops.my_ops.distrib(bias[0], 1, bias[0].shape[1]))
bias_posit.append(torch.ops.my_ops.distrib(bias[1], 1, bias[1].shape[1]))
bias_posit.append(torch.ops.my_ops.distrib(bias[2], 1, bias[2].shape[1]))

weights_posit.append(torch.ops.my_ops.distrib(weights_[0], 1, weights_[0].shape[1]))
weights_posit.append(torch.ops.my_ops.distrib(weights_[1], 1, weights_[1].shape[1]))
weights_posit.append(torch.ops.my_ops.distrib(weights_[2], 1, weights_[2].shape[1]))
for i in range(3):
    all_fp.append(bias[i].detach().numpy()[0])
    all_fp.append(weights_[i].detach().numpy()[0])
    all_posit.append(bias_posit[i].detach().numpy()[0])
    all_posit.append(weights_posit[i].detach().numpy()[0])   

#PLOTTING DISTRIBUTIONS
plot_pos(all_fp,all_posit,8,0) #change n and es here and in distrib/op.cpp
#CALCULATE AVG RELATIVE ERROR
find_error(bias,bias_posit,weights_,weights_posit)







