import torch
import torch.nn as nn
import math
import numpy as np
torch.manual_seed(0)
from NLayerNet import *

#torch.ops.load_library("distrib/build/libdistrib.so")

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
#f=open('/home/amritha/Project/Operator/result.txt','w')
posit_accuracy = accuracy(test_model(x_valid[0:128*N]),y_valid[0:128*N])
print('Accuracy of Posit: ',posit_accuracy)
print('time elapsed: ',(datetime.now()-start).seconds)

#f.write(str(posit_accuracy))
#f.close
#CONVERT WEIGHTS & BIAS TO POSIT
"""bias_posit = []
weights_posit = []
bias_posit.append(torch.ops.my_ops.distrib(bias[0], 1, bias[0].shape[1]))
bias_posit.append(torch.ops.my_ops.distrib(bias[1], 1, bias[1].shape[1]))
bias_posit.append(torch.ops.my_ops.distrib(bias[2], 1, bias[2].shape[1]))

weights_posit.append(torch.ops.my_ops.distrib(weights_[0], 1, weights_[0].shape[1]))
weights_posit.append(torch.ops.my_ops.distrib(weights_[1], 1, weights_[1].shape[1]))
weights_posit.append(torch.ops.my_ops.distrib(weights_[2], 1, weights_[2].shape[1]))
#PLOTTING DISTRIBUTIONS
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':200})
plt.figure()
plt.hist(bias[0].detach().numpy()[0], alpha = 0.5, bins = 50, color='b', label = 'FP')
plt.hist(bias_posit[0].detach().numpy()[0],alpha = 0.5,  bins = 50, color='r',label = "posit(8,2)")
plt.savefig("Distrib.png")
"""
#CALCULATE AVG RELATIVE ERROR
"""
error = []
for i in range(3):
    sum = 0
    for j in range(bias[i].shape[1]):
        diff= abs(bias[i].detach().numpy()[0][j]-bias_posit[i].detach().numpy()[0][j])/abs(bias[i].detach().numpy()[0][j])
        #if (diff>1):
           #print(bias[i].detach().numpy()[0][j])
           #print(bias_posit[i].detach().numpy()[0][j])
        sum = sum + diff
    sum = sum / bias[i].shape[1]
    error.append(sum)
print('Average Relative Error(Bias): ',(error[0]+error[1]+error[2])/3)
#print(error)
error = []
for i in range(3):
    sum = 0
    for j in range(weights_[i].shape[1]):
        sum = sum + abs(weights_[i].detach().numpy()[0][j]-weights_posit[i].detach().numpy()[0][j])/abs(weights_[i].detach().numpy()[0][j])
    sum = sum / weights_[i].shape[1]
    error.append(sum)


print('Average Relative Error(Weights): ',(error[0]+error[1]+error[2])/3)

"""
