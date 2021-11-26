import torch
import torch.nn as nn
import math
import numpy as np
torch.ops.load_library("mat_mul/build/libmat_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")
torch.manual_seed(0)  

N = 64 #Batch size


def posit_forward(x,i,approx_type):
    # add_bias = torch.zeros(x.shape[0],weights[i].shape[1])
    # for n in range((x.shape[0]-1)//N+1):
    #     mul = torch.ops.my_ops.mat_mul(x[n*N:n*N+N],weights[i],N,x.shape[1],weights[i].shape[1])
    #     add_bias1 = mul + bias[i]
    #     add_bias[n*N:n*N+N] = add_bias1
    mul = torch.ops.my_ops.mat_mul(x,weights[i],x.shape[0],x.shape[1],weights[i].shape[1],8,approx_type)
    
    add_bias = torch.ops.my_ops.posit_add(mul,bias[i],mul.shape[0],mul.shape[1],8)
    #add_bias = mul + bias[i]
    return add_bias
class TwoLayerNet(torch.nn.Module):
    def __init__(self, layers,activ,isPosit,approx_type=0):

        super(TwoLayerNet, self).__init__()
        self.layers = layers
        self.activation = activ
        self.number = len(layers)
        self.isPosit = isPosit
        self.approx_type=approx_type
        self.linear = []
        for i in range(self.number-1):
            self.linear.append(torch.nn.Linear(layers[i],layers[i+1]))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, x):
        for i in range(self.number-1):
            #linear_ = torch.nn.Linear(self.layers[i], self.layers[i+1])
            if(self.isPosit==False):
                linear = self.linear[i](x)
            else:
                linear = posit_forward(x,i,self.approx_type)
            activ = linear
            if(self.activation[i]=='relu'):
                activ = linear.clamp(min=0)
            elif(self.activation[i]=='sigmoid'):
                activ =  torch.sigmoid(linear)
            elif(self.activation[i]=='logsoftmax'):
                #m = nn.LogSoftmax()
                activ = torch.nn.functional.log_softmax(linear)
            else:
                activ = linear
            x = activ
        return activ

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    actual = torch.argmax(yb, dim=1)
    return (preds == actual).float().mean()

layers = [784,100,64,10]
activ = ['relu','linear','logsoftmax']

#DATASETS
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


# Construct our model by instantiating the class defined above
model = TwoLayerNet(layers,activ,False)
print(model)
#print(model.parameters()))
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1.5e-3)
n = x_train.shape[0]
for t in range((n - 1) // N + 1):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train[t*N:t*N+N])
    #print(y_train[1])
    # Compute and print loss
    loss = criterion(y_pred, y_train[t*N:t*N+N])
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(accuracy(model(x_train),y_train))
print(accuracy(model(x_valid[0:16*N]),y_valid[0:16*N]))

#Get layer weights
weights = []
bias = []
for i in range(len(layers)-1):
    weights.append(model.linear[i].weight.t())
    b = model.linear[i].bias.t()
    bias.append(torch.reshape(b,(1,b.shape[0])))
    #bias.append(model.linear[i].bias.t())
#print(bias[0].shape)
#bias0 = torch.reshape(bias[0],(1,bias[0].shape[0]))
#print(bias0.shape)
from datetime import datetime
start=datetime.now()

test_model = TwoLayerNet(layers,activ,True)
print('Accuracy of Posit: ',accuracy(test_model(x_valid[0:16*N]),y_valid[0:16*N]))

print('time elapsed: ',(datetime.now()-start).seconds)