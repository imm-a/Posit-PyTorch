import math
import torch.nn as nn
import numpy as np
import math

import torch
torch.ops.load_library("pos_mul/build/libpos_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")

class MyLinearLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size_in, size_out,weights,bias,posit):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.zeros(size_out, requires_grad=True)
        self.bias = nn.Parameter(bias)
        self.posit = posit
        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        if(self.posit == True):
          mul = torch.ops.my_ops.pos_mul(x,self.weights,bs,self.size_in,self.size_out)
          add_bias = mul + self.bias
        #w_times_x= x @ self.weights + self.bias
        else:
          add_bias = x @ self.weights + self.bias
        return add_bias  # w times x + b

sizes = [784,300,10]        
weights1 = torch.randn(sizes[0], sizes[1]) / math.sqrt(sizes[0])
weights1.requires_grad_()
bias1 = torch.zeros(sizes[1], requires_grad=True)
weights2 = torch.randn(sizes[1], sizes[2]) / math.sqrt(sizes[0])
weights2.requires_grad_()
bias2 = torch.zeros(sizes[2], requires_grad=True)
class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear = nn.Linear(256, 2)
        self.linear1 = MyLinearLayer(sizes[0], sizes[1],weights1,bias1,False)
        self.linear2 = MyLinearLayer(sizes[1], sizes[2],weights2,bias2,False)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        #x = x.view(-1, 256)
        m = torch.nn.ReLU()
        return torch.sigmoid(self.linear2(x))


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
torch.manual_seed(0)   
from pathlib import Path
import requests
basic_model = BasicModel()
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

#pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
#print(x_train.shape)


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#print(x_train, y_train)
#print(x_train.shape)
#print(y_train.min(), y_train.max())
bs = 32
#print(basic_model(x_valid[0:bs]))

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(basic_model.parameters(), lr=0.5)

#for t in range(500):
 #   # Forward pass: Compute predicted y by passing x to the model
  #  y_pred = model(x)

    # Compute and print loss
#    loss = criterion(y_pred, y)
 #   if t % 100 == 99:
 #       print(t, loss.item())
#
    # Zero gradients, perform a backward pass, and update the weights.
  #  optimizer.zero_grad()
   # loss.backward()
    #optimizer.step()


lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        y_pred = basic_model(xb)
        #print(y_pred.shape)
        #print(yb.shape)
        loss = loss_func(y_pred,yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print(accuracy(basic_model(xb),yb))
