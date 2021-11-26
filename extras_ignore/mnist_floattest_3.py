import torch
import torch.nn as nn
import math
torch.ops.load_library("pos_mul/build/libpos_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    actual = torch.argmax(yb, dim=1)
    return (preds == actual).float().mean()
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 784, 200, 10

# Create random Tensors to hold inputs and outputs
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
model = TwoLayerNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
for t in range(500):
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
print(accuracy(model(x_train[0:N]),y_train[0:N]))
print(accuracy(model(x_valid),y_valid))
w1 = model.linear1.weight.t()
b1 = model.linear1.bias
w2 = model.linear2.weight.t()
b2 = model.linear2.bias
###POSIT INFERENCE
class PositLinear(nn.Module):
    #Custom Linear layer but mimics a standard linear layer
    def __init__(self, size_in, size_out,weights,bias,posit):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.weights = weights  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = bias
        self.posit = posit

    def forward(self, x):
        if(self.posit == True):
          mul = torch.ops.my_ops.pos_mul(x,self.weights,N,self.size_in,self.size_out)
          add_bias = mul + self.bias
        #w_times_x= x @ self.weights + self.bias
        else:
          add_bias = x @ self.weights + self.bias
        return add_bias  # w times x + b

class PositModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear = nn.Linear(256, 2)
        self.linear1 = PositLinear(D_in, H,w1,b1,True)
        self.linear2 = PositLinear(H, D_out,w2,b2,True)

    def forward(self, x):
        x = self.linear1(x).clamp(min=0)
        #x = x.view(-1, 256)
        #m = torch.nn.ReLU()
        return self.linear2(x)
posit_model = PositModel()

print(accuracy(posit_model(x_train[0:N]),y_train[0:N]))
        
