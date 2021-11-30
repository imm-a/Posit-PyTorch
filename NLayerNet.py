import torch
import torch.nn as nn
import math
import numpy as np
from posit_mac import posit_forward
torch.ops.load_library("mat_mul/build/libmat_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")
torch.manual_seed(0)  


class NLayerNet(torch.nn.Module):
    def __init__(self, layers,activ,approx_type=0):

        super(NLayerNet, self).__init__()
        self.layers = layers
        self.activation = activ
        self.number = len(layers)

        self.approx_type=approx_type
        self.linear = []
        for i in range(self.number-1):
            self.linear.append(torch.nn.Linear(layers[i],layers[i+1]))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, x):
        for i in range(self.number-1):
            #linear_ = torch.nn.Linear(self.layers[i], self.layers[i+1])
            
            linear = self.linear[i](x)

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



class PositLayerNet(torch.nn.Module):
    def __init__(self, layers,activ,weights,bias,approx_type=0):

        super(PositLayerNet, self).__init__()
        self.layers = layers
        self.activation = activ
        self.number = len(layers)
        self.approx_type=approx_type
        self.weights = weights
        self.bias = bias
        self.linear = []
        for i in range(self.number-1):
            self.linear.append(torch.nn.Linear(layers[i],layers[i+1]))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, x):
        for i in range(self.number-1):
            linear = posit_forward(x,i,self.weights,self.bias,self.approx_type,16,16)
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