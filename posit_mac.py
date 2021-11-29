import torch
import torch.nn as nn
torch.ops.load_library("mat_mul/build/libmat_mul.so")
torch.ops.load_library("posit_add/build/libposit_add.so")
torch.manual_seed(0)  


def posit_forward(x,layer_no,weights,bias,approx_type,n_mult,n_add):
    #approx_type: currently refers to the type of approximate multiplier - 0 for accurate and 1 for approximate
    #n_mult: number of multipliers
    #n_add: number of adders
    mul = torch.ops.my_ops.mat_mul(x,weights[layer_no],x.shape[0],x.shape[1],weights[layer_no].shape[1],n_mult,approx_type)
    add_bias = torch.ops.my_ops.posit_add(mul,bias[layer_no],mul.shape[0],mul.shape[1],n_add,0)
    return add_bias