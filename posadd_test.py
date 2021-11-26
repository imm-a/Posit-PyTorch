import torch
torch.ops.load_library("posit_add/build/libposit_add.so")
a = torch.randn(3, 3)*10
b = torch.rand(1,3)*10
print(a)
print(b)
print("Adding...")
print(torch.ops.my_ops.posit_add(a, b,3,3,1))
print("\n")
print(a+b)
