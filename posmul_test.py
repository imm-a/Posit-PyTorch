import torch
torch.manual_seed(0) 
torch.ops.load_library("mat_mul/build/libmat_mul.so")
#torch.ops.load_library("pos_mul/build/libpos_mul.so")
a = torch.randn((3, 4),requires_grad=True)
b = torch.rand((4, 3),requires_grad=True)
print(a)
print(b)
print("Multiplying...")
print('1: ',torch.ops.my_ops.mat_mul(a, b,3,4,3,3,0))
print('\n')
print('2: ',a @ b)
print('\n')
print('3: ',torch.matmul(a,b))

#def compute(a,b):
 #   x = torch.ops.my_ops.pos_mul(a, b,3,4,3)
  #  return x
#print('1: ',torch.ops.my_ops.pos_mul(a, b,3,4,3))    

#inputs = [a, b]
#trace = torch.jit.trace(compute, inputs)
#print(trace.graph)
