import torch
torch.manual_seed(0) 
torch.ops.load_library("conv2d/build/libconv2d.so")
#torch.ops.load_library("pos_mul/build/libpos_mul.so")
a = torch.ones((3, 3),requires_grad=True)
b = torch.rand((5, 5),requires_grad=True)
print(a)
print(b)
print("Multiplying...")
print('1: ',torch.ops.my_ops.conv2d(b,a,5,3,1,0))
print('\n')
#print('2: ',a @ b)
#print('\n')
#print('3: ',torch.matmul(a,b))

#def compute(a,b):
 #   x = torch.ops.my_ops.pos_mul(a, b,3,4,3)
  #  return x
#print('1: ',torch.ops.my_ops.pos_mul(a, b,3,4,3))    

#inputs = [a, b]
#trace = torch.jit.trace(compute, inputs)
#print(trace.graph)
