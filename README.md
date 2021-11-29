# Posit-PyTorch
C++ operators: mat_mul and posit_add
NLayerNet: Contains the linear layer classes for Floating point and Posit
posit_test: Contains MNIST training and inference
Using: https://github.com/stillwater-sc/universal

Current status: Posit inference enabled using Linear layers and lookup tables. Parallel computation enabled using OpenMP. Accurate and 1-bit removed Approx Posit Multiplier available.

To run:
 - Edit lists with layers and activations in mnist_posit.py
 - python3 mnist_posit.py

To be improved/fixed:
 - Lookup table values are const int in mat_mul/op.cpp and posit_add/op.cpp which have to be changed manually
 - Eliminate redundant quantization
 - Using operators directly instead of lookup tables
 - Improve framework to set parameters such as approximations for layers
 - Convolution 
