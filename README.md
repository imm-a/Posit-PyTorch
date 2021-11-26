# Posit-PyTorch
Current status: Posit inference enabled using lookup tables. Parallel computation enabled using OpenMP.

To be improved/fixed:
 - Lookup table values are const int in mat_mul/op.cpp and posit_add/op.cpp which have to be changed manually
 - Eliminate redundant quantization
