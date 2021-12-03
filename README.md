# Posit-PyTorch
C++ operators: mat_mul and posit_add

NLayerNet: Contains the linear layer classes for Floating point and Posit

posit_test: Contains MNIST training and inference

Using: https://github.com/stillwater-sc/universal

Current status: Posit inference enabled using Linear layers and lookup tables. Parallel computation enabled using OpenMP. Accurate and 1-bit removed Approx Posit Multiplier available.

To run:
 1. Edit the lists with layers and activations in posit_test.py
 2. To change (N,ES) configuration:
    - Edit the "n1" and "es1" assignments in both functions in mat_mul/op.cpp. 
    - To sync changes
      ```
      cd mat_mul/build
      cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
      make -j
      ```
    - These commands have to be run everytime a change is made to the C++ files. Repear the same for "posit_add".
  3. Run
     ```
     python posit_test.py
     ```

To be improved/fixed:
 - Lookup table values are const int in mat_mul/op.cpp and posit_add/op.cpp which have to be changed manually
 - Using operators directly instead of lookup tables
 - Improve framework to set parameters such as approximations for layers
 - Convolution 
