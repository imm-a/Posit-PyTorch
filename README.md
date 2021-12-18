# Posit-PyTorch
Posit-PyTorch is a framework to carry out Neural Network inference using various Posit operators. The hardware is based on [smallPosit HDL](https://github.com/starbrilliance/SmallPositHDL).

**C++ operators: mat_mul and posit_add**

**Important: Clone Universal separately if it doesn't work**

NLayerNet: Contains the linear layer classes for Floating point and Posit

posit_test: Contains MNIST training and inference

Using: https://github.com/stillwater-sc/universal

```
git clone https://github.com/stillwater-sc/universal.git
```
Dependencies:
1. PyTorch: Install with appropriate settings from https://pytorch.org/
2. OpenMP
3. NumPy


***Current status:*** 
- Posit inference enabled using Linear layers and lookup tables. 
- Parallel computation enabled using OpenMP. 
- Accurate and 1-bit removed Approx Posit Multiplier available (set parameters in NLayerNet)

To run:
 1. Edit the lists with layers and activations in posit_test.py
 2. In NLayerNet: Change parameters such as n_mult, n_add and approx_type as required. (n_mult = number of parallel multiplications, n_add = number of parallel additions, approx_type = 0 for accurate and 1 for reduced precision)
 4. To change (N,ES) configuration:
    - Edit the "n1" and "es1" assignments in both functions in mat_mul/op.cpp. 
    - The first time, delete the existing build directory and make a new one inside the mat_mul and posit_add directories.
      ```
      rm -r build
      mkdir build
      ```
    - To sync changes
      ```
      cd mat_mul/build
      cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
      make -j
      ```
    - Run ``` make -j ``` everytime a change is made to the C++ files. Repeat the same for "posit_add".
  3. Run
     ```
     python posit_test.py
     ```

To be improved/fixed:
 - Lookup table values are const int in mat_mul/op.cpp and posit_add/op.cpp which have to be changed manually
 - Using operators directly instead of lookup tables
 - Improve framework to set layerwise approximations
 - Convolution 
