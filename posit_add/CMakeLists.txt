cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(warp_perspective)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_C_FLAGS}")
# Define our library target
add_library(posit_add SHARED op.cpp)
# Enable C++14
target_compile_features(posit_add PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(posit_add "${TORCH_LIBRARIES}" "${OpenMP_CXX_FLAGS}")

