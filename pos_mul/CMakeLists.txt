cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pos_mul)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Define our library target
add_library(pos_mul SHARED op.cpp)
# Enable C++14
target_compile_features(pos_mul PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(pos_mul "${TORCH_LIBRARIES}")

