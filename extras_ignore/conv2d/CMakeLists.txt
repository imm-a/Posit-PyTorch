cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(conv2d)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
#find_package(UNIVERSAL CONFIG REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ${OpenMP_C_FLAGS}")
# Define our library target
include_directories("${CMAKE_CURRENT_LIST_DIR}/../universal/include")
add_library(conv2d SHARED op.cpp)
# Enable C++14
target_compile_features(conv2d PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(conv2d "${TORCH_LIBRARIES}" "${OpenMP_CXX_FLAGS}")

