cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_suite LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")

include_directories(include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Use nvcc
set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

add_executable(gemm_suite src/main.cpp)

set_target_properties(gemm_suite PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
