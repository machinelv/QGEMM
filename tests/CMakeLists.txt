cmake_minimum_required(VERSION 3.21)

# Set build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Project configuration for CUDA
project(CUDA_GEMM_Benchmark LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures")

# Find packages
find_package(CUDAToolkit REQUIRED)

set(CUTLASS_INCLUDE_DIR )

# Define preprocessor macro
add_definitions(-DTEST_ON_CUDA)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Find CUTLASS
find_path(CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/ref/cutlass/include
        /usr/local/include
        /opt/cutlass/include
    NO_DEFAULT_PATH
)

if(NOT CUTLASS_INCLUDE_DIR)
    message(WARNING "CUTLASS not found. Trying to use system installation.")
    find_path(CUTLASS_INCLUDE_DIR
        NAMES cutlass/cutlass.h
        PATHS /usr/local/include /opt/cutlass/include
    )
endif()

if(CUTLASS_INCLUDE_DIR)
    message(STATUS "Found CUTLASS: ${CUTLASS_INCLUDE_DIR}")
    include_directories(${CUTLASS_INCLUDE_DIR})
else()
    message(FATAL_ERROR "CUTLASS not found. Please install CUTLASS or set CUTLASS_INCLUDE_DIR.")
endif()

# Source files
set(GEMM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gemm/test_cutlass_gemm.cpp
)

# Create a library for GEMM functions
add_library(gemm_lib ${GEMM_SOURCES})

# Set properties for the library
set_target_properties(gemm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link CUDA libraries to the GEMM library
target_link_libraries(gemm_lib
    CUDA::cudart
    CUDA::cublas
)

# Test executable
add_executable(cuda_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cuda_benchmark.cpp
)

# Set properties for the executable
set_target_properties(cuda_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Link libraries to the executable
target_link_libraries(cuda_benchmark
    gemm_lib
    CUDA::cudart
    CUDA::cublas
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(cuda_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
    )
endif()

# Optional: Create a simple test target
enable_testing()
add_test(NAME cuda_benchmark_test COMMAND cuda_benchmark)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
